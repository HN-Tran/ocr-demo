"""Batch-Benchmarking — N Dateien × M Runner (lokale Modelle + Engines).

Phase 1: in-memory Job-Store, sequenzielles Abarbeiten der Paare. Jeder
Eintrag bekommt CER/WER/F1, falls eine Referenz mitgeliefert wurde,
plus Token-/Zeichen-Count und Latenz. Ein Job entspricht einem POST
auf ``/api/benchmark``; das Frontend pollt ``GET /api/benchmark/{id}``.

Konzepte:
- ``BenchmarkRow``  — eine Zeile in der Ergebnistabelle, also ein
  konkretes (Datei, Runner)-Paar.
- ``BenchmarkJob``  — eine Batch-Ausführung. Enthält Zeilen + Aggregate
  pro Runner.
- ``BenchmarkJobStore`` — ``asyncio.Lock`` + ``dict``, prozesslokal.
  Nicht persistent; Phase 2 hängt MLflow als Side-Channel dran.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from app.services.backend_router import OCRBackendRouter
from app.services.compare_engines import Engine
from app.services.compare_metrics import _tokenize, reference_only
from app.services.mlflow_sink import MlflowSink, disabled_sink

logger = logging.getLogger("ocr-demo.benchmark")

RowStatus = Literal["pending", "running", "done", "error"]
JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]


@dataclass
class BenchmarkRow:
    file_index: int
    file_name: str
    runner_kind: Literal["local_model", "engine"]
    runner_label: str
    status: RowStatus = "pending"
    text_chars: int = 0
    text_tokens: int = 0
    latency_ms: int = 0
    cer: float | None = None
    wer: float | None = None
    token_f1: float | None = None
    avg_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    text: str = ""
    reference: str = ""


@dataclass
class BenchmarkJob:
    id: str
    created_at: datetime
    status: JobStatus
    progress_done: int
    progress_total: int
    options: dict[str, Any]
    rows: list[BenchmarkRow] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    mlflow_run_id: str | None = None
    mlflow_run_url: str | None = None
    cancelled: bool = False


def _serialize_job(job: BenchmarkJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "created_at": job.created_at.isoformat(),
        "status": job.status,
        "progress": {"done": job.progress_done, "total": job.progress_total},
        "options": job.options,
        "rows": [asdict(r) for r in job.rows],
        "aggregate": job.aggregate,
        "error": job.error,
        "mlflow": {
            "run_id": job.mlflow_run_id,
            "run_url": job.mlflow_run_url,
        },
    }


class BenchmarkJobStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._jobs: dict[str, BenchmarkJob] = {}

    async def create(self, options: dict[str, Any], total: int) -> BenchmarkJob:
        async with self._lock:
            job = BenchmarkJob(
                id=uuid4().hex[:12],
                created_at=datetime.now(timezone.utc),
                status="queued",
                progress_done=0,
                progress_total=total,
                options=options,
            )
            self._jobs[job.id] = job
            return job

    async def get(self, job_id: str) -> BenchmarkJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def all(self) -> list[BenchmarkJob]:
        async with self._lock:
            return list(self._jobs.values())

    async def drop(self, job_id: str) -> bool:
        async with self._lock:
            return self._jobs.pop(job_id, None) is not None


# ---------------------------------------------------------------------------
# Runners — adapter-artige Hülle, damit local_model + engine gleich aussehen.
# ---------------------------------------------------------------------------


@dataclass
class _LocalModelRunner:
    label: str
    model: str
    pipeline: OCRBackendRouter
    backend: str | None = None
    assemble_from_regions: bool = False

    kind = "local_model"

    async def analyze(
        self, image_bytes: bytes, content_type: str
    ) -> tuple[str, list[dict[str, Any]], list[str], float | None]:
        result, _selected = await self.pipeline.run(
            backend=self.backend,
            image_bytes=image_bytes,
            content_type=content_type,
            mode="plain",
            schema_name=None,
            model=self.model,
            task="ocr_text",
            custom_prompt=None,
            token_limit=None,
            gif_max_frames=None,
            expert_assemble_from_regions=self.assemble_from_regions or None,
        )
        words = _flatten_word_polys(result.layout)
        return result.text, words, list(result.warnings or []), _avg_conf(words)


@dataclass
class _EngineRunner:
    label: str
    engine: Engine

    kind = "engine"

    async def analyze(
        self, image_bytes: bytes, content_type: str
    ) -> tuple[str, list[dict[str, Any]], list[str], float | None]:
        result = await self.engine.analyze(image_bytes, content_type)
        words = [w for page in result.words_per_page for w in page]
        return result.text, words, list(result.warnings), _avg_conf(words)


def _flatten_word_polys(layout: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not layout:
        return []
    flat: list[dict[str, Any]] = []
    for page in layout:
        wp = page.get("word_polys") if isinstance(page, dict) else None
        if isinstance(wp, list):
            flat.extend(w for w in wp if isinstance(w, dict))
    return flat


def _avg_conf(words: list[dict[str, Any]]) -> float | None:
    confs = [
        float(w["confidence"])
        for w in words
        if isinstance(w.get("confidence"), (int, float)) and float(w["confidence"]) > 0
    ]
    if not confs:
        return None
    return sum(confs) / len(confs)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


async def _cleanup_after(store: BenchmarkJobStore, job_id: str, delay_s: float) -> None:
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    await store.drop(job_id)


async def run_benchmark_job(
    *,
    job: BenchmarkJob,
    files: list[tuple[str, bytes, str]],  # (filename, bytes, content_type)
    references: list[str],
    runners: list[_LocalModelRunner | _EngineRunner],
    store: BenchmarkJobStore,
    mlflow_sink: MlflowSink | None = None,
    job_ttl_s: float = 3600.0,
) -> None:
    """Sequentielle Ausführung aller (Datei, Runner)-Paare.

    Sequenziell bewusst — parallel auf demselben Ollama bringt nichts (Modell
    wird ohnehin geladen-und-festgenagelt) und macht die Latenz-Messung wertlos.

    Wenn ``mlflow_sink`` aktiv ist, wird zusätzlich pro Job ein Parent-Run und
    pro (Datei, Runner)-Paar ein Nested-Run in MLflow geloggt — Hypothese und
    Referenz landen als Artefakte, Aggregate als Metriken am Parent.
    """
    sink = mlflow_sink or disabled_sink()
    job.status = "running"
    try:
        with sink.run(name=f"benchmark-{job.id}") as parent_run:
            if parent_run is not None:
                job.mlflow_run_id = getattr(parent_run.info, "run_id", None)
                job.mlflow_run_url = sink.run_url(job.mlflow_run_id)
            sink.log_params(
                {
                    "job_id": job.id,
                    "file_count": len(files),
                    "runner_count": len(runners),
                    "models": ",".join(job.options.get("models", [])),
                    "engines": ",".join(job.options.get("engines", [])),
                }
            )

            for f_idx, (name, content, ctype) in enumerate(files):
                if job.cancelled:
                    break
                ref = references[f_idx] if f_idx < len(references) else ""
                for runner in runners:
                    if job.cancelled:
                        break
                    row = BenchmarkRow(
                        file_index=f_idx,
                        file_name=name,
                        runner_kind=runner.kind,  # type: ignore[arg-type]
                        runner_label=runner.label,
                        status="running",
                    )
                    job.rows.append(row)
                    started = time.perf_counter()
                    text = ""
                    try:
                        text, _words, warnings, avg_conf = await runner.analyze(content, ctype)
                        row.text = text
                        row.text_chars = len(text)
                        row.text_tokens = len(_tokenize(text))
                        row.latency_ms = int((time.perf_counter() - started) * 1000)
                        row.warnings = warnings
                        row.avg_confidence = avg_conf
                        if ref.strip():
                            row.reference = ref
                            ref_block = reference_only(ref, text)["ours"]
                            row.cer = float(ref_block["cer"])
                            row.wer = float(ref_block["wer"])
                            row.token_f1 = float(ref_block["token_f1"])
                        row.status = "done"
                    except Exception as exc:  # noqa: BLE001
                        row.latency_ms = int((time.perf_counter() - started) * 1000)
                        row.status = "error"
                        row.error = f"{type(exc).__name__}: {exc}"
                        logger.warning(
                            "Benchmark-Zeile fehlgeschlagen (job=%s, file=%s, runner=%s): %s",
                            job.id,
                            name,
                            runner.label,
                            exc,
                        )
                    job.progress_done += 1

                    # Per-(Datei, Runner) als Nested-Run in MLflow.
                    with sink.run(name=f"{name}/{runner.label}", nested=True):
                        sink.log_params(
                            {
                                "file_index": f_idx,
                                "file_name": name,
                                "runner_kind": row.runner_kind,
                                "runner_label": row.runner_label,
                                "has_reference": bool(ref.strip()),
                            }
                        )
                        sink.log_metrics(
                            {
                                "text_chars": row.text_chars,
                                "text_tokens": row.text_tokens,
                                "latency_ms": row.latency_ms,
                                "cer": row.cer,
                                "wer": row.wer,
                                "token_f1": row.token_f1,
                                "avg_confidence": row.avg_confidence,
                            }
                        )
                        if text:
                            sink.log_text(text, "hypothesis.txt")
                        if ref.strip():
                            sink.log_text(ref, "reference.txt")
                        if row.error:
                            sink.log_text(row.error, "error.txt")

            job.aggregate = _aggregate(job.rows)

            # Aggregat als flache Metriken am Parent loggen.
            for label, stats in job.aggregate.get("per_runner", {}).items():
                slug = _label_slug(label)
                sink.log_metrics(
                    {
                        f"{slug}.mean_cer": stats.get("mean_cer"),
                        f"{slug}.mean_wer": stats.get("mean_wer"),
                        f"{slug}.mean_token_f1": stats.get("mean_token_f1"),
                        f"{slug}.mean_latency_ms": stats.get("mean_latency_ms"),
                        f"{slug}.success_count": stats.get("success_count"),
                        f"{slug}.failure_count": stats.get("failure_count"),
                    }
                )

        job.status = "cancelled" if job.cancelled else "done"
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Benchmark-Job %s abgebrochen", job.id)
    finally:
        async with store._lock:  # noqa: SLF001 — same-module access
            store._jobs[job.id] = job
        ttl = 0.0 if job.cancelled else job_ttl_s
        asyncio.create_task(_cleanup_after(store, job.id, ttl))


def _label_slug(label: str) -> str:
    """Make a label safe for use as part of an MLflow metric name."""
    return "".join(c if c.isalnum() else "_" for c in label).strip("_") or "runner"


def _aggregate(rows: list[BenchmarkRow]) -> dict[str, Any]:
    by_runner: dict[str, list[BenchmarkRow]] = {}
    for row in rows:
        by_runner.setdefault(row.runner_label, []).append(row)
    out: dict[str, Any] = {"per_runner": {}}
    for label, group in by_runner.items():
        successes = [r for r in group if r.status == "done"]
        cers = [r.cer for r in successes if r.cer is not None]
        wers = [r.wer for r in successes if r.wer is not None]
        f1s = [r.token_f1 for r in successes if r.token_f1 is not None]
        latencies = [r.latency_ms for r in successes]
        out["per_runner"][label] = {
            "doc_count": len(group),
            "success_count": len(successes),
            "failure_count": len(group) - len(successes),
            "mean_cer": statistics.fmean(cers) if cers else None,
            "stdev_cer": statistics.stdev(cers) if len(cers) > 1 else None,
            "mean_wer": statistics.fmean(wers) if wers else None,
            "mean_token_f1": statistics.fmean(f1s) if f1s else None,
            "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        }
    return out


__all__ = [
    "BenchmarkJob",
    "BenchmarkJobStore",
    "BenchmarkRow",
    "_EngineRunner",
    "_LocalModelRunner",
    "_serialize_job",
    "run_benchmark_job",
]
