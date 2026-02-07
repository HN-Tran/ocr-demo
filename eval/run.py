from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from app.config import get_settings
from app.services.ocr_pipeline import OCRPipeline
from app.services.ollama_client import OllamaClient
from eval.metrics import cer, field_accuracy, wer


def _load_manifest(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(content)
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must contain a list.")
        return payload
    records: list[dict] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


async def _run_sample(pipeline: OCRPipeline, sample: dict, samples_dir: Path) -> dict:
    image_path = samples_dir / sample["sample"]
    image_bytes = image_path.read_bytes()
    mode = sample.get("mode", "plain")
    schema_name = sample.get("schema_name")

    result = await pipeline.run(
        image_bytes=image_bytes,
        mode=mode,
        schema_name=schema_name,
        model=sample.get("model"),
    )

    sample_report = {
        "sample": sample["sample"],
        "mode": mode,
        "schema_name": schema_name,
        "latency_ms": result.latency_ms,
        "warnings": result.warnings,
        "metrics": {},
    }

    if mode == "plain":
        reference_text = sample.get("text", "")
        sample_report["metrics"]["cer"] = cer(reference_text, result.text)
        sample_report["metrics"]["wer"] = wer(reference_text, result.text)
    else:
        reference_structured = sample.get("structured", {})
        sample_report["metrics"]["field_accuracy"] = field_accuracy(
            reference_structured,
            result.structured or {},
        )
    return sample_report


async def run_eval(manifest: Path, samples_dir: Path, reports_dir: Path) -> Path:
    settings = get_settings()
    pipeline = OCRPipeline(
        ollama_client=OllamaClient(
            base_url=settings.ollama_base_url,
            timeout_s=settings.request_timeout_s,
        ),
        default_model=settings.ollama_model,
        max_image_dim=settings.max_image_dim,
    )
    samples = _load_manifest(manifest)
    reports = []
    for sample in samples:
        reports.append(await _run_sample(pipeline, sample, samples_dir))

    aggregate: dict[str, float] = {}
    plain_cers = [r["metrics"]["cer"] for r in reports if "cer" in r["metrics"]]
    plain_wers = [r["metrics"]["wer"] for r in reports if "wer" in r["metrics"]]
    struct_acc = [
        r["metrics"]["field_accuracy"] for r in reports if "field_accuracy" in r["metrics"]
    ]
    if plain_cers:
        aggregate["mean_cer"] = mean(plain_cers)
    if plain_wers:
        aggregate["mean_wer"] = mean(plain_wers)
    if struct_acc:
        aggregate["mean_field_accuracy"] = mean(struct_acc)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = reports_dir / f"eval_report_{timestamp}.json"
    output_payload = {
        "timestamp_utc": timestamp,
        "manifest": str(manifest),
        "samples_dir": str(samples_dir),
        "aggregate": aggregate,
        "samples": reports,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR benchmark.")
    parser.add_argument(
        "--manifest",
        default="data/ground_truth/manifest.jsonl",
        help="Path to benchmark manifest (.json or .jsonl).",
    )
    parser.add_argument(
        "--samples-dir",
        default="data/samples",
        help="Directory containing sample images.",
    )
    parser.add_argument(
        "--reports-dir",
        default="eval/reports",
        help="Directory for output reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = asyncio.run(
        run_eval(
            manifest=Path(args.manifest),
            samples_dir=Path(args.samples_dir),
            reports_dir=Path(args.reports_dir),
        )
    )
    print(f"Wrote evaluation report: {path}")


if __name__ == "__main__":
    main()
