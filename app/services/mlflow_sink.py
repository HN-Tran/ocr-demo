"""MLflow als optionaler Side-Channel fürs Batch-Benchmark.

MLflow ist eine optionale Abhängigkeit (``pip install '.[mlflow]'``).
Ohne ``MLFLOW_TRACKING_URI`` oder bei fehlendem ``mlflow`` liefern alle
Funktionen leere Kontextmanager / Noops zurück — der Caller muss nichts
über Anwesenheit prüfen.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

logger = logging.getLogger("docread.mlflow")

try:
    import mlflow as _mlflow

    _MLFLOW_INSTALLED = True
except ImportError:
    _mlflow = None
    _MLFLOW_INSTALLED = False


class MlflowSink:
    """Dünner Wrapper. Wenn nicht aktiv, sind alle Methoden Noops."""

    def __init__(self, *, tracking_uri: str, experiment_name: str) -> None:
        self.enabled = bool(tracking_uri) and _MLFLOW_INSTALLED
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._experiment_id: str | None = None
        if self.enabled:
            try:
                _mlflow.set_tracking_uri(tracking_uri)
                exp = _mlflow.set_experiment(experiment_name)
                self._experiment_id = getattr(exp, "experiment_id", None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLflow-Setup fehlgeschlagen, schalte ab: %s", exc)
                self.enabled = False
        elif tracking_uri and not _MLFLOW_INSTALLED:
            logger.warning(
                "MLFLOW_TRACKING_URI gesetzt, aber mlflow nicht installiert. "
                "Mit pip install '.[mlflow]' nachziehen."
            )

    @contextmanager
    def run(self, *, name: str, nested: bool = False) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return
        with _mlflow.start_run(run_name=name, nested=nested) as active:
            yield active

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            # MLflow erlaubt keine None-Werte, mappen wir auf "".
            _mlflow.log_params({k: ("" if v is None else v) for k, v in params.items()})
        except Exception as exc:  # noqa: BLE001
            logger.debug("log_params fehlgeschlagen: %s", exc)

    def log_metrics(self, metrics: dict[str, float | None]) -> None:
        if not self.enabled:
            return
        try:
            _mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("log_metrics fehlgeschlagen: %s", exc)

    def log_text(self, content: str, artifact_path: str) -> None:
        if not self.enabled or not content:
            return
        try:
            _mlflow.log_text(content, artifact_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("log_text fehlgeschlagen: %s", exc)

    def run_url(self, run_id: str | None) -> str | None:
        """Best-effort URL — funktioniert für HTTP-Tracking-Server.

        ``file:`` / lokale Verzeichnisse haben keine sinnvolle URL.
        """
        if not self.enabled or not run_id or not self._experiment_id:
            return None
        if not self.tracking_uri.startswith(("http://", "https://")):
            return None
        base = self.tracking_uri.rstrip("/")
        return f"{base}/#/experiments/{self._experiment_id}/runs/{run_id}"


def make_sink(*, tracking_uri: str, experiment_name: str) -> MlflowSink:
    return MlflowSink(tracking_uri=tracking_uri, experiment_name=experiment_name)


def disabled_sink() -> MlflowSink:
    sink = MlflowSink.__new__(MlflowSink)
    sink.enabled = False
    sink.tracking_uri = ""
    sink.experiment_name = ""
    sink._experiment_id = None
    return sink


# Re-exported for callers that want a typed nullcontext placeholder.
NULL_CTX = nullcontext
