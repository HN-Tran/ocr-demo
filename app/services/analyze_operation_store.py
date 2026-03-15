from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class AnalyzeOperation:
    id: str
    model_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    request_id: str
    payload: dict[str, Any] | None = None
    error: dict[str, str] | None = None


class AnalyzeOperationStore:
    def __init__(self, *, storage_dir: str | Path | None = None) -> None:
        self._operations: dict[str, AnalyzeOperation] = {}
        self._lock = asyncio.Lock()
        self._storage_dir = Path(storage_dir) if storage_dir else None
        if self._storage_dir is not None:
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _serialize_operation(operation: AnalyzeOperation) -> dict[str, Any]:
        return {
            "id": operation.id,
            "model_id": operation.model_id,
            "status": operation.status,
            "created_at": operation.created_at.isoformat(),
            "updated_at": operation.updated_at.isoformat(),
            "request_id": operation.request_id,
            "payload": operation.payload,
            "error": operation.error,
        }

    @staticmethod
    def _deserialize_operation(payload: dict[str, Any]) -> AnalyzeOperation:
        return AnalyzeOperation(
            id=str(payload["id"]),
            model_id=str(payload["model_id"]),
            status=str(payload["status"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            updated_at=datetime.fromisoformat(str(payload["updated_at"])),
            request_id=str(payload["request_id"]),
            payload=payload.get("payload"),
            error=payload.get("error"),
        )

    def _operation_path(self, operation_id: str) -> Path | None:
        if self._storage_dir is None:
            return None
        return self._storage_dir / f"{operation_id}.json"

    def _persist_operation(self, operation: AnalyzeOperation) -> None:
        operation_path = self._operation_path(operation.id)
        if operation_path is None:
            return

        temp_path = operation_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(self._serialize_operation(operation), ensure_ascii=True),
            encoding="utf-8",
        )
        temp_path.replace(operation_path)

    def _load_operation(self, operation_id: str) -> AnalyzeOperation | None:
        operation_path = self._operation_path(operation_id)
        if operation_path is None or not operation_path.is_file():
            return None
        payload = json.loads(operation_path.read_text(encoding="utf-8"))
        return self._deserialize_operation(payload)

    async def create(self, *, model_id: str, created_at: datetime) -> AnalyzeOperation:
        operation = AnalyzeOperation(
            id=str(uuid4()),
            model_id=model_id,
            status="notStarted",
            created_at=created_at,
            updated_at=created_at,
            request_id=str(uuid4()),
        )
        async with self._lock:
            self._operations[operation.id] = operation
            self._persist_operation(operation)
        return operation

    async def mark_running(
        self, operation_id: str, *, started_at: datetime
    ) -> AnalyzeOperation | None:
        async with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None:
                operation = self._load_operation(operation_id)
                if operation is None:
                    return None
                self._operations[operation_id] = operation
            operation.status = "running"
            operation.updated_at = started_at
            self._persist_operation(operation)
            return operation

    async def get(self, operation_id: str) -> AnalyzeOperation | None:
        async with self._lock:
            operation = self._operations.get(operation_id)
            if operation is not None:
                return operation
            persisted_operation = self._load_operation(operation_id)
            if persisted_operation is not None:
                self._operations[operation_id] = persisted_operation
            return persisted_operation

    async def mark_succeeded(
        self, operation_id: str, *, payload: dict[str, Any], completed_at: datetime
    ) -> AnalyzeOperation | None:
        async with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None:
                return None
            operation.status = "succeeded"
            operation.updated_at = completed_at
            operation.payload = payload
            operation.error = None
            self._persist_operation(operation)
            return operation

    async def mark_failed(
        self,
        operation_id: str,
        *,
        code: str,
        message: str,
        failed_at: datetime,
    ) -> AnalyzeOperation | None:
        async with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None:
                return None
            operation.status = "failed"
            operation.updated_at = failed_at
            operation.payload = None
            operation.error = {"code": code, "message": message}
            self._persist_operation(operation)
            return operation
