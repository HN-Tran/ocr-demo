from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.services.analyze_operation_store import AnalyzeOperationStore


def test_store_persists_operations_to_disk(tmp_path: Path) -> None:
    created_at = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
    completed_at = datetime(2026, 3, 9, 12, 1, tzinfo=timezone.utc)

    first_store = AnalyzeOperationStore(storage_dir=tmp_path)
    operation = asyncio.run(first_store.create(model_id="prebuilt-read", created_at=created_at))
    asyncio.run(
        first_store.mark_succeeded(
            operation.id,
            payload={"status": "succeeded", "analyzeResult": {"content": "hello"}},
            completed_at=completed_at,
        )
    )

    second_store = AnalyzeOperationStore(storage_dir=tmp_path)
    loaded_operation = asyncio.run(second_store.get(operation.id))

    assert loaded_operation is not None
    assert loaded_operation.id == operation.id
    assert loaded_operation.status == "succeeded"
    assert loaded_operation.payload == {
        "status": "succeeded",
        "analyzeResult": {"content": "hello"},
    }
    assert loaded_operation.request_id == operation.request_id
    assert loaded_operation.created_at == created_at
    assert loaded_operation.updated_at == completed_at
