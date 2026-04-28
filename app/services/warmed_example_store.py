from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WarmedExample:
    slot: int
    label: str
    filename: str
    ocr_response: dict[str, Any]
    compare_response: dict[str, Any] | None


class WarmedExampleStore:
    """In-memory cache of pre-computed OCR + compare results per example slot.

    Filled in the background at app startup so the example shortcut buttons
    can serve instantly. Misses fall back to the live path on the frontend.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._entries: dict[int, WarmedExample] = {}

    async def store(self, entry: WarmedExample) -> None:
        async with self._lock:
            self._entries[entry.slot] = entry

    async def get(self, slot: int) -> WarmedExample | None:
        async with self._lock:
            return self._entries.get(slot)

    async def all(self) -> list[WarmedExample]:
        async with self._lock:
            return list(self._entries.values())
