from __future__ import annotations

from unittest.mock import patch

from app.config import get_settings
from app.services.layout_detector import resolve_layout_device


def test_resolve_layout_device_cpu() -> None:
    assert resolve_layout_device("cpu") == "cpu"


def test_resolve_layout_device_auto_without_cuda() -> None:
    with patch("app.services.layout_detector.torch.cuda.is_available", return_value=False):
        assert resolve_layout_device("auto") == "cpu"


def test_resolve_layout_device_cuda_when_available() -> None:
    with patch("app.services.layout_detector.torch.cuda.is_available", return_value=True):
        assert resolve_layout_device("cuda") == "cuda"
        assert resolve_layout_device("rocm") == "cuda"
        assert resolve_layout_device("cuda:1") == "cuda:1"


def test_resolve_layout_device_cuda_unavailable_falls_back() -> None:
    with patch("app.services.layout_detector.torch.cuda.is_available", return_value=False):
        assert resolve_layout_device("cuda") == "cpu"


def test_settings_default_layout_device(monkeypatch) -> None:
    monkeypatch.delenv("OCR_EXPERT_LAYOUT_DEVICE", raising=False)
    assert get_settings().ocr_expert_layout_device == "auto"
