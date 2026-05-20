from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import cast

import pytest
from fastapi import FastAPI
from starlette.routing import Mount, Route

from app.config import Settings
from app.main import _create_ocr_app, create_app


def _settings(*, app_base_path: str = "") -> Settings:
    return Settings(
        app_name="docread",
        app_base_path=app_base_path,
        analyze_store_dir="/tmp/docread-analyze-results-tests",
        inference_provider="ollama",
        inference_base_url="http://localhost:11434",
        inference_model="glm-ocr:latest",
        inference_api_key="",
        inference_vision_models=(),
        inference_vision_probe=True,
        inference_extra_providers={},
        ollama_base_url="http://localhost:11434",
        ollama_model="glm-ocr:latest",
        ocr_backend="direct",
        ocr_expert_enable_layout=True,
        ocr_expert_layout_model="PaddlePaddle/PP-DocLayoutV3_safetensors",
        ocr_expert_layout_device="auto",
        ocr_expert_table_transformer=False,
        ocr_expert_per_region_ocr=True,
        ocr_expert_text_anchor=True,
        ocr_expert_text_anchor_threshold=60.0,
        ocr_expert_compare_include_detector_only=False,
        ocr_expert_layout_max_dim=1800,
        ocr_binarized_min_dim=1800,
        azure_preset_label="",
        azure_preset_endpoint="",
        azure_preset_layout_endpoint="",
        azure_preset_key="",
        mlflow_tracking_uri="",
        mlflow_experiment_name="docread",
        benchmark_max_files=50,
        benchmark_max_runners=5,
        benchmark_job_ttl_s=3600.0,
        examples=(),
        ocr_word_detector="none",
        default_token_limit=16384,
        request_timeout_s=120.0,
        max_upload_bytes=8 * 1024 * 1024,
        max_image_dim=2048,
        verify_ssl=False,
        deskew_enabled=False,
        deskew_min_angle_deg=0.5,
        host="127.0.0.1",
        port=8000,
        app_locale="en",
    )


def _route_by_path(routes: Sequence[object], path: str) -> Route:
    for route in routes:
        if isinstance(route, Route) and route.path == path:
            return route
    raise AssertionError(f"Route '{path}' nicht gefunden")


def test_status_endpoint_exists_on_direct_app() -> None:
    app = _create_ocr_app(settings=_settings())
    route_paths = {route.path for route in app.routes if hasattr(route, "path")}

    assert "/status" in route_paths
    assert "/status/" in route_paths
    assert "/ready" in route_paths
    assert "/ContainerReadiness" in route_paths
    assert "/ContainerLiveness" in route_paths
    assert "/formrecognizer/documentModels/{modelId}:syncAnalyze" in route_paths
    assert "/formrecognizer/documentModels/{modelId}:analyze" in route_paths
    assert "/formrecognizer/documentModels/{modelId}/analyzeResults/{rId}" in route_paths

    status_route = _route_by_path(app.routes, "/status")
    payload = asyncio.run(status_route.endpoint())
    assert payload["status"] == "ok"
    assert payload["service"] == "prebuilt-read"
    assert payload["apiStatus"] == "Healthy"


def test_status_endpoint_exists_on_wrapped_app_root_and_mounted_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.get_settings", lambda: _settings(app_base_path="/ocr"))
    app = create_app()

    root_paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/status" in root_paths
    assert "/status/" in root_paths

    root_status_route = _route_by_path(app.routes, "/status")
    root_payload = asyncio.run(root_status_route.endpoint())
    assert root_payload["status"] == "ok"
    assert root_payload["service"] == "prebuilt-read"

    mounted_ocr_app = cast(
        FastAPI,
        next(
            route.app for route in app.routes if isinstance(route, Mount) and route.path == "/ocr"
        ),
    )
    mounted_paths = {route.path for route in mounted_ocr_app.routes if hasattr(route, "path")}
    assert "/status" in mounted_paths
    assert "/status/" in mounted_paths

    mounted_status_route = _route_by_path(mounted_ocr_app.routes, "/status")
    mounted_payload = asyncio.run(mounted_status_route.endpoint())
    assert mounted_payload["status"] == "ok"
    assert mounted_payload["service"] == "prebuilt-read"
