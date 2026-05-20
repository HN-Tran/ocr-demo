from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import compat_router, router, warm_example
from app.config import Settings, get_settings
from app.services.analyze_operation_store import AnalyzeOperationStore
from app.services.backend_router import OCRBackendRouter
from app.services.benchmark import BenchmarkJobStore
from app.i18n import SUPPORTED_LOCALES, load_messages, resolve_locale
from app.i18n.compare import available_engines as localized_compare_engines
from app.services.document_pipeline import DocumentPipeline
from app.services.mlflow_sink import make_sink as make_mlflow_sink
from app.services.ocr_pipeline import OCRPipeline
from app.services.inference import create_vision_registry
from app.services.warmed_example_store import WarmedExampleStore
from app.services.word_detector import WordDetector, create_word_detector

logger = logging.getLogger("docread")


def _try_create_word_detector(name: str) -> WordDetector | None:
    try:
        return create_word_detector(name)
    except ImportError:
        logger.warning("Wort-Detektor %r nicht verfügbar (Paket nicht installiert).", name)
        return None


def _normalize_base_path(value: str) -> str:
    path = value.strip()
    if not path or path == "/":
        return ""
    if not path.startswith("/"):
        path = f"/{path}"
    return path.rstrip("/")


def _ui_context(request: Request, settings: Settings) -> dict[str, Any]:
    locale = resolve_locale(
        settings_locale=settings.app_locale,
        cookie=request.cookies.get("app_locale"),
        query=request.query_params.get("lang"),
    )
    ui = load_messages(locale)
    catalog = {code: load_messages(code) for code in SUPPORTED_LOCALES}
    return {
        "app_locale": locale,
        "ui": ui,
        "ui_messages_json": json.dumps(ui, ensure_ascii=False),
        "ui_catalog_json": json.dumps(catalog, ensure_ascii=False),
    }


def _status_payload() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "prebuilt-read",
        "apiStatus": "Healthy",
        "apiStatusMessage": "Service is running.",
    }


def _create_ocr_app(*, settings: Settings) -> FastAPI:
    app = FastAPI(title=settings.app_name)

    logger = logging.getLogger("docread")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app.state.logger = logger

    vision_registry = create_vision_registry(settings)
    ocr_pipeline = OCRPipeline(
        vision_registry=vision_registry,
        default_model=settings.inference_model,
        default_token_limit=settings.default_token_limit,
        max_image_dim=settings.max_image_dim,
        binarized_min_dim=settings.ocr_binarized_min_dim,
        deskew_enabled=settings.deskew_enabled,
        deskew_min_angle_deg=settings.deskew_min_angle_deg,
    )
    document_pipeline = DocumentPipeline(
        direct_pipeline=ocr_pipeline,
        vision_registry=vision_registry,
        default_model=settings.inference_model,
        enable_layout=settings.ocr_expert_enable_layout,
        layout_model=settings.ocr_expert_layout_model,
        timeout_s=settings.request_timeout_s,
        enable_table_transformer=settings.ocr_expert_table_transformer,
        enable_per_region_ocr=settings.ocr_expert_per_region_ocr,
        enable_text_anchor=settings.ocr_expert_text_anchor,
        text_anchor_threshold=settings.ocr_expert_text_anchor_threshold,
        word_detector=_try_create_word_detector(settings.ocr_word_detector),
        layout_max_dim=settings.ocr_expert_layout_max_dim,
    )
    ocr_backend_router = OCRBackendRouter(
        default_backend=settings.ocr_backend,
        backends={
            "direct": ocr_pipeline,
            "expert": document_pipeline,
        },
    )
    app.state.vision_registry = vision_registry
    app.state.vision_client = vision_registry.get(settings.inference_provider)
    app.state.ocr_pipeline = ocr_pipeline
    app.state.ocr_backend_router = ocr_backend_router
    app.state.analyze_operation_store = AnalyzeOperationStore(
        storage_dir=settings.analyze_store_dir
    )
    app.state.warmed_example_store = WarmedExampleStore()
    app.state.benchmark_store = BenchmarkJobStore()
    app.state.mlflow_sink = make_mlflow_sink(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )

    @app.on_event("startup")
    async def _warm_examples_on_startup() -> None:
        if not settings.examples:
            return
        store: WarmedExampleStore = app.state.warmed_example_store

        async def _warm_one(slot: int, label: str, path_str: str) -> None:
            file_path = Path(path_str)
            if not file_path.is_file():
                logger.warning(
                    "Beispiel %d (%s): Datei %s nicht gefunden — Warm-Up übersprungen.",
                    slot,
                    label,
                    file_path,
                )
                return
            try:
                entry = await warm_example(
                    slot=slot,
                    label=label,
                    file_path=file_path,
                    pipeline=ocr_backend_router,
                    azure_endpoint=settings.azure_preset_endpoint,
                    azure_key=settings.azure_preset_key,
                    verify_ssl=settings.verify_ssl,
                    include_detector_only=settings.ocr_expert_compare_include_detector_only,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Beispiel %d (%s) konnte nicht vorgewärmt werden: %s", slot, label, exc
                )
                return
            await store.store(entry)
            logger.info(
                "Beispiel %d (%s) vorgewärmt (compare=%s).",
                slot,
                label,
                "ja" if entry.compare_response else "nein",
            )

        # Run all warmups in the background; don't block the event loop on
        # them. The cached endpoint returns 404 until each task completes.
        for idx, (label, path_str) in enumerate(settings.examples, start=1):
            asyncio.create_task(_warm_one(idx, label, path_str))

    base_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))
    app.state.templates = templates
    static_files = (
        base_dir / "static" / "styles.css",
        base_dir / "static" / "app.js",
        base_dir / "static" / "benchmark.css",
        base_dir / "static" / "benchmark.js",
    )
    static_version = str(int(max(path.stat().st_mtime for path in static_files)))
    app.state.static_version = static_version

    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
    app.include_router(router)
    app.include_router(compat_router)

    @app.get("/status")
    async def status() -> dict[str, str]:
        return _status_payload()

    app.add_api_route("/status/", status, methods=["GET"], include_in_schema=False)

    @app.get("/benchmark", response_class=HTMLResponse)
    async def benchmark_page(request: Request) -> HTMLResponse:
        version = cast(str, request.app.state.static_version)
        app_base_path = cast(str, request.scope.get("root_path", ""))
        return templates.TemplateResponse(
            request=request,
            name="benchmark.html",
            context={
                **_ui_context(request, settings),
                "compare_engines": localized_compare_engines(
                    resolve_locale(
                        settings_locale=settings.app_locale,
                        cookie=request.cookies.get("app_locale"),
                    )
                ),
                "azure_preset_label": f"{settings.azure_preset_label} (Plain)" if settings.azure_preset_label else "",
                "azure_preset_layout_label": f"{settings.azure_preset_label} (Layout)" if settings.azure_preset_layout_endpoint and settings.azure_preset_label else "",
                "static_version": version,
                "app_base_path": app_base_path,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        version = cast(str, request.app.state.static_version)
        app_base_path = cast(str, request.scope.get("root_path", ""))
        locale = resolve_locale(
            settings_locale=settings.app_locale,
            cookie=request.cookies.get("app_locale"),
            query=request.query_params.get("lang"),
        )
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                **_ui_context(request, settings),
                "default_model": settings.inference_model,
                "inference_provider": settings.inference_provider,
                "inference_providers": vision_registry.provider_ids,
                "default_token_limit": settings.default_token_limit,
                "default_backend": settings.ocr_backend,
                "default_expert_enable_layout": settings.ocr_expert_enable_layout,
                "default_expert_layout_model": settings.ocr_expert_layout_model,
                "default_expert_table_transformer": settings.ocr_expert_table_transformer,
                "default_expert_per_region_ocr": settings.ocr_expert_per_region_ocr,
                "default_expert_text_anchor": settings.ocr_expert_text_anchor,
                "default_expert_text_anchor_threshold": settings.ocr_expert_text_anchor_threshold,
                "default_expert_word_detector": settings.ocr_word_detector,
                "default_expert_compare_include_detector_only": (
                    settings.ocr_expert_compare_include_detector_only
                ),
                "azure_preset_label": f"{settings.azure_preset_label} (Plain)" if settings.azure_preset_label else "",
                "azure_preset_endpoint": settings.azure_preset_endpoint,
                "azure_preset_layout_label": f"{settings.azure_preset_label} (Layout)" if settings.azure_preset_layout_endpoint and settings.azure_preset_label else "",
                "azure_preset_layout_endpoint": settings.azure_preset_layout_endpoint,
                "compare_engines": localized_compare_engines(locale),
                "examples": [
                    {"slot": idx + 1, "label": label}
                    for idx, (label, _path) in enumerate(settings.examples)
                ],
                "static_version": version,
                "app_base_path": app_base_path,
            },
        )

    return app


def create_app() -> FastAPI:
    settings = get_settings()
    ocr_app = _create_ocr_app(settings=settings)
    base_path = _normalize_base_path(settings.app_base_path)
    if not base_path:
        return ocr_app

    root_app = FastAPI(title=settings.app_name)

    @root_app.get("/status")
    async def status() -> dict[str, str]:
        return _status_payload()

    root_app.add_api_route("/status/", status, methods=["GET"], include_in_schema=False)
    root_app.mount(base_path, ocr_app)
    return root_app


app = create_app()
