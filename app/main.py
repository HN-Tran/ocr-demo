from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import compat_router, router
from app.config import Settings, get_settings
from app.services.analyze_operation_store import AnalyzeOperationStore
from app.services.backend_router import OCRBackendRouter
from app.services.document_pipeline import DocumentPipeline
from app.services.ocr_pipeline import OCRPipeline
from app.services.ollama_client import OllamaClient


def _normalize_base_path(value: str) -> str:
    path = value.strip()
    if not path or path == "/":
        return ""
    if not path.startswith("/"):
        path = f"/{path}"
    return path.rstrip("/")


def _status_payload() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "prebuilt-read",
        "apiStatus": "Healthy",
        "apiStatusMessage": "Service is running.",
    }


def _create_ocr_app(*, settings: Settings) -> FastAPI:
    app = FastAPI(title=settings.app_name)

    logger = logging.getLogger("ocr-demo")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app.state.logger = logger

    ollama_client = OllamaClient(
        base_url=settings.ollama_base_url,
        timeout_s=settings.request_timeout_s,
    )
    ocr_pipeline = OCRPipeline(
        ollama_client=ollama_client,
        default_model=settings.ollama_model,
        default_token_limit=settings.default_token_limit,
        max_image_dim=settings.max_image_dim,
    )
    document_pipeline = DocumentPipeline(
        direct_pipeline=ocr_pipeline,
        ollama_client=ollama_client,
        default_model=settings.ollama_model,
        enable_layout=settings.ocr_expert_enable_layout,
        layout_model=settings.ocr_expert_layout_model,
        timeout_s=settings.request_timeout_s,
    )
    ocr_backend_router = OCRBackendRouter(
        default_backend=settings.ocr_backend,
        backends={
            "direct": ocr_pipeline,
            "expert": document_pipeline,
        },
    )
    app.state.ollama_client = ollama_client
    app.state.ocr_pipeline = ocr_pipeline
    app.state.ocr_backend_router = ocr_backend_router
    app.state.analyze_operation_store = AnalyzeOperationStore(
        storage_dir=settings.analyze_store_dir
    )

    base_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))
    app.state.templates = templates
    static_files = (base_dir / "static" / "styles.css", base_dir / "static" / "app.js")
    static_version = str(int(max(path.stat().st_mtime for path in static_files)))
    app.state.static_version = static_version

    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
    app.include_router(router)
    app.include_router(compat_router)

    @app.get("/status")
    async def status() -> dict[str, str]:
        return _status_payload()

    app.add_api_route("/status/", status, methods=["GET"], include_in_schema=False)

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        version = cast(str, request.app.state.static_version)
        app_base_path = cast(str, request.scope.get("root_path", ""))
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "default_model": settings.ollama_model,
                "default_token_limit": settings.default_token_limit,
                "default_backend": settings.ocr_backend,
                "default_expert_enable_layout": settings.ocr_expert_enable_layout,
                "default_expert_layout_model": settings.ocr_expert_layout_model,
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
