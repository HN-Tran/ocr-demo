from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.config import get_settings
from app.services.ocr_pipeline import OCRPipeline
from app.services.ollama_client import OllamaClient


def create_app() -> FastAPI:
    settings = get_settings()
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
    app.state.ollama_client = ollama_client
    app.state.ocr_pipeline = ocr_pipeline

    base_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))
    app.state.templates = templates
    static_files = (base_dir / "static" / "styles.css", base_dir / "static" / "app.js")
    static_version = str(int(max(path.stat().st_mtime for path in static_files)))
    app.state.static_version = static_version

    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
    app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        version = cast(str, request.app.state.static_version)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "default_model": settings.ollama_model,
                "default_token_limit": settings.default_token_limit,
                "static_version": version,
            },
        )

    return app


app = create_app()
