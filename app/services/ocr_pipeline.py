from __future__ import annotations

import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from app.schemas import SCHEMA_REGISTRY
from app.services.ollama_client import OllamaClient
from app.services.structured import parse_structured_output

PLAIN_TASK_OCR_TEXT = "ocr_text"
PLAIN_TASK_DESCRIBE_IMAGE = "describe_image"
PLAIN_TASK_READ_SCENE_TEXT = "read_scene_text"
SUPPORTED_PLAIN_TASKS = (
    PLAIN_TASK_OCR_TEXT,
    PLAIN_TASK_DESCRIBE_IMAGE,
    PLAIN_TASK_READ_SCENE_TEXT,
)


@dataclass
class OCRResult:
    text: str
    structured: dict | None
    model: str
    mode: str
    schema_name: str | None
    latency_ms: int
    warnings: list[str]


class OCRPipeline:
    def __init__(
        self,
        *,
        ollama_client: OllamaClient,
        default_model: str,
        max_image_dim: int,
    ) -> None:
        self.ollama_client = ollama_client
        self.default_model = default_model
        self.max_image_dim = max_image_dim
        prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
        self.plain_prompt_template = (prompts_dir / "plain_ocr.txt").read_text(encoding="utf-8")
        self.structured_prompt_template = (prompts_dir / "structured_ocr.txt").read_text(
            encoding="utf-8"
        )

    def _preprocess(self, image_bytes: bytes) -> tuple[bytes, list[str]]:
        warnings: list[str] = []
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")

            width, height = image.size
            max_dim = max(width, height)
            if max_dim > self.max_image_dim:
                ratio = self.max_image_dim / max_dim
                new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                warnings.append(
                    f"Input resized from {width}x{height} to {new_size[0]}x{new_size[1]}"
                )

            output = BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue(), warnings

    def _build_plain_prompt(self, *, task: str | None, custom_prompt: str | None) -> str:
        if custom_prompt and custom_prompt.strip():
            return custom_prompt.strip()

        selected_task = (task or PLAIN_TASK_OCR_TEXT).strip()
        if selected_task == PLAIN_TASK_OCR_TEXT:
            return self.plain_prompt_template
        if selected_task == PLAIN_TASK_DESCRIBE_IMAGE:
            return "Describe this image in concise, factual detail. Return plain text only."
        if selected_task == PLAIN_TASK_READ_SCENE_TEXT:
            return (
                "Read and transcribe all visible text from this image. "
                "If no text is visible, return exactly: No visible text."
            )
        raise ValueError(
            f"Unknown task '{selected_task}'. Supported tasks: {', '.join(SUPPORTED_PLAIN_TASKS)}"
        )

    def _build_structured_prompt(self, schema_name: str) -> str:
        schema = SCHEMA_REGISTRY.get(schema_name)
        if schema is None:
            raise ValueError(f"Unknown schema_name '{schema_name}'")
        return self.structured_prompt_template.format(
            schema_name=schema_name,
            schema_description=schema["description"],
            schema_json=json.dumps(schema["fields"], indent=2),
            field_names=", ".join(schema["fields"].keys()),
        )

    async def run(
        self,
        *,
        image_bytes: bytes,
        mode: str,
        schema_name: str | None,
        model: str | None = None,
        task: str | None = None,
        custom_prompt: str | None = None,
    ) -> OCRResult:
        warnings: list[str] = []
        selected_model = model or self.default_model
        prepared_image, preprocess_warnings = self._preprocess(image_bytes)
        warnings.extend(preprocess_warnings)

        if mode == "plain":
            prompt = self._build_plain_prompt(task=task, custom_prompt=custom_prompt)
        elif mode == "structured":
            if not schema_name:
                raise ValueError("schema_name is required for structured mode")
            if custom_prompt and custom_prompt.strip():
                raise ValueError("custom_prompt is only supported for plain mode")
            if task and task.strip() and task.strip() != PLAIN_TASK_OCR_TEXT:
                raise ValueError("task is only supported for plain mode")
            prompt = self._build_structured_prompt(schema_name)
        else:
            raise ValueError(f"Unsupported mode '{mode}'")

        start = time.perf_counter()
        raw_output = await self.ollama_client.run_ocr(
            image_bytes=prepared_image,
            prompt=prompt,
            model=selected_model,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        if mode == "plain":
            text = raw_output.strip()
            structured = None
        else:
            schema = SCHEMA_REGISTRY[schema_name or ""]
            parse_result = parse_structured_output(raw_output, list(schema["fields"].keys()))
            warnings.extend(parse_result.warnings)
            text = raw_output.strip()
            structured = parse_result.data

        return OCRResult(
            text=text,
            structured=structured,
            model=selected_model,
            mode=mode,
            schema_name=schema_name,
            latency_ms=latency_ms,
            warnings=warnings,
        )
