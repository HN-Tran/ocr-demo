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
        default_token_limit: int,
        max_image_dim: int,
    ) -> None:
        if default_token_limit < 1:
            raise ValueError("default_token_limit muss eine positive ganze Zahl sein")
        self.ollama_client = ollama_client
        self.default_model = default_model
        self.default_token_limit = default_token_limit
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
                    f"Eingabe wurde von {width}x{height} auf {new_size[0]}x{new_size[1]} skaliert"
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
            return "Beschreibe dieses Bild knapp und sachlich. Gib nur Klartext zurück."
        if selected_task == PLAIN_TASK_READ_SCENE_TEXT:
            return (
                "Lies und transkribiere den gesamten sichtbaren Text aus diesem Bild. "
                "Wenn kein Text sichtbar ist, gib exakt aus: Kein sichtbarer Text."
            )
        raise ValueError(
            f"Unbekannte Aufgabe '{selected_task}'. Unterstützte Aufgaben: {', '.join(SUPPORTED_PLAIN_TASKS)}"
        )

    def _build_structured_prompt(self, schema_name: str) -> str:
        schema = SCHEMA_REGISTRY.get(schema_name)
        if schema is None:
            raise ValueError(f"Unbekannter schema_name '{schema_name}'")
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
        token_limit: int | None = None,
    ) -> OCRResult:
        warnings: list[str] = []
        selected_model = model or self.default_model
        selected_token_limit = self.default_token_limit if token_limit is None else token_limit
        if selected_token_limit < 1:
            raise ValueError("token_limit muss eine positive ganze Zahl sein")
        prepared_image, preprocess_warnings = self._preprocess(image_bytes)
        warnings.extend(preprocess_warnings)

        if mode == "plain":
            prompt = self._build_plain_prompt(task=task, custom_prompt=custom_prompt)
        elif mode == "structured":
            if not schema_name:
                raise ValueError("schema_name ist für den strukturierten Modus erforderlich")
            if custom_prompt and custom_prompt.strip():
                raise ValueError("custom_prompt wird nur im Klartextmodus unterstützt")
            if task and task.strip() and task.strip() != PLAIN_TASK_OCR_TEXT:
                raise ValueError("task wird nur im Klartextmodus unterstützt")
            prompt = self._build_structured_prompt(schema_name)
        else:
            raise ValueError(f"Nicht unterstützter Modus '{mode}'")

        start = time.perf_counter()
        raw_output = await self.ollama_client.run_ocr(
            image_bytes=prepared_image,
            prompt=prompt,
            model=selected_model,
            num_ctx=selected_token_limit,
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
