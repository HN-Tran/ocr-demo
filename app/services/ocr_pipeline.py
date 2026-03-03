from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

import pypdfium2 as pdfium
from PIL import Image, ImageOps

from app.schemas import SCHEMA_REGISTRY
from app.services.ollama_client import OllamaClient, OllamaError
from app.services.structured import parse_structured_output

PLAIN_TASK_OCR_TEXT = "ocr_text"
PLAIN_TASK_DESCRIBE_IMAGE = "describe_image"
PLAIN_TASK_READ_SCENE_TEXT = "read_scene_text"
PLAIN_TASK_TABLE_MARKDOWN = "extract_table_markdown"
PLAIN_TASK_SUMMARIZE_DOCUMENT = "summarize_document"
SUPPORTED_PLAIN_TASKS = (
    PLAIN_TASK_OCR_TEXT,
    PLAIN_TASK_DESCRIBE_IMAGE,
    PLAIN_TASK_READ_SCENE_TEXT,
    PLAIN_TASK_TABLE_MARKDOWN,
    PLAIN_TASK_SUMMARIZE_DOCUMENT,
)
MAX_TOKEN_LIMIT = 128000
AUTO_SCHEMA_NAME = "auto"
PLAIN_TASK_PROMPTS: dict[str, str] = {
    PLAIN_TASK_DESCRIBE_IMAGE: "Describe this image briefly.",
    PLAIN_TASK_READ_SCENE_TEXT: (
        "Task: Transcribe all visible scene text exactly as shown.\n"
        "Rules: Keep original language, casing, punctuation, and line breaks. Do not translate.\n"
        "If no text is visible, output exactly: No visible text."
    ),
    PLAIN_TASK_TABLE_MARKDOWN: (
        "Task: Extract all visible tables.\n"
        "Output: Markdown tables only.\n"
        "Rules: Preserve cell text exactly and do not translate.\n"
        "If multiple tables exist, label them as 'Table 1', 'Table 2', etc.\n"
        "If no table is present, output exactly: No table detected."
    ),
    PLAIN_TASK_SUMMARIZE_DOCUMENT: (
        "Task: Summarize the document.\n"
        "Output: 3-6 bullet points in German.\n"
        "Rules: Keep names, numbers, and codes unchanged.\n"
        "If no readable content exists, output exactly: Kein lesbarer Inhalt."
    ),
}
PLAIN_TASK_RETRY_PROMPTS: dict[str, str] = {
    PLAIN_TASK_OCR_TEXT: (
        "Extrahiere sichtbaren Text exakt mit Zeilenumbrüchen. "
        "Originalsprache beibehalten, nicht übersetzen. "
        "Bild nicht beschreiben. Wenn kein Text sichtbar ist, gib exakt aus: Kein sichtbarer Text."
    ),
    PLAIN_TASK_DESCRIBE_IMAGE: (
        "Describe this image in 1-3 concise sentences. Do not transcribe text."
    ),
    PLAIN_TASK_READ_SCENE_TEXT: (
        "Transcribe visible text exactly. Keep original language and line breaks. "
        "Output only text. If none: No visible text."
    ),
    PLAIN_TASK_TABLE_MARKDOWN: (
        "Return visible tables as Markdown only. Do not translate cell text. "
        "If none: No table detected."
    ),
    PLAIN_TASK_SUMMARIZE_DOCUMENT: (
        "Summarize in 3-5 German bullet points. Output bullets only. "
        "If unreadable: Kein lesbarer Inhalt."
    ),
}
PLAIN_TASK_FINAL_RETRY_PROMPTS: dict[str, str] = {
    PLAIN_TASK_OCR_TEXT: (
        "Nur sichtbaren Text transkribieren. "
        "Keine Anweisungen wiederholen. "
        "Wenn kein Text sichtbar ist: Kein sichtbarer Text."
    ),
    PLAIN_TASK_DESCRIBE_IMAGE: (
        "Describe what is visually happening in the image. "
        "If text dominates, summarize it briefly instead of transcribing."
    ),
    PLAIN_TASK_READ_SCENE_TEXT: (
        "Nur sichtbaren Text mit Zeilenumbrüchen transkribieren. "
        "Keine Anweisungen wiederholen."
    ),
    PLAIN_TASK_TABLE_MARKDOWN: (
        "Nur Tabellen als Markdown ausgeben. Keine Anweisungen wiederholen."
    ),
    PLAIN_TASK_SUMMARIZE_DOCUMENT: (
        "Dokument in 3-5 deutschen Stichpunkten zusammenfassen. Keine Anweisungen wiederholen."
    ),
}


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
        if default_token_limit > MAX_TOKEN_LIMIT:
            raise ValueError(f"default_token_limit darf {MAX_TOKEN_LIMIT} nicht überschreiten")
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

    def _render_pdf_pages(self, pdf_bytes: bytes) -> tuple[list[bytes], list[str]]:
        warnings: list[str] = []
        document = None
        rendered_pages: list[bytes] = []

        try:
            document = pdfium.PdfDocument(pdf_bytes)
            page_count = len(document)
            if page_count < 1:
                raise ValueError("PDF enthält keine Seiten")

            for page_index in range(page_count):
                page = None
                bitmap = None
                try:
                    page = document[page_index]
                    bitmap = page.render(scale=2.0)
                    image = bitmap.to_pil()
                    output = BytesIO()
                    image.save(output, format="PNG", optimize=True)
                    rendered_pages.append(output.getvalue())
                finally:
                    if bitmap is not None and hasattr(bitmap, "close"):
                        bitmap.close()
                    if page is not None and hasattr(page, "close"):
                        page.close()

            if page_count > 1:
                warnings.append(f"PDF hat {page_count} Seiten; alle Seiten wurden verarbeitet")
            return rendered_pages, warnings
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError("PDF konnte nicht verarbeitet werden") from exc
        finally:
            if document is not None and hasattr(document, "close"):
                document.close()

    def _build_plain_prompt(self, *, selected_task: str, custom_prompt: str | None) -> str:
        if custom_prompt and custom_prompt.strip():
            return custom_prompt.strip()

        if selected_task == PLAIN_TASK_OCR_TEXT:
            return self.plain_prompt_template
        if selected_task in PLAIN_TASK_PROMPTS:
            return PLAIN_TASK_PROMPTS[selected_task]
        raise ValueError(
            f"Unbekannte Aufgabe '{selected_task}'. Unterstützte Aufgaben: {', '.join(SUPPORTED_PLAIN_TASKS)}"
        )

    def _build_plain_retry_prompt(self, *, selected_task: str) -> str:
        if selected_task in PLAIN_TASK_RETRY_PROMPTS:
            return PLAIN_TASK_RETRY_PROMPTS[selected_task]
        raise ValueError(
            f"Unbekannte Aufgabe '{selected_task}'. Unterstützte Aufgaben: {', '.join(SUPPORTED_PLAIN_TASKS)}"
        )

    def _build_plain_final_retry_prompt(self, *, selected_task: str) -> str:
        if selected_task in PLAIN_TASK_FINAL_RETRY_PROMPTS:
            return PLAIN_TASK_FINAL_RETRY_PROMPTS[selected_task]
        raise ValueError(
            f"Unbekannte Aufgabe '{selected_task}'. Unterstützte Aufgaben: {', '.join(SUPPORTED_PLAIN_TASKS)}"
        )

    def _looks_like_prompt_echo(self, *, prompt: str, output: str) -> bool:
        prompt_norm = re.sub(r"[^0-9a-zA-ZäöüÄÖÜß]+", " ", prompt.strip().lower())
        output_norm = re.sub(r"[^0-9a-zA-ZäöüÄÖÜß]+", " ", output.strip().lower())
        prompt_norm = " ".join(prompt_norm.split())
        output_norm = " ".join(output_norm.split())
        if not prompt_norm or not output_norm:
            return False
        if output_norm == prompt_norm:
            return True

        if len(output_norm) >= 12 and prompt_norm.startswith(output_norm):
            return True
        if len(prompt_norm) >= 12 and output_norm.startswith(prompt_norm):
            return True

        min_match_len = max(12, int(len(prompt_norm) * 0.35))
        if len(output_norm) >= min_match_len and output_norm in prompt_norm:
            return True
        if len(prompt_norm) >= min_match_len and prompt_norm in output_norm:
            return True

        instruction_markers = (
            "task",
            "rules",
            "output",
            "regeln",
            "gib",
            "extrahiere",
            "beschreibe",
            "do not",
        )
        marker_hits = sum(1 for marker in instruction_markers if marker in output_norm)
        if marker_hits >= 2 and len(output_norm.split()) <= 34:
            return True

        instruction_fragments = (
            "nur klartext",
            "klartext zurueckgeben",
            "klartext zurückgeben",
            "only plain text",
            "plain text only",
            "antworte auf deutsch",
            "answer in german",
        )
        if any(fragment in output_norm for fragment in instruction_fragments):
            return True

        prompt_tokens = {t for t in prompt_norm.split() if len(t) >= 3}
        output_tokens = [t for t in output_norm.split() if len(t) >= 3]
        if prompt_tokens and output_tokens:
            overlap = sum(1 for token in output_tokens if token in prompt_tokens)
            overlap_ratio = overlap / len(output_tokens)
            if overlap_ratio >= 0.72 and len(output_tokens) <= 24:
                return True
        return False

    def _looks_like_image_description(self, output: str) -> bool:
        normalized = " ".join(output.strip().lower().split())
        if not normalized:
            return False
        description_markers = (
            "das bild zeigt",
            "auf dem bild",
            "im bild ist",
            "im bild sind",
            "this image shows",
            "the image shows",
            "in the image",
            "a photo of",
            "an image of",
        )
        return any(marker in normalized for marker in description_markers)

    def _looks_like_ocr_transcript(self, output: str) -> bool:
        normalized = output.strip()
        if not normalized:
            return False

        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        short_lines = sum(1 for line in lines if len(line.split()) <= 5)
        numeric_like = sum(1 for line in lines if any(ch.isdigit() for ch in line))
        field_like = sum(1 for line in lines if any(token in line for token in (":", "#", "=", "/", "|")))
        upper_heavy = sum(
            1
            for line in lines
            if len(line) >= 6
            and (sum(1 for ch in line if ch.isupper()) / max(1, sum(1 for ch in line if ch.isalpha()))) > 0.6
        )

        if short_lines >= max(3, int(len(lines) * 0.6)) and (numeric_like >= 2 or field_like >= 2):
            return True
        if len(lines) == 2 and numeric_like >= 1 and field_like >= 1 and short_lines == 2:
            return True
        if numeric_like >= 3 and field_like >= 1:
            return True
        if upper_heavy >= 2 and numeric_like >= 1:
            return True

        return False

    def _is_no_visible_text(self, output: str) -> bool:
        normalized = " ".join(output.strip().lower().split())
        if not normalized:
            return True
        no_text_markers = (
            "kein sichtbarer text",
            "kein text sichtbar",
            "no visible text",
            "no text visible",
        )
        return any(marker in normalized for marker in no_text_markers)

    def _empty_structured_for_schema(self, schema_fields: dict[str, str]) -> dict[str, object]:
        empty_payload: dict[str, object] = {}
        for key, field_type in schema_fields.items():
            empty_payload[key] = [] if field_type.startswith("array") else None
        return empty_payload

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

    def _build_schema_selection_prompt(self) -> str:
        schema_lines = [
            f"- {name}: {meta['description']}" for name, meta in SCHEMA_REGISTRY.items()
        ]
        return (
            "You classify a document into exactly one OCR schema.\n"
            "Output format: return only the schema_name and nothing else.\n"
            "Available schema_name:\n"
            f"{chr(10).join(schema_lines)}\n"
            f"Allowed answers: {', '.join(SCHEMA_REGISTRY.keys())}"
        )

    async def _auto_detect_schema(
        self,
        *,
        prepared_image: bytes,
        selected_model: str,
        selected_token_limit: int,
    ) -> str:
        classifier_prompt = self._build_schema_selection_prompt()
        raw_choice = await self.ollama_client.run_ocr(
            image_bytes=prepared_image,
            prompt=classifier_prompt,
            model=selected_model,
            num_ctx=selected_token_limit,
        )
        normalized = raw_choice.strip().lower().replace("`", "")
        for schema_name in SCHEMA_REGISTRY:
            if normalized == schema_name or normalized.startswith(schema_name):
                return schema_name
        for schema_name in SCHEMA_REGISTRY:
            if schema_name in normalized:
                return schema_name
        raise ValueError(
            "Schema konnte nicht automatisch erkannt werden. Bitte schema_name manuell wählen."
        )

    @staticmethod
    def _with_page_prefix(warning: str, page_number: int, total_pages: int) -> str:
        if total_pages <= 1:
            return warning
        return f"Seite {page_number}: {warning}"

    def _prepare_images(
        self, *, image_bytes: bytes, content_type: str | None
    ) -> tuple[list[bytes], list[str]]:
        warnings: list[str] = []
        if content_type == "application/pdf":
            source_images, pdf_warnings = self._render_pdf_pages(image_bytes)
            warnings.extend(pdf_warnings)
        else:
            source_images = [image_bytes]

        prepared_images: list[bytes] = []
        total_pages = len(source_images)
        for idx, page_image in enumerate(source_images, start=1):
            prepared_image, preprocess_warnings = self._preprocess(page_image)
            prepared_images.append(prepared_image)
            warnings.extend(
                self._with_page_prefix(warning, idx, total_pages)
                for warning in preprocess_warnings
            )
        return prepared_images, warnings

    async def _run_plain_on_image(
        self,
        *,
        prepared_image: bytes,
        prompt: str,
        selected_plain_task: str,
        selected_model: str,
        selected_token_limit: int,
        has_custom_plain_prompt: bool,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []
        raw_output = await self.ollama_client.run_ocr(
            image_bytes=prepared_image,
            prompt=prompt,
            model=selected_model,
            num_ctx=selected_token_limit,
        )

        if not has_custom_plain_prompt:
            should_retry = self._looks_like_prompt_echo(prompt=prompt, output=raw_output)
            if selected_plain_task == PLAIN_TASK_OCR_TEXT and self._looks_like_image_description(
                raw_output
            ):
                warnings.append(
                    "Antwort wirkte wie Bildbeschreibung statt OCR-Text; Anfrage mit Kurzprompt wiederholt."
                )
                should_retry = True
            if selected_plain_task == PLAIN_TASK_DESCRIBE_IMAGE and self._looks_like_ocr_transcript(
                raw_output
            ):
                warnings.append(
                    "Bildbeschreibung wirkte wie OCR-Transkript; Anfrage mit beschreibendem Kurzprompt wiederholt."
                )
                should_retry = True
            if should_retry:
                retry_prompt = self._build_plain_retry_prompt(selected_task=selected_plain_task)
                if self._looks_like_prompt_echo(prompt=prompt, output=raw_output):
                    warnings.append("Prompt-Echo erkannt; Anfrage mit Kurzprompt wiederholt.")
                raw_output = await self.ollama_client.run_ocr(
                    image_bytes=prepared_image,
                    prompt=retry_prompt,
                    model=selected_model,
                    num_ctx=selected_token_limit,
                )
                needs_final_retry = False
                if self._looks_like_prompt_echo(prompt=retry_prompt, output=raw_output):
                    warnings.append("Modell gibt weiterhin promptähnlichen Text zurück.")
                    needs_final_retry = True
                if selected_plain_task == PLAIN_TASK_OCR_TEXT and self._looks_like_image_description(
                    raw_output
                ):
                    warnings.append(
                        "Antwort wirkte weiterhin wie Bildbeschreibung statt OCR-Text."
                    )
                    needs_final_retry = True
                if selected_plain_task == PLAIN_TASK_DESCRIBE_IMAGE and self._looks_like_ocr_transcript(
                    raw_output
                ):
                    warnings.append("Antwort wirkte weiterhin wie OCR-Transkript; finaler Retry.")
                    needs_final_retry = True

                if needs_final_retry:
                    final_retry_prompt = self._build_plain_final_retry_prompt(
                        selected_task=selected_plain_task
                    )
                    raw_output = await self.ollama_client.run_ocr(
                        image_bytes=prepared_image,
                        prompt=final_retry_prompt,
                        model=selected_model,
                        num_ctx=selected_token_limit,
                    )
                    if self._looks_like_prompt_echo(prompt=final_retry_prompt, output=raw_output):
                        warnings.append("Modell spiegelt weiterhin die Anweisung statt Inhalt.")
                    if selected_plain_task == PLAIN_TASK_OCR_TEXT and self._looks_like_image_description(
                        raw_output
                    ):
                        warnings.append("Modell liefert weiterhin Bildbeschreibung statt OCR-Text.")

        text = raw_output.strip()
        if not has_custom_plain_prompt and text and self._looks_like_prompt_echo(prompt=prompt, output=text):
            warnings.append("Ausgabe wurde als Prompt-Echo verworfen.")
            if selected_plain_task == PLAIN_TASK_OCR_TEXT:
                text = "Kein verwertbarer OCR-Text erkannt."
            elif selected_plain_task == PLAIN_TASK_DESCRIBE_IMAGE:
                text = "Keine verwertbare Bildbeschreibung erzeugt."
        return text, warnings

    @staticmethod
    def _is_missing_structured_value(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, list) and len(value) == 0:
            return True
        return False

    def _merge_structured_payloads(
        self, *, payloads: list[dict[str, object]], schema_fields: dict[str, str]
    ) -> tuple[dict[str, object], list[str]]:
        merged = self._empty_structured_for_schema(schema_fields)
        warnings: list[str] = []

        for field, field_type in schema_fields.items():
            is_array = field_type.startswith("array")
            for page_index, payload in enumerate(payloads, start=1):
                value = payload.get(field)
                if self._is_missing_structured_value(value):
                    continue

                if is_array:
                    target = merged[field]
                    if not isinstance(target, list):
                        target = []
                    if isinstance(value, list):
                        for item in value:
                            if item not in target:
                                target.append(item)
                    else:
                        if value not in target:
                            target.append(value)
                    merged[field] = target
                    continue

                current = merged[field]
                if self._is_missing_structured_value(current):
                    merged[field] = value
                    continue
                if current != value:
                    warnings.append(
                        f"Konflikt bei Feld '{field}' zwischen Seiten; Wert von Seite 1 wird beibehalten (abweichend auf Seite {page_index})."
                    )
        return merged, warnings

    async def run(
        self,
        *,
        image_bytes: bytes,
        content_type: str | None = None,
        mode: str,
        schema_name: str | None,
        model: str | None = None,
        task: str | None = None,
        custom_prompt: str | None = None,
        token_limit: int | None = None,
    ) -> OCRResult:
        warnings: list[str] = []
        selected_plain_task = (task or PLAIN_TASK_OCR_TEXT).strip()
        selected_model = (model or "").strip() or self.default_model
        selected_token_limit = self.default_token_limit if token_limit is None else token_limit
        has_custom_plain_prompt = bool(custom_prompt and custom_prompt.strip())
        if selected_token_limit < 1:
            raise ValueError("token_limit muss eine positive ganze Zahl sein")
        if selected_token_limit > MAX_TOKEN_LIMIT:
            raise ValueError(f"token_limit darf {MAX_TOKEN_LIMIT} nicht überschreiten")

        prepared_images, prepare_warnings = self._prepare_images(
            image_bytes=image_bytes, content_type=content_type
        )
        warnings.extend(prepare_warnings)
        total_pages = len(prepared_images)

        if mode == "plain":
            prompt = self._build_plain_prompt(
                selected_task=selected_plain_task, custom_prompt=custom_prompt
            )
            resolved_schema_name = None
        elif mode == "structured":
            if custom_prompt and custom_prompt.strip():
                raise ValueError("custom_prompt wird nur im Klartextmodus unterstützt")
            if task and task.strip() and task.strip() != PLAIN_TASK_OCR_TEXT:
                raise ValueError("task wird nur im Klartextmodus unterstützt")
            resolved_schema_name = (schema_name or "").strip()
            if not resolved_schema_name or resolved_schema_name == AUTO_SCHEMA_NAME:
                last_detect_error: ValueError | None = None
                for prepared_image in prepared_images:
                    try:
                        resolved_schema_name = await self._auto_detect_schema(
                            prepared_image=prepared_image,
                            selected_model=selected_model,
                            selected_token_limit=selected_token_limit,
                        )
                        break
                    except ValueError as exc:
                        last_detect_error = exc
                if not resolved_schema_name or resolved_schema_name == AUTO_SCHEMA_NAME:
                    raise last_detect_error or ValueError(
                        "Schema konnte nicht automatisch erkannt werden. Bitte schema_name manuell wählen."
                    )
                warnings.append(f"Automatisch erkanntes Schema: {resolved_schema_name}")
            prompt = self._build_structured_prompt(resolved_schema_name)
        else:
            raise ValueError(f"Nicht unterstützter Modus '{mode}'")

        start = time.perf_counter()

        if mode == "plain":
            page_texts: list[str] = []
            for page_index, prepared_image in enumerate(prepared_images, start=1):
                page_text, page_warnings = await self._run_plain_on_image(
                    prepared_image=prepared_image,
                    prompt=prompt,
                    selected_plain_task=selected_plain_task,
                    selected_model=selected_model,
                    selected_token_limit=selected_token_limit,
                    has_custom_plain_prompt=has_custom_plain_prompt,
                )
                page_texts.append(page_text)
                warnings.extend(
                    self._with_page_prefix(warning, page_index, total_pages)
                    for warning in page_warnings
                )

            if total_pages <= 1:
                text = page_texts[0] if page_texts else ""
            else:
                text = "\n\n".join(
                    f"--- Seite {page_index} ---\n{page_texts[page_index - 1]}"
                    for page_index in range(1, total_pages + 1)
                )
            structured = None
        else:
            schema = SCHEMA_REGISTRY[resolved_schema_name or ""]
            expected_fields = list(schema["fields"].keys())

            raw_outputs: list[str] = []
            parsed_payloads: list[dict[str, object]] = []
            for page_index, prepared_image in enumerate(prepared_images, start=1):
                raw_output = await self.ollama_client.run_ocr(
                    image_bytes=prepared_image,
                    prompt=prompt,
                    model=selected_model,
                    num_ctx=selected_token_limit,
                )
                raw_outputs.append(raw_output.strip())
                parse_result = parse_structured_output(raw_output, expected_fields)
                warnings.extend(
                    self._with_page_prefix(warning, page_index, total_pages)
                    for warning in parse_result.warnings
                )
                if parse_result.data is not None:
                    parsed_payloads.append(cast(dict[str, object], parse_result.data))

            if total_pages <= 1:
                text = raw_outputs[0] if raw_outputs else ""
            else:
                text = "\n\n".join(
                    f"--- Seite {page_index} ---\n{raw_outputs[page_index - 1]}"
                    for page_index in range(1, total_pages + 1)
                )

            structured = None
            if parsed_payloads:
                structured, merge_warnings = self._merge_structured_payloads(
                    payloads=parsed_payloads,
                    schema_fields=cast(dict[str, str], schema["fields"]),
                )
                warnings.extend(merge_warnings)

            if structured is not None:
                try:
                    has_visible_text = False
                    for prepared_image in prepared_images:
                        evidence_text = await self.ollama_client.run_ocr(
                            image_bytes=prepared_image,
                            prompt=self.plain_prompt_template,
                            model=selected_model,
                            num_ctx=min(selected_token_limit, 4096),
                        )
                        if not self._is_no_visible_text(evidence_text):
                            has_visible_text = True
                            break
                    if not has_visible_text:
                        structured = self._empty_structured_for_schema(schema["fields"])
                        warnings.append(
                            "Kein sichtbarer Text erkannt; strukturierte Felder wurden auf null gesetzt."
                        )
                except OllamaError as exc:
                    warnings.append(f"Evidenzprüfung für strukturierte Ausgabe übersprungen: {exc}")

        latency_ms = int((time.perf_counter() - start) * 1000)
        return OCRResult(
            text=text,
            structured=structured,
            model=selected_model,
            mode=mode,
            schema_name=resolved_schema_name,
            latency_ms=latency_ms,
            warnings=warnings,
        )
