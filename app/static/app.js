const form = document.getElementById("ocr-form");
const pageEl = document.querySelector(".page");
const fileEl = document.getElementById("file");
const modeEl = document.getElementById("mode");
const taskEl = document.getElementById("task");
const customPromptEl = document.getElementById("custom_prompt");
const schemaNameEl = document.getElementById("schema_name");
const modelEl = document.getElementById("model");
const tokenLimitEl = document.getElementById("token_limit");
const taskWrap = document.getElementById("task-wrap");
const customPromptWrap = document.getElementById("custom-prompt-wrap");
const schemaWrap = document.getElementById("schema-wrap");
const advancedPanelEl = document.getElementById("advanced-panel");
const advancedToggleEl = document.getElementById("advanced-toggle");
const onboardingCardEl = document.getElementById("onboarding-card");
const dropzoneEl = document.getElementById("dropzone");
const pickFileBtnEl = document.getElementById("pick-file-btn");
const changeFileBtnEl = document.getElementById("change-file-btn");
const previewWrapEl = document.getElementById("preview-wrap");
const resultPanelEl = document.getElementById("result-panel");
const globalDropOverlayEl = document.getElementById("global-drop-overlay");
const outputEl = document.getElementById("output");
const jsonOutputEl = document.getElementById("json-output");
const structuredWrapEl = document.getElementById("structured-wrap");
const metaEl = document.getElementById("meta");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");
const themeLightBtn = document.getElementById("theme-light-btn");
const themeDarkBtn = document.getElementById("theme-dark-btn");
const loadingOverlayEl = document.getElementById("loading-overlay");
const previewEmptyEl = document.getElementById("preview-empty");
const previewImageEl = document.getElementById("preview-image");
const previewPdfEl = document.getElementById("preview-pdf");
const previewPdfLinkEl = document.getElementById("preview-pdf-link");

let lastResponse = null;
let previewUrl = null;
let activeRequestController = null;
let rerunTimer = null;
let globalDragDepth = 0;
const THEME_KEY = "ocr-demo-theme";
const PRESET_STRUCTURED_MODES = new Set(["invoice_basic", "receipt_basic"]);

function isStructuredMode(modeValue) {
  return modeValue === "structured" || PRESET_STRUCTURED_MODES.has(modeValue);
}

function applyTheme(theme) {
  const root = document.documentElement;
  const isDark = theme === "dark";
  if (isDark) {
    root.setAttribute("data-theme", "dark");
  } else {
    root.removeAttribute("data-theme");
  }
  themeLightBtn.setAttribute("aria-pressed", String(!isDark));
  themeDarkBtn.setAttribute("aria-pressed", String(isDark));
}

function initTheme() {
  const savedTheme = localStorage.getItem(THEME_KEY);
  if (savedTheme === "dark" || savedTheme === "light") {
    applyTheme(savedTheme);
    return;
  }

  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(prefersDark ? "dark" : "light");
}

function toggleModeDependentFields() {
  const useCustomSchema = modeEl.value === "structured";
  schemaWrap.classList.toggle("hidden", !useCustomSchema);
  taskWrap.classList.toggle("hidden", isStructuredMode(modeEl.value));
  customPromptWrap.classList.toggle("hidden", isStructuredMode(modeEl.value));
}

function setAdvancedOpen(isOpen) {
  advancedPanelEl.classList.toggle("is-collapsed", !isOpen);
  advancedToggleEl.setAttribute("aria-expanded", String(isOpen));
  advancedToggleEl.textContent = isOpen ? "Expertenoptionen ausblenden" : "Expertenoptionen";
}

function setLoading(isLoading) {
  loadingOverlayEl.classList.toggle("is-active", isLoading);
}

function clearOutput() {
  outputEl.textContent = "";
  jsonOutputEl.innerHTML = "";
  structuredWrapEl.classList.add("hidden");
  downloadBtn.classList.add("hidden");
  lastResponse = null;
}

function setWorkspaceVisible(isVisible) {
  if (pageEl) {
    pageEl.classList.toggle("is-start", !isVisible);
    pageEl.classList.toggle("has-workspace", isVisible);
  }
  onboardingCardEl.classList.toggle("hidden", isVisible);
  previewWrapEl.classList.toggle("hidden", !isVisible);
  resultPanelEl.classList.toggle("hidden", !isVisible);
}

function openFilePicker() {
  fileEl.click();
}

function setGlobalDropActive(isActive) {
  globalDropOverlayEl.classList.toggle("is-active", isActive);
}

function clearDropHighlights() {
  dropzoneEl.classList.remove("drag-over");
  previewWrapEl.classList.remove("drag-over");
}

function applyDroppedFile(file) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileEl.files = transfer.files;
  fileEl.dispatchEvent(new Event("change", { bubbles: true }));
}

function eventHasFiles(event) {
  if (!event.dataTransfer || !event.dataTransfer.types) {
    return false;
  }
  return Array.from(event.dataTransfer.types).includes("Files");
}

function firstDroppedFile(event) {
  if (!event.dataTransfer || !event.dataTransfer.files || event.dataTransfer.files.length === 0) {
    return null;
  }
  return event.dataTransfer.files[0];
}

function clearPreview(message = "Keine Datei ausgewählt.") {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  previewEmptyEl.textContent = message;
  previewEmptyEl.classList.remove("hidden");
  previewImageEl.classList.add("hidden");
  previewImageEl.removeAttribute("src");
  previewPdfEl.classList.add("hidden");
  previewPdfEl.removeAttribute("data");
  previewPdfLinkEl.classList.add("hidden");
  previewPdfLinkEl.removeAttribute("href");
}

function updatePreview() {
  const file = fileEl.files && fileEl.files[0];
  if (!file) {
    clearPreview();
    return;
  }

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  previewUrl = URL.createObjectURL(file);
  previewEmptyEl.classList.add("hidden");

  if (file.type.startsWith("image/")) {
    previewImageEl.src = previewUrl;
    previewImageEl.classList.remove("hidden");
    previewPdfEl.classList.add("hidden");
    previewPdfEl.removeAttribute("data");
    previewPdfLinkEl.classList.add("hidden");
    previewPdfLinkEl.removeAttribute("href");
    return;
  }

  if (file.type === "application/pdf") {
    previewPdfEl.data = previewUrl;
    previewPdfEl.classList.remove("hidden");
    previewPdfLinkEl.href = previewUrl;
    previewPdfLinkEl.classList.remove("hidden");
    previewImageEl.classList.add("hidden");
    previewImageEl.removeAttribute("src");
    return;
  }

  clearPreview(`Für "${file.type || "unbekannt"}" ist keine Vorschau verfügbar.`);
}

function currentFile() {
  return fileEl.files && fileEl.files[0] ? fileEl.files[0] : null;
}

function scheduleAutoRun(delayMs = 0) {
  if (!currentFile()) {
    return;
  }
  if (rerunTimer !== null) {
    clearTimeout(rerunTimer);
    rerunTimer = null;
  }
  if (delayMs <= 0) {
    void runOCR();
    return;
  }
  rerunTimer = window.setTimeout(() => {
    rerunTimer = null;
    void runOCR();
  }, delayMs);
}

function buildPayload() {
  const payload = new FormData(form);
  const selectedMode = String(payload.get("mode") || "plain");
  if (PRESET_STRUCTURED_MODES.has(selectedMode)) {
    payload.set("mode", "structured");
    payload.set("schema_name", selectedMode);
  }

  const tokenLimitRaw = String(payload.get("token_limit") || "").trim();
  if (tokenLimitRaw) {
    const tokenLimit = Number(tokenLimitRaw);
    if (!Number.isInteger(tokenLimit) || tokenLimit < 1) {
      throw new Error("Token-Limit muss eine positive ganze Zahl sein.");
    }
    payload.set("token_limit", String(tokenLimit));
  } else {
    payload.delete("token_limit");
  }

  const isStructured = payload.get("mode") === "structured";
  if (!isStructured) {
    payload.delete("schema_name");
    const customPrompt = String(payload.get("custom_prompt") || "").trim();
    if (customPrompt) {
      payload.set("custom_prompt", customPrompt);
    } else {
      payload.delete("custom_prompt");
    }
    if (!payload.get("task")) {
      payload.set("task", "ocr_text");
    }
  } else {
    const schemaName = String(payload.get("schema_name") || "").trim();
    if (!schemaName) {
      payload.delete("schema_name");
    } else {
      payload.set("schema_name", schemaName);
    }
    payload.delete("task");
    payload.delete("custom_prompt");
  }
  if (!payload.get("model")) {
    payload.delete("model");
  }
  return payload;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function highlightJson(value) {
  const json = JSON.stringify(value, null, 2);
  const escaped = escapeHtml(json);
  return escaped.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(?:\s*:)?|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)/g,
    (token) => {
      let className = "json-number";
      if (token.startsWith('"') && token.endsWith(":")) {
        className = "json-key";
      } else if (token.startsWith('"')) {
        className = "json-string";
      } else if (token === "true" || token === "false") {
        className = "json-boolean";
      } else if (token === "null") {
        className = "json-null";
      }
      return `<span class="${className}">${token}</span>`;
    }
  );
}

async function runOCR() {
  const selectedFile = currentFile();
  if (!selectedFile) {
    if (activeRequestController) {
      activeRequestController.abort();
      activeRequestController = null;
    }
    setLoading(false);
    metaEl.textContent = "Datei auswählen, um OCR zu starten.";
    clearOutput();
    setWorkspaceVisible(false);
    return;
  }

  if (selectedFile.type === "application/pdf") {
    if (activeRequestController) {
      activeRequestController.abort();
      activeRequestController = null;
    }
    setLoading(false);
    metaEl.textContent = "PDF-Vorschau wird unterstützt, OCR derzeit jedoch nur für PNG/JPEG/WEBP.";
    clearOutput();
    setWorkspaceVisible(true);
    return;
  }

  if (activeRequestController) {
    activeRequestController.abort();
  }
  const controller = new AbortController();
  activeRequestController = controller;
  setLoading(true);
  clearOutput();
  setWorkspaceVisible(true);
  metaEl.textContent = "OCR wird ausgeführt...";

  try {
    const response = await fetch("/api/ocr", {
      method: "POST",
      body: buildPayload(),
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "OCR fehlgeschlagen");
    }

    lastResponse = data;
    outputEl.textContent = data.text || "(kein Inhalt)";
    const showStructured = data.mode === "structured" && !!data.structured;
    structuredWrapEl.classList.toggle("hidden", !showStructured);
    downloadBtn.classList.toggle("hidden", !showStructured);
    if (showStructured) {
      jsonOutputEl.innerHTML = highlightJson(data.structured);
    } else {
      jsonOutputEl.innerHTML = "";
    }

    const warnings = (data.warnings || []).join(" | ");
    metaEl.textContent = `Modell: ${data.model} | Latenz: ${data.latency_ms} ms${warnings ? ` | Hinweise: ${warnings}` : ""}`;
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    metaEl.textContent = `Fehler: ${error.message}`;
  } finally {
    if (activeRequestController === controller) {
      activeRequestController = null;
      setLoading(false);
    }
  }
}

themeLightBtn.addEventListener("click", () => {
  localStorage.setItem(THEME_KEY, "light");
  applyTheme("light");
});

themeDarkBtn.addEventListener("click", () => {
  localStorage.setItem(THEME_KEY, "dark");
  applyTheme("dark");
});

advancedToggleEl.addEventListener("click", () => {
  const isOpen = advancedToggleEl.getAttribute("aria-expanded") === "true";
  setAdvancedOpen(!isOpen);
});

pickFileBtnEl.addEventListener("click", openFilePicker);
changeFileBtnEl.addEventListener("click", openFilePicker);
dropzoneEl.addEventListener("click", openFilePicker);
dropzoneEl.addEventListener("dragover", (event) => {
  if (!eventHasFiles(event)) {
    return;
  }
  event.preventDefault();
  dropzoneEl.classList.add("drag-over");
});
dropzoneEl.addEventListener("dragleave", () => {
  dropzoneEl.classList.remove("drag-over");
});
dropzoneEl.addEventListener("drop", (event) => {
  if (!eventHasFiles(event)) {
    return;
  }
  event.preventDefault();
  globalDragDepth = 0;
  setGlobalDropActive(false);
  clearDropHighlights();
  dropzoneEl.classList.remove("drag-over");
  const file = firstDroppedFile(event);
  if (!file) return;
  applyDroppedFile(file);
});
previewWrapEl.addEventListener("dragover", (event) => {
  if (!eventHasFiles(event)) {
    return;
  }
  event.preventDefault();
  previewWrapEl.classList.add("drag-over");
});
previewWrapEl.addEventListener("dragleave", () => {
  previewWrapEl.classList.remove("drag-over");
});
previewWrapEl.addEventListener("drop", (event) => {
  if (!eventHasFiles(event)) {
    return;
  }
  event.preventDefault();
  globalDragDepth = 0;
  setGlobalDropActive(false);
  clearDropHighlights();
  previewWrapEl.classList.remove("drag-over");
  const file = firstDroppedFile(event);
  if (!file) return;
  applyDroppedFile(file);
});
window.addEventListener("dragenter", (event) => {
  if (!eventHasFiles(event)) {
    return;
  }
  event.preventDefault();
  globalDragDepth += 1;
  setGlobalDropActive(true);
});
window.addEventListener("dragover", (event) => {
  if (!eventHasFiles(event)) {
    return;
  }
  event.preventDefault();
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = "copy";
  }
  setGlobalDropActive(true);
});
window.addEventListener("dragleave", (event) => {
  if (globalDragDepth <= 0) {
    return;
  }
  event.preventDefault();
  if (event.relatedTarget === null) {
    globalDragDepth = 0;
  } else {
    globalDragDepth = Math.max(0, globalDragDepth - 1);
  }
  if (globalDragDepth === 0) {
    setGlobalDropActive(false);
    clearDropHighlights();
  }
});
window.addEventListener("drop", (event) => {
  event.preventDefault();
  globalDragDepth = 0;
  setGlobalDropActive(false);
  clearDropHighlights();
  if (!eventHasFiles(event)) {
    return;
  }
  const file = firstDroppedFile(event);
  if (!file) return;
  applyDroppedFile(file);
});

fileEl.addEventListener("change", () => {
  if (rerunTimer !== null) {
    clearTimeout(rerunTimer);
    rerunTimer = null;
  }
  setWorkspaceVisible(!!currentFile());
  updatePreview();
  void runOCR();
});
modeEl.addEventListener("change", () => {
  toggleModeDependentFields();
  scheduleAutoRun();
});
taskEl.addEventListener("change", () => {
  scheduleAutoRun();
});
schemaNameEl.addEventListener("change", () => {
  scheduleAutoRun();
});
modelEl.addEventListener("change", () => {
  scheduleAutoRun();
});
tokenLimitEl.addEventListener("input", () => {
  scheduleAutoRun(450);
});
customPromptEl.addEventListener("input", () => {
  scheduleAutoRun(450);
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  scheduleAutoRun();
});
initTheme();
setAdvancedOpen(false);
toggleModeDependentFields();
setWorkspaceVisible(false);
clearPreview();
setLoading(false);

copyBtn.addEventListener("click", async () => {
  const text = lastResponse && lastResponse.mode === "structured" && lastResponse.structured
    ? JSON.stringify(lastResponse.structured, null, 2)
    : outputEl.textContent || "";
  if (!text) return;
  await navigator.clipboard.writeText(text);
});

downloadBtn.addEventListener("click", () => {
  if (!lastResponse) return;
  const blob = new Blob([JSON.stringify(lastResponse, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "ocr-ergebnis.json";
  link.click();
  URL.revokeObjectURL(url);
});
