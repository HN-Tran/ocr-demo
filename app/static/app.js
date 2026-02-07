const form = document.getElementById("ocr-form");
const fileEl = document.getElementById("file");
const modeEl = document.getElementById("mode");
const taskWrap = document.getElementById("task-wrap");
const customPromptWrap = document.getElementById("custom-prompt-wrap");
const schemaWrap = document.getElementById("schema-wrap");
const outputEl = document.getElementById("output");
const jsonOutputEl = document.getElementById("json-output");
const metaEl = document.getElementById("meta");
const runBtn = document.getElementById("run-btn");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");
const themeBtn = document.getElementById("theme-btn");
const previewEmptyEl = document.getElementById("preview-empty");
const previewImageEl = document.getElementById("preview-image");
const previewPdfEl = document.getElementById("preview-pdf");
const previewPdfLinkEl = document.getElementById("preview-pdf-link");

let lastResponse = null;
let previewUrl = null;
const THEME_KEY = "ocr-demo-theme";

function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "dark") {
    root.setAttribute("data-theme", "dark");
    themeBtn.textContent = "Light mode";
  } else {
    root.removeAttribute("data-theme");
    themeBtn.textContent = "Dark mode";
  }
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
  const isStructured = modeEl.value === "structured";
  schemaWrap.classList.toggle("hidden", !isStructured);
  taskWrap.classList.toggle("hidden", isStructured);
  customPromptWrap.classList.toggle("hidden", isStructured);
}

function clearPreview(message = "No file selected.") {
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

  clearPreview(`Preview not available for "${file.type || "unknown"}".`);
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

themeBtn.addEventListener("click", () => {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const nextTheme = isDark ? "light" : "dark";
  localStorage.setItem(THEME_KEY, nextTheme);
  applyTheme(nextTheme);
});

modeEl.addEventListener("change", toggleModeDependentFields);
fileEl.addEventListener("change", updatePreview);
initTheme();
toggleModeDependentFields();
clearPreview();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  outputEl.textContent = "";
  jsonOutputEl.innerHTML = "";
  metaEl.textContent = "Running OCR...";
  lastResponse = null;

  const payload = new FormData(form);
  const selectedFile = fileEl.files && fileEl.files[0];
  if (selectedFile && selectedFile.type === "application/pdf") {
    metaEl.textContent = "PDF preview is supported, but OCR currently supports PNG/JPEG/WEBP.";
    return;
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
    payload.delete("task");
    payload.delete("custom_prompt");
  }
  if (!payload.get("model")) {
    payload.delete("model");
  }

  runBtn.disabled = true;
  try {
    const response = await fetch("/api/ocr", {
      method: "POST",
      body: payload,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "OCR failed");
    }
    lastResponse = data;
    outputEl.textContent = data.text || "";
    if (data.structured) {
      jsonOutputEl.innerHTML = highlightJson(data.structured);
    } else {
      jsonOutputEl.textContent = "(no structured payload)";
    }
    const warnings = (data.warnings || []).join(" | ");
    metaEl.textContent = `Model: ${data.model} | Latency: ${data.latency_ms} ms${warnings ? ` | Warnings: ${warnings}` : ""}`;
  } catch (error) {
    metaEl.textContent = `Error: ${error.message}`;
  } finally {
    runBtn.disabled = false;
  }
});

copyBtn.addEventListener("click", async () => {
  const text = lastResponse && lastResponse.structured
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
  link.download = "ocr-result.json";
  link.click();
  URL.revokeObjectURL(url);
});
