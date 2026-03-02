const form = document.getElementById("ocr-form");
const pageEl = document.querySelector(".page");
const fileEl = document.getElementById("file");
const modeEl = document.getElementById("mode");
const taskEl = document.getElementById("task");
const customPromptEl = document.getElementById("custom_prompt");
const schemaNameEl = document.getElementById("schema_name");
const modelEl = document.getElementById("model");
const tokenLimitEl = document.getElementById("token_limit");
const applyOptionsBtnEl = document.getElementById("apply-options-btn");
const advancedDirtyHintEl = document.getElementById("advanced-dirty-hint");
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
const tablePreviewWrapEl = document.getElementById("table-preview-wrap");
const tablePreviewBodyEl = document.getElementById("table-preview-body");
const structuredWrapEl = document.getElementById("structured-wrap");
const metaEl = document.getElementById("meta");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");
const downloadCsvBtn = document.getElementById("download-csv-btn");
const themeLightBtn = document.getElementById("theme-light-btn");
const themeDarkBtn = document.getElementById("theme-dark-btn");
const loadingOverlayEl = document.getElementById("loading-overlay");
const previewEmptyEl = document.getElementById("preview-empty");
const previewImageEl = document.getElementById("preview-image");
const previewPdfEl = document.getElementById("preview-pdf");
const previewPdfLinkEl = document.getElementById("preview-pdf-link");
const appBasePath = (document.body?.dataset.basePath || "").replace(/\/$/, "");
const ocrEndpoint = `${appBasePath}/api/ocr`;

let lastResponse = null;
let lastTableMatrices = [];
let previewUrl = null;
let activeRequestController = null;
let globalDragDepth = 0;
let hasPendingAdvancedChanges = false;
const THEME_KEY = "ocr-demo-theme";
const MAX_TOKEN_LIMIT = 128000;

function isStructuredMode(modeValue) {
  return modeValue === "structured";
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

function setAdvancedDirty(isDirty) {
  hasPendingAdvancedChanges = isDirty;
  applyOptionsBtnEl.disabled = !isDirty;
  advancedDirtyHintEl.classList.toggle("hidden", !isDirty);
}

function clearOutput() {
  outputEl.textContent = "";
  outputEl.classList.remove("hidden");
  jsonOutputEl.innerHTML = "";
  tablePreviewBodyEl.innerHTML = "";
  tablePreviewWrapEl.classList.add("hidden");
  structuredWrapEl.classList.add("hidden");
  downloadBtn.classList.add("hidden");
  downloadCsvBtn.classList.add("hidden");
  lastResponse = null;
  lastTableMatrices = [];
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

function buildPayload() {
  const payload = new FormData(form);

  const tokenLimitRaw = String(payload.get("token_limit") || "").trim();
  if (tokenLimitRaw) {
    const tokenLimit = Number(tokenLimitRaw);
    if (!Number.isInteger(tokenLimit) || tokenLimit < 1) {
      throw new Error("Token-Limit muss eine positive ganze Zahl sein.");
    }
    if (tokenLimit > MAX_TOKEN_LIMIT) {
      throw new Error(`Token-Limit darf ${MAX_TOKEN_LIMIT} nicht überschreiten.`);
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

function csvCell(value) {
  const raw = value === null || value === undefined ? "" : String(value);
  return `"${raw.replaceAll('"', '""')}"`;
}

function matrixToCsv(matrix) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    return null;
  }
  const width = Math.max(...matrix.map((row) => row.length), 1);
  const normalized = matrix.map((row) =>
    Array.from({ length: width }, (_, idx) => String(row[idx] ?? ""))
  );
  const lines = normalized.map((row) => row.map(csvCell).join(","));
  return `${lines.join("\n")}\n`;
}

function buildTableCsv(structured) {
  return matrixToCsv(buildMatrixFromStructuredTable(structured));
}

function splitMarkdownRow(line) {
  const trimmed = line.trim();
  const withoutLeftPipe = trimmed.startsWith("|") ? trimmed.slice(1) : trimmed;
  const withoutEdgePipes = withoutLeftPipe.endsWith("|")
    ? withoutLeftPipe.slice(0, -1)
    : withoutLeftPipe;
  return withoutEdgePipes.split("|").map((cell) => cell.trim());
}

function isMarkdownSeparatorLine(line) {
  const cells = splitMarkdownRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.replaceAll(" ", "")));
}

function extractMarkdownTableMatrix(raw) {
  const lines = raw.split(/\r?\n/);
  for (let i = 0; i < lines.length - 1; i += 1) {
    if (!lines[i].includes("|") || !lines[i + 1].includes("|")) {
      continue;
    }
    if (!isMarkdownSeparatorLine(lines[i + 1])) {
      continue;
    }

    const header = splitMarkdownRow(lines[i]);
    const rows = [];
    for (let j = i + 2; j < lines.length; j += 1) {
      const rowLine = lines[j];
      if (!rowLine.trim() || !rowLine.includes("|")) {
        break;
      }
      if (isMarkdownSeparatorLine(rowLine)) {
        continue;
      }
      rows.push(splitMarkdownRow(rowLine));
    }

    if (header.length === 0) {
      continue;
    }
    const width = Math.max(header.length, ...rows.map((row) => row.length), 1);
    const normalizedHeader = Array.from({ length: width }, (_, idx) => header[idx] ?? "");
    const normalizedRows = rows.map((row) =>
      Array.from({ length: width }, (_, idx) => row[idx] ?? "")
    );
    return [normalizedHeader, ...normalizedRows];
  }
  return null;
}

function extractHtmlTableMatrices(raw) {
  if (!/<table[\s>]/i.test(raw)) {
    return [];
  }
  const parser = new DOMParser();
  const documentRoot = parser.parseFromString(raw, "text/html");
  const tables = Array.from(documentRoot.querySelectorAll("table"));
  const matrices = [];

  for (const table of tables) {
    const rows = Array.from(table.querySelectorAll("tr")).map((tr) =>
      Array.from(tr.querySelectorAll("th,td")).map((cell) => {
        const clone = cell.cloneNode(true);
        if (clone && typeof clone.querySelectorAll === "function") {
          clone.querySelectorAll("br").forEach((br) => {
            br.replaceWith("\n");
          });
          return (clone.textContent || "")
            .replace(/\r/g, "")
            .replace(/[ \t]+\n/g, "\n")
            .replace(/\n[ \t]+/g, "\n")
            .trim();
        }
        return (cell.textContent || "").trim();
      })
    );
    const nonEmptyRows = rows.filter((row) => row.some((cell) => cell.length > 0));
    if (nonEmptyRows.length > 0) {
      matrices.push(nonEmptyRows);
    }
  }
  return matrices;
}

function buildMatrixFromStructuredTable(structured) {
  if (!structured || typeof structured !== "object") {
    return null;
  }
  const columns = Array.isArray(structured.columns) ? structured.columns : [];
  const rows = Array.isArray(structured.rows) ? structured.rows : [];
  if (columns.length === 0 && rows.length === 0) {
    return null;
  }

  const width = Math.max(
    columns.length,
    ...rows.map((row) => (Array.isArray(row) ? row.length : 0)),
    1
  );
  const header = (columns.length ? columns : Array.from({ length: width }, (_, i) => `col_${i + 1}`))
    .slice(0, width)
    .map((cell) => String(cell ?? ""));
  const normalizedRows = rows.map((row) => {
    const rowValues = Array.isArray(row) ? row : [];
    return Array.from({ length: width }, (_, idx) => String(rowValues[idx] ?? ""));
  });

  return [header, ...normalizedRows];
}

function matrixToMarkdown(matrix) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    return "";
  }
  const width = Math.max(...matrix.map((row) => row.length), 1);
  const normalized = matrix.map((row) =>
    Array.from({ length: width }, (_, idx) => String(row[idx] ?? "").replaceAll("|", "\\|"))
  );

  const header = normalized[0];
  const separator = Array.from({ length: width }, () => "---");
  const lines = [
    `| ${header.join(" | ")} |`,
    `| ${separator.join(" | ")} |`,
  ];
  for (const row of normalized.slice(1)) {
    lines.push(`| ${row.join(" | ")} |`);
  }
  return lines.join("\n");
}

function matricesToMarkdown(matrices) {
  if (!Array.isArray(matrices) || matrices.length === 0) {
    return "";
  }
  if (matrices.length === 1) {
    return matrixToMarkdown(matrices[0]);
  }
  return matrices
    .map((matrix, idx) => `Tabelle ${idx + 1}\n${matrixToMarkdown(matrix)}`)
    .join("\n\n");
}

function renderTablePreview(matrices) {
  if (!Array.isArray(matrices) || matrices.length === 0) {
    tablePreviewBodyEl.innerHTML = "";
    tablePreviewWrapEl.classList.add("hidden");
    return;
  }

  tablePreviewBodyEl.innerHTML = "";
  matrices.forEach((matrix, idx) => {
    if (!Array.isArray(matrix) || matrix.length === 0) {
      return;
    }
    if (matrices.length > 1) {
      const label = document.createElement("p");
      label.className = "table-preview-label";
      label.textContent = `Tabelle ${idx + 1}`;
      tablePreviewBodyEl.appendChild(label);
    }

    const block = document.createElement("div");
    block.className = "table-preview-block";
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const tbody = document.createElement("tbody");
    const headerRow = document.createElement("tr");

    const width = Math.max(...matrix.map((row) => row.length), 1);
    const normalized = matrix.map((row) =>
      Array.from({ length: width }, (_, cellIdx) => String(row[cellIdx] ?? ""))
    );

    normalized[0].forEach((value) => {
      const th = document.createElement("th");
      th.textContent = value;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    normalized.slice(1).forEach((row) => {
      const tr = document.createElement("tr");
      row.forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    table.appendChild(thead);
    table.appendChild(tbody);
    block.appendChild(table);
    tablePreviewBodyEl.appendChild(block);
  });
  tablePreviewWrapEl.classList.remove("hidden");
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
    const payload = buildPayload();
    const requestMode = String(payload.get("mode") || "plain");
    const requestTask = String(payload.get("task") || "");
    const response = await fetch(ocrEndpoint, {
      method: "POST",
      body: payload,
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "OCR fehlgeschlagen");
    }

    lastResponse = data;
    setAdvancedDirty(false);
    let displayText = data.text || "(kein Inhalt)";
    let tableMatrices = [];

    if (requestMode === "plain" && requestTask === "extract_table_markdown") {
      const htmlTables = extractHtmlTableMatrices(displayText);
      if (htmlTables.length > 0) {
        tableMatrices = htmlTables;
        displayText = matricesToMarkdown(htmlTables);
      } else {
        const markdownTable = extractMarkdownTableMatrix(displayText);
        if (markdownTable) {
          tableMatrices = [markdownTable];
        }
      }
    }
    if (data.mode === "structured" && data.schema_name === "table_basic") {
      const structuredTable = buildMatrixFromStructuredTable(data.structured);
      if (structuredTable) {
        tableMatrices = [structuredTable];
      }
    }

    const showStructured = data.mode === "structured" && !!data.structured;
    const showTableCsv = tableMatrices.length > 0;
    if (showStructured) {
      outputEl.textContent = "";
      outputEl.classList.add("hidden");
      tablePreviewBodyEl.innerHTML = "";
      tablePreviewWrapEl.classList.add("hidden");
      lastTableMatrices = [];
    } else {
      outputEl.classList.remove("hidden");
      outputEl.textContent = displayText;
      renderTablePreview(tableMatrices);
      lastTableMatrices = tableMatrices;
    }
    structuredWrapEl.classList.toggle("hidden", !showStructured);
    downloadBtn.classList.toggle("hidden", !showStructured);
    downloadCsvBtn.classList.toggle("hidden", !showTableCsv);
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
  setWorkspaceVisible(!!currentFile());
  updatePreview();
  void runOCR();
});
modeEl.addEventListener("change", () => {
  toggleModeDependentFields();
  setAdvancedDirty(true);
});
taskEl.addEventListener("change", () => {
  setAdvancedDirty(true);
});
schemaNameEl.addEventListener("change", () => {
  setAdvancedDirty(true);
});
modelEl.addEventListener("input", () => {
  setAdvancedDirty(true);
});
tokenLimitEl.addEventListener("input", () => {
  setAdvancedDirty(true);
});
customPromptEl.addEventListener("input", () => {
  setAdvancedDirty(true);
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  void runOCR();
});
applyOptionsBtnEl.addEventListener("click", () => {
  if (!hasPendingAdvancedChanges) {
    return;
  }
  if (!currentFile()) {
    setAdvancedDirty(false);
    return;
  }
  void runOCR();
});
initTheme();
setAdvancedOpen(false);
toggleModeDependentFields();
setWorkspaceVisible(false);
clearPreview();
setLoading(false);
setAdvancedDirty(false);

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

downloadCsvBtn.addEventListener("click", () => {
  if (lastTableMatrices.length > 0) {
    const csv = matrixToCsv(lastTableMatrices[0]);
    if (!csv) return;
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "ocr-tabelle.csv";
    link.click();
    URL.revokeObjectURL(url);
    return;
  }
  if (!lastResponse || lastResponse.schema_name !== "table_basic") return;
  const csv = buildTableCsv(lastResponse.structured);
  if (!csv) return;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "ocr-tabelle.csv";
  link.click();
  URL.revokeObjectURL(url);
});
