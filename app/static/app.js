const form = document.getElementById("ocr-form");
const pageEl = document.querySelector(".page");
const fileEl = document.getElementById("file");
const modeEl = document.getElementById("mode");
const taskEl = document.getElementById("task");
const customPromptEl = document.getElementById("custom_prompt");
const schemaNameEl = document.getElementById("schema_name");
const backendEl = document.getElementById("backend");
const expertEnableLayoutEl = document.getElementById("expert_enable_layout");
const modelEl = document.getElementById("model");
const tokenLimitEl = document.getElementById("token_limit");
const gifMaxFramesEl = document.getElementById("gif_max_frames");
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
const rawWrapEl = document.getElementById("raw-wrap");
const markdownPreviewWrapEl = document.getElementById("markdown-preview-wrap");
const markdownPreviewEl = document.getElementById("markdown-preview");
const resultViewSwitchEl = document.getElementById("result-view-switch");
const resultViewLayoutBtnEl = document.getElementById("result-view-layout-btn");
const resultViewMarkdownBtnEl = document.getElementById("result-view-markdown-btn");
const resultViewRawBtnEl = document.getElementById("result-view-raw-btn");
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
const previewImageStageEl = document.getElementById("preview-image-stage");
const previewImageEl = document.getElementById("preview-image");
const previewLayoutOverlayEl = document.getElementById("preview-layout-overlay");
const previewPdfEl = document.getElementById("preview-pdf");
const previewPdfLinkEl = document.getElementById("preview-pdf-link");
const layoutWrapEl = document.getElementById("layout-wrap");
const layoutSummaryEl = document.getElementById("layout-summary");
const layoutPagesEl = document.getElementById("layout-pages");
const layoutVisualizationsEl = document.getElementById("layout-visualizations");
const appBasePath = (document.body?.dataset.basePath || "").replace(/\/$/, "");
const ocrEndpoint = `${appBasePath}/api/ocr`;

let lastResponse = null;
let lastTableMatrices = [];
let previewUrl = null;
let activeRequestController = null;
let globalDragDepth = 0;
let hasPendingAdvancedChanges = false;
let activeResultView = "layout";
const THEME_KEY = "ocr-demo-theme";
const MAX_TOKEN_LIMIT = 128000;
const MAX_GIF_FRAMES = 32;
const INLINE_PREVIEWABLE_IMAGE_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
]);
const TIFF_IMAGE_TYPES = new Set(["image/tif", "image/tiff", "image/x-tiff"]);
const LAYOUT_REGION_KIND_ALIASES = new Map([
  ["text", "text"],
  ["text_block", "text"],
  ["textblock", "text"],
  ["paragraph", "text"],
  ["body", "text"],
  ["body_text", "text"],
  ["plain_text", "text"],
  ["line", "text"],
  ["table", "table"],
  ["table_block", "table"],
  ["table_region", "table"],
  ["title", "title"],
  ["heading", "title"],
  ["header_title", "title"],
  ["section_title", "title"],
  ["figure", "figure"],
  ["image", "figure"],
  ["picture", "figure"],
  ["illustration", "figure"],
  ["chart", "figure"],
  ["caption", "caption"],
  ["list", "list"],
  ["list_item", "list"],
  ["bullet_list", "list"],
  ["header", "header"],
  ["page_header", "header"],
  ["footer", "footer"],
  ["page_footer", "footer"],
  ["footnote", "footer"],
  ["formula", "formula"],
  ["equation", "formula"],
  ["math", "formula"],
  ["code", "code"],
  ["barcode", "code"],
  ["qr_code", "code"],
  ["signature", "signature"],
  ["stamp", "signature"],
]);
const RESULT_VIEW_BUTTONS = {
  layout: resultViewLayoutBtnEl,
  markdown: resultViewMarkdownBtnEl,
  raw: resultViewRawBtnEl,
};
const RESULT_VIEW_WRAPS = {
  layout: layoutWrapEl,
  markdown: markdownPreviewWrapEl,
  raw: rawWrapEl,
};

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
  rawWrapEl.classList.add("hidden");
  clearMarkdownPreview();
  clearResultViewSwitch();
  jsonOutputEl.innerHTML = "";
  tablePreviewBodyEl.innerHTML = "";
  tablePreviewWrapEl.classList.add("hidden");
  structuredWrapEl.classList.add("hidden");
  downloadBtn.classList.add("hidden");
  downloadCsvBtn.classList.add("hidden");
  lastResponse = null;
  lastTableMatrices = [];
  clearLayoutDisplay();
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
  previewImageStageEl.classList.add("hidden");
  previewImageEl.classList.add("hidden");
  previewImageEl.removeAttribute("src");
  clearLayoutOverlay();
  previewPdfEl.classList.add("hidden");
  previewPdfEl.removeAttribute("data");
  previewPdfLinkEl.classList.add("hidden");
  previewPdfLinkEl.removeAttribute("href");
}

function resetPreviewSurfaces() {
  previewImageStageEl.classList.add("hidden");
  previewImageEl.classList.add("hidden");
  previewImageEl.removeAttribute("src");
  previewPdfEl.classList.add("hidden");
  previewPdfEl.removeAttribute("data");
  previewPdfLinkEl.classList.add("hidden");
  previewPdfLinkEl.removeAttribute("href");
}

function fileExtension(name) {
  const normalized = String(name || "").trim().toLowerCase();
  const dotIndex = normalized.lastIndexOf(".");
  if (dotIndex < 0) {
    return "";
  }
  return normalized.slice(dotIndex);
}

function isTiffLikeFile(file) {
  if (!file) {
    return false;
  }
  if (TIFF_IMAGE_TYPES.has(file.type)) {
    return true;
  }
  const extension = fileExtension(file.name);
  return extension === ".tif" || extension === ".tiff";
}

function isInlinePreviewableImage(file) {
  if (!file) {
    return false;
  }
  return INLINE_PREVIEWABLE_IMAGE_TYPES.has(file.type);
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
  clearLayoutOverlay();
  resetPreviewSurfaces();

  if (isInlinePreviewableImage(file)) {
    previewImageEl.src = previewUrl;
    previewImageStageEl.classList.remove("hidden");
    previewImageEl.classList.remove("hidden");
    return;
  }

  if (file.type === "application/pdf") {
    const pdfUrl = previewUrl;
    previewPdfEl.classList.remove("hidden");
    window.requestAnimationFrame(() => {
      if (previewUrl === pdfUrl) {
        previewPdfEl.data = pdfUrl;
      }
    });
    previewPdfLinkEl.href = previewUrl;
    previewPdfLinkEl.textContent = "PDF in neuem Tab öffnen";
    previewPdfLinkEl.classList.remove("hidden");
    return;
  }

  if (isTiffLikeFile(file)) {
    previewEmptyEl.textContent =
      "TIFF-Vorschau wird vom Browser nicht zuverlässig unterstützt. OCR läuft trotzdem.";
    previewEmptyEl.classList.remove("hidden");
    previewPdfLinkEl.href = previewUrl;
    previewPdfLinkEl.textContent = "Datei in neuem Tab öffnen";
    previewPdfLinkEl.classList.remove("hidden");
    return;
  }

  clearPreview(`Für "${file.type || "unbekannt"}" ist keine Vorschau verfügbar.`);
}

function currentFile() {
  return fileEl.files && fileEl.files[0] ? fileEl.files[0] : null;
}

function clearLayoutOverlay() {
  previewLayoutOverlayEl.innerHTML = "";
  previewLayoutOverlayEl.classList.add("hidden");
}

function clearLayoutDisplay() {
  clearLayoutOverlay();
  layoutSummaryEl.textContent = "";
  layoutPagesEl.innerHTML = "";
  layoutVisualizationsEl.innerHTML = "";
  layoutVisualizationsEl.classList.add("hidden");
  layoutWrapEl.classList.add("hidden");
}

function clearResultViewSwitch() {
  resultViewSwitchEl.classList.add("hidden");
  Object.entries(RESULT_VIEW_BUTTONS).forEach(([viewName, buttonEl]) => {
    buttonEl.classList.add("hidden");
    buttonEl.classList.remove("is-active");
    buttonEl.setAttribute("aria-pressed", "false");
    const wrapEl = RESULT_VIEW_WRAPS[viewName];
    if (wrapEl) {
      wrapEl.classList.add("hidden");
    }
  });
}

function setActiveResultView(viewName, availableViews) {
  activeResultView = viewName;
  Object.entries(RESULT_VIEW_BUTTONS).forEach(([candidateView, buttonEl]) => {
    const isVisible = availableViews.includes(candidateView);
    const isActive = candidateView === viewName && isVisible;
    buttonEl.classList.toggle("hidden", !isVisible);
    buttonEl.classList.toggle("is-active", isActive);
    buttonEl.setAttribute("aria-pressed", String(isActive));
    const wrapEl = RESULT_VIEW_WRAPS[candidateView];
    if (wrapEl) {
      wrapEl.classList.toggle("hidden", !isActive);
    }
  });
  resultViewSwitchEl.classList.toggle("hidden", availableViews.length <= 1);
}

function configurePlainResultViews({ hasLayout, hasMarkdown }) {
  const availableViews = [];
  if (hasLayout) {
    availableViews.push("layout");
  }
  if (hasMarkdown) {
    availableViews.push("markdown");
  }
  availableViews.push("raw");
  const defaultView = hasLayout ? "layout" : hasMarkdown ? "markdown" : "raw";
  setActiveResultView(defaultView, availableViews);
}

function truncateText(value, maxLength = 140) {
  const normalized = String(value || "").trim().replace(/\s+/g, " ");
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 1)}…`;
}

function normalizeLayoutRegionKind(label) {
  const normalized = String(label || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!normalized) {
    return "other";
  }
  return LAYOUT_REGION_KIND_ALIASES.get(normalized) || "other";
}

function normalizeLayoutPages(layout) {
  if (!Array.isArray(layout)) {
    return [];
  }
  return layout.filter(
    (page) => page && typeof page === "object" && Array.isArray(page.regions)
  );
}

function formatConfidence(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue) || numericValue < 0) {
    return null;
  }
  if (numericValue <= 1) {
    return `${(numericValue * 100).toFixed(1)}%`;
  }
  return numericValue.toFixed(2);
}

function getRegionConfidence(region) {
  if (!region || typeof region !== "object") {
    return null;
  }

  const rawValue =
    region.confidence !== undefined && region.confidence !== null
      ? region.confidence
      : region.score;
  const numericValue = Number(rawValue);
  if (!Number.isFinite(numericValue) || numericValue < 0) {
    return null;
  }
  return numericValue;
}

function collectLayoutConfidenceStats(layoutPages) {
  const values = [];
  layoutPages.forEach((page) => {
    const regions = Array.isArray(page?.regions) ? page.regions : [];
    regions.forEach((region) => {
      const numericValue = getRegionConfidence(region);
      if (numericValue !== null) {
        values.push(numericValue);
      }
    });
  });
  if (values.length === 0) {
    return null;
  }

  const total = values.reduce((sum, value) => sum + value, 0);
  return {
    count: values.length,
    average: total / values.length,
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

function getLayoutCoordinateScale(values) {
  return Math.max(...values.map((value) => Math.abs(value))) <= 1.5 ? 1 : 1000;
}

function normalizedBboxToPercentages(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) {
    return null;
  }
  const values = bbox.map((value) => Number(value));
  if (values.some((value) => !Number.isFinite(value))) {
    return null;
  }
  const scale = getLayoutCoordinateScale(values);
  const left = Math.max(0, Math.min(100, (values[0] / scale) * 100));
  const top = Math.max(0, Math.min(100, (values[1] / scale) * 100));
  const right = Math.max(0, Math.min(100, (values[2] / scale) * 100));
  const bottom = Math.max(0, Math.min(100, (values[3] / scale) * 100));
  if (right <= left || bottom <= top) {
    return null;
  }
  return {
    left,
    top,
    width: right - left,
    height: bottom - top,
  };
}

function normalizedPolygonToPercentages(polygon) {
  if (!Array.isArray(polygon) || polygon.length < 8 || polygon.length % 2 !== 0) {
    return null;
  }
  const values = polygon.map((value) => Number(value));
  if (values.some((value) => !Number.isFinite(value))) {
    return null;
  }

  const scale = getLayoutCoordinateScale(values);
  const points = [];
  for (let index = 0; index < values.length; index += 2) {
    points.push({
      x: Math.max(0, Math.min(100, (values[index] / scale) * 100)),
      y: Math.max(0, Math.min(100, (values[index + 1] / scale) * 100)),
    });
  }
  return points.length >= 4 ? points : null;
}

function polygonPointsToBounds(points) {
  if (!Array.isArray(points) || points.length < 3) {
    return null;
  }
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const left = Math.max(0, Math.min(...xs));
  const top = Math.max(0, Math.min(...ys));
  const right = Math.min(100, Math.max(...xs));
  const bottom = Math.min(100, Math.max(...ys));
  if (right <= left || bottom <= top) {
    return null;
  }
  return {
    left,
    top,
    width: right - left,
    height: bottom - top,
  };
}

function getPolygonBadgeAnchors(points) {
  if (!Array.isArray(points) || points.length < 3) {
    return null;
  }

  let topLeft = points[0];
  let topRight = points[0];
  for (const point of points) {
    if (point.x + point.y < topLeft.x + topLeft.y) {
      topLeft = point;
    }
    if (point.x - point.y > topRight.x - topRight.y) {
      topRight = point;
    }
  }

  return { topLeft, topRight };
}

function renderLayoutOverlay(layoutPages) {
  clearLayoutOverlay();
  const file = currentFile();
  if (!file || !file.type.startsWith("image/")) {
    return;
  }

  const firstPage = layoutPages[0];
  if (!firstPage || !Array.isArray(firstPage.regions)) {
    return;
  }

  const svgNs = "http://www.w3.org/2000/svg";
  const svgEl = document.createElementNS(svgNs, "svg");
  svgEl.setAttribute("viewBox", "0 0 100 100");
  svgEl.setAttribute("preserveAspectRatio", "none");
  svgEl.classList.add("preview-layout-svg");
  previewLayoutOverlayEl.appendChild(svgEl);

  let overlayCount = 0;
  firstPage.regions.forEach((region, index) => {
    const polygonPoints = normalizedPolygonToPercentages(region.polygon);
    const percentages = polygonPoints
      ? polygonPointsToBounds(polygonPoints)
      : normalizedBboxToPercentages(region.bbox_2d);
    if (!percentages) {
      return;
    }

    const label = String(region.label || `Region ${index + 1}`);
    const regionKind = normalizeLayoutRegionKind(label);

    const shapePoints = polygonPoints || [
      { x: percentages.left, y: percentages.top },
      { x: percentages.left + percentages.width, y: percentages.top },
      { x: percentages.left + percentages.width, y: percentages.top + percentages.height },
      { x: percentages.left, y: percentages.top + percentages.height },
    ];
    const shapeEl = document.createElementNS(svgNs, "polygon");
    shapeEl.classList.add("preview-layout-shape");
    shapeEl.dataset.regionKind = regionKind;
    shapeEl.setAttribute(
      "points",
      shapePoints.map((point) => `${point.x},${point.y}`).join(" ")
    );
    svgEl.appendChild(shapeEl);

    const boxEl = document.createElement("div");
    boxEl.className = "preview-layout-box";
    if (polygonPoints) {
      boxEl.classList.add("is-polygon");
    }
    boxEl.style.left = `${percentages.left}%`;
    boxEl.style.top = `${percentages.top}%`;
    boxEl.style.width = `${percentages.width}%`;
    boxEl.style.height = `${percentages.height}%`;

    boxEl.dataset.regionKind = regionKind;
    const contentPreview = truncateText(region.content || "", 80);
    const confidenceValue = getRegionConfidence(region);
    const confidenceLabel = formatConfidence(confidenceValue);
    const labelWithConfidence = confidenceLabel ? `${label} (${confidenceLabel})` : label;
    boxEl.title = contentPreview ? `${labelWithConfidence}: ${contentPreview}` : labelWithConfidence;

    const badgeEl = document.createElement("span");
    badgeEl.className = "preview-layout-badge";
    badgeEl.dataset.regionKind = regionKind;
    badgeEl.textContent = confidenceLabel ? `${label} ${confidenceLabel}` : label;
    if (polygonPoints) {
      const anchors = getPolygonBadgeAnchors(polygonPoints);
      if (anchors && percentages.width > 0 && percentages.height > 0) {
        const relativeLeft = ((anchors.topLeft.x - percentages.left) / percentages.width) * 100;
        const relativeTop = ((anchors.topLeft.y - percentages.top) / percentages.height) * 100;
        badgeEl.style.left = `${relativeLeft}%`;
        badgeEl.style.top = `${relativeTop}%`;
      }
    }
    boxEl.appendChild(badgeEl);
    previewLayoutOverlayEl.appendChild(boxEl);
    overlayCount += 1;
  });

  previewLayoutOverlayEl.classList.toggle("hidden", overlayCount === 0);
}

function renderLayoutVisualizations(visualizations) {
  layoutVisualizationsEl.innerHTML = "";
  if (!Array.isArray(visualizations) || visualizations.length === 0) {
    layoutVisualizationsEl.classList.add("hidden");
    return;
  }

  visualizations.forEach((source, index) => {
    const imgEl = document.createElement("img");
    imgEl.src = String(source);
    imgEl.alt = `Layout-Visualisierung ${index + 1}`;
    imgEl.loading = "lazy";
    layoutVisualizationsEl.appendChild(imgEl);
  });
  layoutVisualizationsEl.classList.remove("hidden");
}

function renderLayoutPanel(layoutPages, visualizations) {
  clearLayoutDisplay();
  if (layoutPages.length === 0 && (!Array.isArray(visualizations) || visualizations.length === 0)) {
    return;
  }

  const regionCount = layoutPages.reduce(
    (total, page) => total + (Array.isArray(page.regions) ? page.regions.length : 0),
    0
  );
  const confidenceStats = collectLayoutConfidenceStats(layoutPages);
  const summaryParts = [];
  if (layoutPages.length > 0) {
    summaryParts.push(`${layoutPages.length} Seite(n)`);
    summaryParts.push(`${regionCount} Region(en)`);
  }
  if (confidenceStats) {
    summaryParts.push(
      `Vertrauen: Ø ${formatConfidence(confidenceStats.average)}`
    );
    summaryParts.push(
      `${confidenceStats.count} Score${confidenceStats.count === 1 ? "" : "s"}`
    );
  }
  if (Array.isArray(visualizations) && visualizations.length > 0) {
    summaryParts.push(`${visualizations.length} Visualisierung(en)`);
  }
  layoutSummaryEl.textContent = summaryParts.join(" | ");

  layoutPages.forEach((page, pageIndex) => {
    const pageCardEl = document.createElement("article");
    pageCardEl.className = "layout-page-card";

    const pageTitleEl = document.createElement("p");
    pageTitleEl.className = "layout-page-title";
    pageTitleEl.textContent = `Seite ${page.page_number || pageIndex + 1}`;
    pageCardEl.appendChild(pageTitleEl);

    const regions = Array.isArray(page.regions) ? page.regions : [];
    if (regions.length === 0) {
      const emptyEl = document.createElement("p");
      emptyEl.className = "layout-page-empty";
      emptyEl.textContent = "Keine Regionen im Layout-Output.";
      pageCardEl.appendChild(emptyEl);
      layoutPagesEl.appendChild(pageCardEl);
      return;
    }

    const regionListEl = document.createElement("ol");
    regionListEl.className = "layout-region-list";

    regions.forEach((region, regionIndex) => {
      const regionItemEl = document.createElement("li");
      regionItemEl.className = "layout-region-item";
      const label = String(region.label || `Region ${regionIndex + 1}`);
      const regionKind = normalizeLayoutRegionKind(label);
      regionItemEl.dataset.regionKind = regionKind;

      const regionHeadEl = document.createElement("div");
      regionHeadEl.className = "layout-region-head";

      const labelEl = document.createElement("strong");
      labelEl.className = "layout-region-label";
      labelEl.dataset.regionKind = regionKind;
      labelEl.textContent = label;
      regionHeadEl.appendChild(labelEl);

      const confidenceLabel = formatConfidence(getRegionConfidence(region));
      if (confidenceLabel) {
        const confidenceEl = document.createElement("span");
        confidenceEl.className = "layout-region-confidence";
        confidenceEl.textContent = confidenceLabel;
        regionHeadEl.appendChild(confidenceEl);
      }

      const metaEl = document.createElement("span");
      metaEl.className = "layout-region-meta";
      const bbox = Array.isArray(region.bbox_2d) ? region.bbox_2d.join(", ") : "ohne bbox";
      metaEl.textContent = `#${region.index ?? regionIndex} | bbox: ${bbox}`;
      regionHeadEl.appendChild(metaEl);
      regionItemEl.appendChild(regionHeadEl);

      if (region.content) {
        const contentEl = document.createElement("p");
        contentEl.className = "layout-region-content";
        contentEl.textContent = truncateText(region.content, 220);
        regionItemEl.appendChild(contentEl);
      }

      regionListEl.appendChild(regionItemEl);
    });

    pageCardEl.appendChild(regionListEl);
    layoutPagesEl.appendChild(pageCardEl);
  });

  renderLayoutVisualizations(visualizations);
  layoutWrapEl.classList.remove("hidden");
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

  const gifMaxFramesRaw = String(payload.get("gif_max_frames") || "").trim();
  if (gifMaxFramesRaw) {
    const gifMaxFrames = Number(gifMaxFramesRaw);
    if (!Number.isInteger(gifMaxFrames) || gifMaxFrames < 1) {
      throw new Error("GIF-Frames muss eine positive ganze Zahl sein.");
    }
    if (gifMaxFrames > MAX_GIF_FRAMES) {
      throw new Error(`GIF-Frames darf ${MAX_GIF_FRAMES} nicht überschreiten.`);
    }
    payload.set("gif_max_frames", String(gifMaxFrames));
  } else {
    payload.delete("gif_max_frames");
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
  const backendValue = String(payload.get("backend") || "").trim();
  if (backendValue) {
    payload.set("backend", backendValue);
  } else {
    payload.delete("backend");
  }
  const expertLayoutValue = String(payload.get("expert_enable_layout") || "").trim();
  if (expertLayoutValue === "true" || expertLayoutValue === "false") {
    payload.set("expert_enable_layout", expertLayoutValue);
  } else {
    payload.delete("expert_enable_layout");
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
    const tableBlock = buildTableBlock(matrix, matrices.length > 1 ? `Tabelle ${idx + 1}` : null);
    if (tableBlock) {
      tablePreviewBodyEl.appendChild(tableBlock);
    }
  });
  tablePreviewWrapEl.classList.remove("hidden");
}

function buildTableBlock(matrix, labelText = null) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    return null;
  }

  const fragment = document.createDocumentFragment();
  if (labelText) {
    const label = document.createElement("p");
    label.className = "table-preview-label";
    label.textContent = labelText;
    fragment.appendChild(label);
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
  fragment.appendChild(block);
  return fragment;
}

function clearMarkdownPreview() {
  markdownPreviewEl.innerHTML = "";
  markdownPreviewWrapEl.classList.add("hidden");
}

function appendMarkdownParagraph(container, lines) {
  const paragraphText = lines.join("\n").trim();
  if (!paragraphText) {
    return;
  }
  const paragraphEl = document.createElement("p");
  paragraphEl.className = "markdown-preview-paragraph";
  paragraphEl.textContent = paragraphText;
  container.appendChild(paragraphEl);
}

function appendMarkdownCodeBlock(container, language, code) {
  const blockEl = document.createElement("pre");
  blockEl.className = "markdown-preview-code";
  if (language) {
    blockEl.dataset.language = language;
  }
  const codeEl = document.createElement("code");
  codeEl.textContent = code;
  blockEl.appendChild(codeEl);
  container.appendChild(blockEl);
}

function appendMarkdownList(container, items, ordered) {
  if (!Array.isArray(items) || items.length === 0) {
    return;
  }
  const listEl = document.createElement(ordered ? "ol" : "ul");
  listEl.className = "markdown-preview-list";
  items.forEach((item) => {
    const itemEl = document.createElement("li");
    itemEl.textContent = item;
    listEl.appendChild(itemEl);
  });
  container.appendChild(listEl);
}

function appendMarkdownImageReference(container, referenceText) {
  const itemEl = document.createElement("p");
  itemEl.className = "markdown-preview-image-ref";
  itemEl.textContent = referenceText;
  container.appendChild(itemEl);
}

function appendHtmlTableBlocks(container, htmlSource) {
  const matrices = extractHtmlTableMatrices(htmlSource);
  matrices.forEach((matrix, index) => {
    const block = buildTableBlock(matrix, matrices.length > 1 ? `Tabelle ${index + 1}` : null);
    if (block) {
      container.appendChild(block);
    }
  });
}

function renderMarkdownPreview(markdown) {
  clearMarkdownPreview();
  const normalized = String(markdown || "").trim();
  if (!normalized) {
    return;
  }

  const htmlTableBlocks = [];
  const source = normalized.replace(/<table[\s\S]*?<\/table>/gi, (match) => {
    const blockIndex = htmlTableBlocks.push(match) - 1;
    return `\n\n[[[OCR_HTML_TABLE_${blockIndex}]]]\n\n`;
  });
  const lines = source.split(/\r?\n/);
  const fragment = document.createDocumentFragment();
  let paragraphLines = [];

  const flushParagraph = () => {
    if (paragraphLines.length > 0) {
      appendMarkdownParagraph(fragment, paragraphLines);
      paragraphLines = [];
    }
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      continue;
    }

    const htmlTableMatch = trimmed.match(/^\[\[\[OCR_HTML_TABLE_(\d+)]]]\s*$/);
    if (htmlTableMatch) {
      flushParagraph();
      appendHtmlTableBlocks(fragment, htmlTableBlocks[Number(htmlTableMatch[1])] || "");
      continue;
    }

    if (trimmed.startsWith("```")) {
      flushParagraph();
      const language = trimmed.slice(3).trim();
      const codeLines = [];
      for (index += 1; index < lines.length; index += 1) {
        if (lines[index].trim().startsWith("```")) {
          break;
        }
        codeLines.push(lines[index]);
      }
      appendMarkdownCodeBlock(fragment, language, codeLines.join("\n"));
      continue;
    }

    if (index + 1 < lines.length && line.includes("|") && isMarkdownSeparatorLine(lines[index + 1])) {
      flushParagraph();
      const tableLines = [line, lines[index + 1]];
      for (index += 2; index < lines.length; index += 1) {
        const tableLine = lines[index];
        if (!tableLine.trim() || !tableLine.includes("|")) {
          index -= 1;
          break;
        }
        tableLines.push(tableLine);
      }
      const matrix = extractMarkdownTableMatrix(tableLines.join("\n"));
      const block = buildTableBlock(matrix);
      if (block) {
        fragment.appendChild(block);
      }
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      const level = Math.min(headingMatch[1].length + 1, 6);
      const headingEl = document.createElement(`h${level}`);
      headingEl.className = "markdown-preview-heading";
      headingEl.textContent = headingMatch[2].trim();
      fragment.appendChild(headingEl);
      continue;
    }

    if (/^[-*+]\s+/.test(trimmed)) {
      flushParagraph();
      const items = [trimmed.replace(/^[-*+]\s+/, "").trim()];
      while (index + 1 < lines.length && /^[-*+]\s+/.test(lines[index + 1].trim())) {
        index += 1;
        items.push(lines[index].trim().replace(/^[-*+]\s+/, "").trim());
      }
      appendMarkdownList(fragment, items, false);
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      flushParagraph();
      const items = [trimmed.replace(/^\d+\.\s+/, "").trim()];
      while (index + 1 < lines.length && /^\d+\.\s+/.test(lines[index + 1].trim())) {
        index += 1;
        items.push(lines[index].trim().replace(/^\d+\.\s+/, "").trim());
      }
      appendMarkdownList(fragment, items, true);
      continue;
    }

    const imageRefMatch = trimmed.match(/^!\[[^\]]*]\((.+)\)$/);
    if (imageRefMatch) {
      flushParagraph();
      appendMarkdownImageReference(fragment, `Bildreferenz: ${imageRefMatch[1]}`);
      continue;
    }

    paragraphLines.push(line);
  }

  flushParagraph();
  markdownPreviewEl.appendChild(fragment);
  markdownPreviewWrapEl.classList.remove("hidden");
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
    const markdownPreview = typeof data.markdown === "string" ? data.markdown : "";
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
      rawWrapEl.classList.add("hidden");
      clearMarkdownPreview();
      clearResultViewSwitch();
      tablePreviewBodyEl.innerHTML = "";
      tablePreviewWrapEl.classList.add("hidden");
      lastTableMatrices = [];
    } else {
      outputEl.classList.remove("hidden");
      outputEl.textContent = displayText;
      rawWrapEl.classList.remove("hidden");
      renderMarkdownPreview(markdownPreview);
      renderTablePreview(tableMatrices);
      const layoutPages = normalizeLayoutPages(data.layout);
      renderLayoutPanel(layoutPages, data.layout_visualizations);
      renderLayoutOverlay(layoutPages);
      configurePlainResultViews({
        hasLayout:
          layoutPages.length > 0 ||
          (Array.isArray(data.layout_visualizations) && data.layout_visualizations.length > 0),
        hasMarkdown: markdownPreview.trim().length > 0,
      });
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
    const backend = data.backend || String(payload.get("backend") || "direct");
    metaEl.textContent = `Backend: ${backend} | Modell: ${data.model} | Latenz: ${data.latency_ms} ms${warnings ? ` | Hinweise: ${warnings}` : ""}`;
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
previewImageEl.addEventListener("load", () => {
  const layoutPages = normalizeLayoutPages(lastResponse?.layout);
  renderLayoutOverlay(layoutPages);
});
previewImageEl.addEventListener("error", () => {
  const file = currentFile();
  clearLayoutOverlay();
  previewImageStageEl.classList.add("hidden");
  previewImageEl.classList.add("hidden");
  previewImageEl.removeAttribute("src");
  previewEmptyEl.textContent = isTiffLikeFile(file)
    ? "TIFF-Vorschau wird vom Browser nicht zuverlässig unterstützt. OCR läuft trotzdem."
    : `Vorschau für "${file?.type || file?.name || "unbekannt"}" konnte nicht geladen werden.`;
  previewEmptyEl.classList.remove("hidden");
  if (previewUrl) {
    previewPdfLinkEl.href = previewUrl;
    previewPdfLinkEl.textContent = "Datei in neuem Tab öffnen";
    previewPdfLinkEl.classList.remove("hidden");
  }
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
backendEl.addEventListener("change", () => {
  setAdvancedDirty(true);
});
expertEnableLayoutEl.addEventListener("change", () => {
  setAdvancedDirty(true);
});
modelEl.addEventListener("input", () => {
  setAdvancedDirty(true);
});
tokenLimitEl.addEventListener("input", () => {
  setAdvancedDirty(true);
});
gifMaxFramesEl.addEventListener("input", () => {
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

Object.entries(RESULT_VIEW_BUTTONS).forEach(([viewName, buttonEl]) => {
  buttonEl.addEventListener("click", () => {
    if (buttonEl.classList.contains("hidden")) {
      return;
    }
    const availableViews = Object.entries(RESULT_VIEW_BUTTONS)
      .filter(([, candidateButton]) => !candidateButton.classList.contains("hidden"))
      .map(([candidateView]) => candidateView);
    setActiveResultView(viewName, availableViews);
  });
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
