const form = document.getElementById("ocr-form");
const pageEl = document.querySelector(".page");
const fileEl = document.getElementById("file");
const modeEl = document.getElementById("mode");
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
const markdownPreviewWrapEl = document.getElementById("markdown-preview-wrap");
const markdownPreviewEl = document.getElementById("markdown-preview");
const resultViewSwitchEl = document.getElementById("result-view-switch");
const resultViewLayoutBtnEl = document.getElementById("result-view-layout-btn");
const resultViewWordsBtnEl = document.getElementById("result-view-words-btn");
const resultViewMarkdownBtnEl = document.getElementById("result-view-markdown-btn");
const resultViewDiffBtnEl = document.getElementById("result-view-diff-btn");
const resultViewMetricsBtnEl = document.getElementById("result-view-metrics-btn");
const metricsWrapEl = document.getElementById("metrics-wrap");
const metricsEmptyEl = document.getElementById("metrics-empty");
const diffHeadingEl = document.getElementById("diff-heading");
const diffTheirsLabelEl = document.getElementById("diff-theirs-label");
const diffWrapEl = document.getElementById("diff-wrap");
const diffGroupsEl = document.getElementById("diff-groups");
const diffEmptyEl = document.getElementById("diff-empty");
const diffMatchedListEl = document.getElementById("diff-matched-list");
const diffMismatchListEl = document.getElementById("diff-mismatch-list");
const diffOursListEl = document.getElementById("diff-ours-list");
const diffAzureListEl = document.getElementById("diff-azure-list");
const diffMatchedCountEl = document.getElementById("diff-matched-count");
const diffMismatchCountEl = document.getElementById("diff-mismatch-count");
const diffOursCountEl = document.getElementById("diff-ours-count");
const diffAzureCountEl = document.getElementById("diff-azure-count");
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
const previewZoomEl = document.getElementById("preview-zoom");
const previewZoomSliderEl = document.getElementById("preview-zoom-slider");
const previewZoomValueEl = document.getElementById("preview-zoom-value");
const previewZoomFitBtnEl = document.getElementById("preview-zoom-fit");
const previewZoomInBtnEl = document.getElementById("preview-zoom-in");
const previewZoomOutBtnEl = document.getElementById("preview-zoom-out");
const previewPdfEl = document.getElementById("preview-pdf");
const previewPdfLinkEl = document.getElementById("preview-pdf-link");
const layoutWrapEl = document.getElementById("layout-wrap");
const layoutSummaryEl = document.getElementById("layout-summary");
const layoutPagesEl = document.getElementById("layout-pages");
const wordsWrapEl = document.getElementById("words-wrap");
const wordsSummaryEl = document.getElementById("words-summary");
const wordsPagesEl = document.getElementById("words-pages");
const layoutVisualizationsEl = document.getElementById("layout-visualizations");
const compareFormEl = document.getElementById("compare-form");
const azureEndpointEl = document.getElementById("azure-endpoint");
const azureKeyEl = document.getElementById("azure-key");
const comparePresetBtn = document.getElementById("compare-preset-btn");
const comparePresetLayoutBtn = document.getElementById("compare-preset-layout-btn");
const pageSelectorWrapEl = document.getElementById("page-selector-wrap");
const pageSelectorEl = document.getElementById("page-selector");
const compareSummaryEl = document.getElementById("compare-summary");
const compareEngineEl = document.getElementById("compare-engine");
const engineFieldGroups = Array.from(document.querySelectorAll(".engine-fields"));
const ourModelSelectEl = document.getElementById("our-model");
const theirModelSelectEl = document.getElementById("their-model");
const peerBaseUrlEl = document.getElementById("peer-base-url");
const peerModelSelectEl = document.getElementById("peer-model-select");
const peerModelInputEl = document.getElementById("peer-model-input");
const peerModelHintEl = document.getElementById("peer-model-hint");
const referenceTextEl = document.getElementById("reference-text");
const referenceFileEl = document.getElementById("reference-file");
const referenceClearBtnEl = document.getElementById("reference-clear-btn");
const referenceIgnoreEmbeddedEl = document.getElementById("reference-ignore-embedded");
const referenceIgnoreWrapEl = document.getElementById("reference-ignore-wrap");
const referenceSourceBadgeEl = document.getElementById("reference-source-badge");
const compareReferenceEl = document.getElementById("compare-reference");
const metricsPanelEl = document.getElementById("metrics-panel");
const metricsContentEl = document.getElementById("metrics-content");
const metricsTabBtns = Array.from(document.querySelectorAll("[data-metrics-tab]"));
const tr = (key, params) =>
  typeof window.docreadT === "function" ? window.docreadT(key, params) : key;

function translateWarning(message) {
  if (!message || typeof message !== "string") return message;
  const deLayout = message.match(
    /^Document-Layout: (\d+) Regionen auf (\d+) Seite\(n\) erkannt\.$/
  );
  if (deLayout) {
    return tr("warning_document_layout", { regions: deLayout[1], pages: deLayout[2] });
  }
  const enLayout = message.match(
    /^Document layout: (\d+) regions detected on (\d+) page\(s\)\.$/i
  );
  if (enLayout) {
    return tr("warning_document_layout", { regions: enLayout[1], pages: enLayout[2] });
  }
  return message;
}
const appBasePath = (document.body?.dataset.basePath || "").replace(/\/$/, "");
const ocrEndpoint = `${appBasePath}/api/ocr`;
const compareEndpoint = `${appBasePath}/api/compare`;
const extractPdfTextEndpoint = `${appBasePath}/api/extract-pdf-text`;
const referenceMetricsEndpoint = `${appBasePath}/api/metrics/reference`;

let lastResponse = null;
let lastCompareResponse = null;
let lastOcrApplyContext = null;
let lastDiff = null;
let lastMetrics = null;
let activeMetricsTab = "intrinsic";
let embeddedReferenceText = "";  // text extracted from PDF text-layer, if any
let manualReferenceText = "";    // text the user pasted/uploaded; wins when present
let lastTableMatrices = [];
let previewUrl = null;
let activeRequestController = null;
let globalDragDepth = 0;
let hasPendingAdvancedChanges = false;
let activeResultView = "layout";
let pageImageDataUrls = [];
let currentPageIndex = 0;
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
const WORD_DOCUMENT_TYPES = new Set([
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]);
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
  words: resultViewWordsBtnEl,
  markdown: resultViewMarkdownBtnEl,
  diff: resultViewDiffBtnEl,
  metrics: resultViewMetricsBtnEl,
};
const RESULT_VIEW_WRAPS = {
  layout: layoutWrapEl,
  words: wordsWrapEl,
  markdown: markdownPreviewWrapEl,
  diff: diffWrapEl,
  metrics: metricsWrapEl,
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
  advancedToggleEl.textContent = isOpen ? tr("advanced_toggle_hide") : tr("advanced_toggle");
}

function setLoading(isLoading) {
  loadingOverlayEl.classList.toggle("is-active", isLoading);
  document.body.classList.toggle("is-loading", isLoading);
  const submitBtn = form.querySelector('button[type="submit"]');
  if (submitBtn) {
    submitBtn.disabled = isLoading;
  }
  if (!isLoading) {
    applyOptionsBtnEl.disabled = !hasPendingAdvancedChanges;
  }
}

function setAdvancedDirty(isDirty) {
  hasPendingAdvancedChanges = isDirty;
  applyOptionsBtnEl.disabled = !isDirty;
  advancedDirtyHintEl.classList.toggle("hidden", !isDirty);
}

function clearOutput() {
  outputEl.textContent = "";
  lastDiff = null;
  lastMetrics = null;
  renderMetricsPanel(null);
  clearMarkdownPreview();
  clearDiffPanel();
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

function clearPreview(message = null) {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  pageImageDataUrls = [];
  currentPageIndex = 0;
  pageSelectorWrapEl.classList.add("hidden");
  if (message == null) message = tr("preview_empty");
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
    previewEmptyEl.textContent = tr("preview_pdf_pending");
    previewEmptyEl.classList.remove("hidden");
    return;
  }

  if (isTiffLikeFile(file)) {
    previewEmptyEl.textContent = tr("preview_tiff_pending");
    previewEmptyEl.classList.remove("hidden");
    return;
  }

  if (WORD_DOCUMENT_TYPES.has(file.type) || fileExtension(file.name) === ".doc" || fileExtension(file.name) === ".docx") {
    previewEmptyEl.textContent = tr("preview_word_pending");
    previewEmptyEl.classList.remove("hidden");
    return;
  }

  clearPreview(`Für "${file.type || "unbekannt"}" ist keine Vorschau verfügbar.`);
}

function currentFile() {
  return fileEl.files && fileEl.files[0] ? fileEl.files[0] : null;
}

function populatePageSelector(pageCount) {
  pageSelectorEl.innerHTML = "";
  for (let i = 0; i < pageCount; i++) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = tr("page_n", { n: i + 1 });
    pageSelectorEl.appendChild(opt);
  }
  pageSelectorWrapEl.classList.toggle("hidden", pageCount <= 1);
  currentPageIndex = 0;
  pageSelectorEl.value = "0";
}

function showPageImage(index) {
  if (index < 0 || index >= pageImageDataUrls.length) return;
  currentPageIndex = index;
  previewImageEl.src = pageImageDataUrls[index];
  previewImageStageEl.classList.remove("hidden");
  previewImageEl.classList.remove("hidden");
  previewPdfEl.classList.add("hidden");
  previewEmptyEl.classList.add("hidden");
  const layoutPages = normalizeLayoutPages(lastResponse?.layout);
  renderLayoutOverlay(layoutPages, index);
  renderLayoutPanel(layoutPages, lastResponse?.layout_visualizations, index);
  if (activeResultView === "words") {
    applyWordMode(true, layoutPages);
  }
  if (activeResultView === "diff" && lastDiff) {
    const pageDiff = getActivePageDiff();
    renderDiffOverlay(pageDiff);
    renderDiffPanel(pageDiff);
  }
}

function clearLayoutOverlay() {
  previewLayoutOverlayEl.innerHTML = "";
  previewLayoutOverlayEl.classList.add("hidden");
}

function clearLayoutSidebar() {
  layoutSummaryEl.textContent = "";
  layoutPagesEl.innerHTML = "";
  layoutVisualizationsEl.innerHTML = "";
  layoutVisualizationsEl.classList.add("hidden");
  layoutWrapEl.classList.add("hidden");
  if (wordsSummaryEl) wordsSummaryEl.textContent = "";
  if (wordsPagesEl) wordsPagesEl.innerHTML = "";
  if (wordsWrapEl) wordsWrapEl.classList.add("hidden");
}

function clearLayoutDisplay() {
  clearLayoutOverlay();
  clearLayoutSidebar();
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
  syncOverlayToActiveView();
}

function syncOverlayToActiveView() {
  const layoutPages = normalizeLayoutPages(lastResponse?.layout);
  if (activeResultView === "diff" && lastDiff) {
    renderDiffOverlay(getActivePageDiff());
    renderDiffPanel(getActivePageDiff());
  } else if (activeResultView === "words") {
    renderDiffOverlay(null);
    applyWordMode(true, layoutPages);
  } else {
    renderDiffOverlay(null);
    applyWordMode(false, layoutPages);
    if (layoutPages.length > 0) {
      renderLayoutOverlay(layoutPages, currentPageIndex);
    }
  }
}

function getActivePageDiff() {
  if (!lastDiff || !Array.isArray(lastDiff.pages)) return null;
  return lastDiff.pages[currentPageIndex] || lastDiff.pages[0] || null;
}

function configurePlainResultViews({ hasLayout, hasMarkdown, hasMetrics = false }) {
  const availableViews = [];
  if (hasLayout) {
    availableViews.push("layout");
    availableViews.push("words");
  }
  if (hasMarkdown) {
    availableViews.push("markdown");
  }
  availableViews.push("diff");
  if (hasMetrics) {
    availableViews.push("metrics");
  }
  const defaultView = hasLayout ? "layout" : hasMarkdown ? "markdown" : "diff";
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

function renderLayoutOverlay(layoutPages, pageIndex) {
  clearLayoutOverlay();
  const file = currentFile();
  if (!file) return;
  if (!file.type.startsWith("image/") && pageImageDataUrls.length === 0) return;

  const selectedIndex = pageIndex != null ? pageIndex : currentPageIndex;
  const activePage = layoutPages[selectedIndex] || layoutPages[0];
  if (!activePage || !Array.isArray(activePage.regions)) {
    return;
  }

  const svgNs = "http://www.w3.org/2000/svg";
  const svgEl = document.createElementNS(svgNs, "svg");
  svgEl.setAttribute("viewBox", "0 0 100 100");
  svgEl.setAttribute("preserveAspectRatio", "none");
  svgEl.classList.add("preview-layout-svg");
  previewLayoutOverlayEl.appendChild(svgEl);

  let overlayCount = 0;
  activePage.regions.forEach((region, index) => {
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
    shapeEl.dataset.regionIndex = String(index);
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
    boxEl.dataset.regionIndex = String(index);
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

// Assign each word to the first region whose bbox_2d contains the word's polygon center.
// Returns annotated words with regionLabel and regionKind added.
function assignWordsToRegions(words, regions) {
  return words.map((word) => {
    const poly = word.polygon;
    if (!Array.isArray(poly) || poly.length < 8) {
      return { ...word, regionLabel: "Unbekannt", regionKind: "other" };
    }
    const scale = getLayoutCoordinateScale(poly);
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    for (let i = 0; i < poly.length; i += 2) {
      const px = poly[i] / scale;
      const py = poly[i + 1] / scale;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }
    const rawWordArea = (maxX - minX) * (maxY - minY);
    const wordArea = rawWordArea > 0 ? rawWordArea : 0.000001;

    let bestRegion = null;
    let bestArea = Infinity;

    for (const region of regions) {
      if (!Array.isArray(region.bbox_2d) || region.bbox_2d.length !== 4) continue;
      const vals = region.bbox_2d.map(Number);
      const rs = getLayoutCoordinateScale(vals);
      const [x1, y1, x2, y2] = vals.map((v) => v / rs);
      
      const ix1 = Math.max(minX, x1);
      const iy1 = Math.max(minY, y1);
      const ix2 = Math.min(maxX, x2);
      const iy2 = Math.min(maxY, y2);
      
      if (ix1 < ix2 && iy1 < iy2) {
        const intersection = (ix2 - ix1) * (iy2 - iy1);
        const ioa = intersection / wordArea;
        if (ioa > 0.1) {
          const area = (x2 - x1) * (y2 - y1);
          if (area < bestArea) {
            bestArea = area;
            bestRegion = region;
          }
        }
      }
    }
    
    if (bestRegion) {
      const label = bestRegion.label || "Region";
      return { ...word, regionLabel: label, regionKind: normalizeLayoutRegionKind(label) };
    }
    return { ...word, regionLabel: "Unbekannt", regionKind: "other" };
  });
}

function renderWordOverlay(annotatedWords) {
  const existing = previewLayoutOverlayEl.querySelector(".preview-word-svg");
  if (existing) existing.remove();
  if (!annotatedWords || annotatedWords.length === 0) return;

  const svgNs = "http://www.w3.org/2000/svg";
  const svgEl = document.createElementNS(svgNs, "svg");
  svgEl.setAttribute("viewBox", "0 0 100 100");
  svgEl.setAttribute("preserveAspectRatio", "none");
  svgEl.classList.add("preview-layout-svg", "preview-word-svg");
  previewLayoutOverlayEl.appendChild(svgEl);

  annotatedWords.forEach((word, idx) => {
    const points = normalizedPolygonToPercentages(word.polygon);
    if (!points) return;
    const shapeEl = document.createElementNS(svgNs, "polygon");
    shapeEl.classList.add("preview-word-shape");
    if (word.regionKind) shapeEl.dataset.regionKind = word.regionKind;
    shapeEl.dataset.wordIndex = String(idx);
    shapeEl.setAttribute("points", points.map((p) => `${p.x},${p.y}`).join(" "));
    shapeEl.title = word.content ? `${word.regionLabel}: ${word.content}` : (word.regionLabel || "");
    svgEl.appendChild(shapeEl);
  });
}

function renderWordSidebar(annotatedWords, regions) {
  if (!wordsPagesEl) return;
  wordsPagesEl.innerHTML = "";
  if (wordsSummaryEl) {
    wordsSummaryEl.textContent = tr("words_summary", { count: annotatedWords.length, regions: regions.length });
  }

  // Build ordered groups from regions, then a catch-all
  const groupOrder = regions.map((r) => r.label || "Region");
  const seen = new Set();
  const groups = [];
  for (const label of groupOrder) {
    if (seen.has(label)) continue;
    seen.add(label);
    groups.push({ label, regionKind: normalizeLayoutRegionKind(label), words: [] });
  }
  const unknownGroup = { label: "Unbekannt", regionKind: "other", words: [] };

  annotatedWords.forEach((word, idx) => {
    const entry = { word, idx };
    const group = groups.find((g) => g.label === word.regionLabel);
    if (group) {
      group.words.push(entry);
    } else {
      unknownGroup.words.push(entry);
    }
  });
  if (unknownGroup.words.length > 0) groups.push(unknownGroup);

  for (const group of groups) {
    if (group.words.length === 0) continue;
    const sectionEl = document.createElement("div");
    sectionEl.className = "word-region-group";
    sectionEl.dataset.regionKind = group.regionKind;

    const headEl = document.createElement("strong");
    headEl.className = "layout-region-label";
    headEl.dataset.regionKind = group.regionKind;
    headEl.textContent = `${group.label} (${group.words.length})`;
    sectionEl.appendChild(headEl);

    const listEl = document.createElement("ul");
    listEl.className = "word-list";
    group.words.forEach(({ word, idx }) => {
      const li = document.createElement("li");
      li.className = "word-item";
      li.textContent = word.content || "";
      li.dataset.wordIndex = String(idx);
      const hasBox = Array.isArray(word.polygon) && word.polygon.length >= 8;
      if (!hasBox) li.dataset.hasBox = "false";
      li.addEventListener("mouseenter", () => _highlightWord(idx, true));
      li.addEventListener("mouseleave", () => _highlightWord(idx, false));
      listEl.appendChild(li);
    });
    sectionEl.appendChild(listEl);
    wordsPagesEl.appendChild(sectionEl);
  }
}

function _highlightWord(wordIndex, on) {
  if (wordIndex == null) return;
  previewLayoutOverlayEl
    .querySelectorAll(`.preview-word-svg [data-word-index="${wordIndex}"]`)
    .forEach((el) => el.classList.toggle("is-highlighted", !!on));
  previewLayoutOverlayEl.classList.toggle(
    "has-hover",
    !!previewLayoutOverlayEl.querySelector(".is-highlighted"),
  );
}

function applyWordMode(active, layoutPages) {
  const layoutSvgEls = previewLayoutOverlayEl.querySelectorAll(
    ".preview-layout-svg:not(.preview-word-svg)"
  );
  const layoutBoxEls = previewLayoutOverlayEl.querySelectorAll(".preview-layout-box");

  if (active) {
    layoutSvgEls.forEach((el) => el.classList.add("hidden"));
    layoutBoxEls.forEach((el) => el.classList.add("hidden"));

    const activePage = layoutPages?.[currentPageIndex] || layoutPages?.[0];
    const regions = activePage?.regions || [];
    const rawWordPolys = activePage?.word_polys;
    let annotatedWords;
    if (rawWordPolys && rawWordPolys.length > 0) {
      const detectorWords = rawWordPolys.map((wp) => ({ polygon: wp.polygon, content: wp.content || "" }));
      annotatedWords = assignWordsToRegions(detectorWords, regions);
    } else {
      const pages = lastResponse?.analyzeResult?.pages || [];
      const analyzePageWords = pages?.[currentPageIndex]?.words || pages?.[0]?.words || [];
      annotatedWords = assignWordsToRegions(analyzePageWords, regions);
    }
    renderWordOverlay(annotatedWords);
    renderWordSidebar(annotatedWords, regions);
  } else {
    // Wort-Overlay verlassen: Layout-Polygone wieder einblenden, Word-SVG entfernen.
    // Sichtbarkeit der Sidebar-Section gehört der Tab-Logik (setActiveResultView) —
    // hier NICHT renderLayoutPanel aufrufen, sonst hängt der Layout-Block auf
    // Markdown/Vergleich-Tabs.
    layoutSvgEls.forEach((el) => el.classList.remove("hidden"));
    layoutBoxEls.forEach((el) => el.classList.remove("hidden"));

    const existing = previewLayoutOverlayEl.querySelector(".preview-word-svg");
    if (existing) existing.remove();

    const lp = layoutPages || normalizeLayoutPages(lastResponse?.layout);
    if (layoutBoxEls.length === 0 && lp.length > 0) {
      renderLayoutOverlay(lp, currentPageIndex);
    } else {
      previewLayoutOverlayEl.classList.remove("hidden");
    }
  }
}

function renderDiffOverlay(diff) {
  const existingDiff = previewLayoutOverlayEl.querySelector(".preview-diff-svg");
  if (existingDiff) existingDiff.remove();

  if (!diff) {
    previewLayoutOverlayEl.querySelectorAll(".preview-layout-svg, .preview-layout-box").forEach(
      (el) => el.classList.remove("hidden")
    );
    return;
  }

  previewLayoutOverlayEl.querySelectorAll(".preview-layout-svg:not(.preview-diff-svg), .preview-layout-box").forEach(
    (el) => el.classList.add("hidden")
  );

  const matched = diff.matched || [];
  const mismatched = diff.mismatched || [];
  const onlyOurs = diff.only_ours || [];
  const onlyAzure = diff.only_azure || [];

  const svgNs = "http://www.w3.org/2000/svg";
  const svgEl = document.createElementNS(svgNs, "svg");
  svgEl.setAttribute("viewBox", "0 0 100 100");
  svgEl.setAttribute("preserveAspectRatio", "none");
  svgEl.classList.add("preview-layout-svg", "preview-diff-svg");
  previewLayoutOverlayEl.appendChild(svgEl);
  previewLayoutOverlayEl.classList.remove("hidden");

  function addShape(word, cssClass, title, pairKey) {
    if (!word) return;
    const points = normalizedPolygonToPercentages(word.polygon);
    if (!points) return;
    const el = document.createElementNS(svgNs, "polygon");
    cssClass.split(/\s+/).filter(Boolean).forEach((cls) => el.classList.add(cls));
    el.setAttribute("points", points.map((p) => `${p.x},${p.y}`).join(" "));
    if (pairKey) el.setAttribute("data-pair-key", pairKey);
    const titleEl = document.createElementNS(svgNs, "title");
    titleEl.textContent = title;
    el.appendChild(titleEl);
    svgEl.appendChild(el);
  }

  matched.forEach((pair, idx) => {
    const key = `m-${idx}`;
    const text = pair.ours?.content || pair.azure?.content || "";
    addShape(pair.ours, "preview-diff-shape-matched", tr("diff_match_tooltip", { text }), key);
    addShape(pair.azure, "preview-diff-shape-matched preview-diff-shape-matched-azure", tr("diff_match_azure_tooltip", { engine: "Azure", text }), key);
  });
  mismatched.forEach((pair, idx) => {
    const key = `x-${idx}`;
    const ours = pair.ours?.content || "";
    const azure = pair.azure?.content || "";
    const title = `Abweichung — Unser: "${ours}" | Azure: "${azure}"`;
    addShape(pair.ours, "preview-diff-shape-mismatch preview-diff-shape-mismatch-ours", title, key);
    addShape(pair.azure, "preview-diff-shape-mismatch preview-diff-shape-mismatch-azure", title, key);
  });
  onlyOurs.forEach((w, idx) => {
    addShape(w, "preview-diff-shape-ours", tr("diff_tooltip_ours_only", { text: w.content || "" }), `o-${idx}`);
  });
  onlyAzure.forEach((w, idx) => {
    addShape(w, "preview-diff-shape-azure", tr("diff_tooltip_theirs_only", { name: "Azure", text: w.content || "" }), `a-${idx}`);
  });
}

function clearDiffPanel() {
  if (!diffGroupsEl) return;
  diffGroupsEl.classList.add("hidden");
  if (diffEmptyEl) diffEmptyEl.classList.remove("hidden");
  [diffMatchedListEl, diffMismatchListEl, diffOursListEl, diffAzureListEl].forEach((el) => {
    if (el) el.innerHTML = "";
  });
  [diffMatchedCountEl, diffMismatchCountEl, diffOursCountEl, diffAzureCountEl].forEach((el) => {
    if (el) el.textContent = "0";
  });
}

function renderDiffPanel(diff) {
  if (!diffGroupsEl) return;
  if (!diff) {
    clearDiffPanel();
    return;
  }
  const matched = diff.matched || [];
  const mismatched = diff.mismatched || [];
  const onlyOurs = diff.only_ours || [];
  const onlyAzure = diff.only_azure || [];

  diffEmptyEl?.classList.add("hidden");
  diffGroupsEl.classList.remove("hidden");

  diffMatchedCountEl.textContent = String(matched.length);
  diffMismatchCountEl.textContent = String(mismatched.length);
  diffOursCountEl.textContent = String(onlyOurs.length);
  diffAzureCountEl.textContent = String(onlyAzure.length);

  diffMatchedListEl.innerHTML = "";
  matched.forEach((pair, idx) => {
    const li = document.createElement("li");
    li.className = "diff-row diff-row-matched";
    li.dataset.pairKey = `m-${idx}`;
    li.textContent = pair.ours?.content || pair.azure?.content || "";
    diffMatchedListEl.appendChild(li);
  });

  diffMismatchListEl.innerHTML = "";
  mismatched.forEach((pair, idx) => {
    const li = document.createElement("li");
    li.className = "diff-row diff-row-mismatch";
    li.dataset.pairKey = `x-${idx}`;
    const ours = document.createElement("span");
    ours.className = "diff-token diff-token-ours";
    ours.textContent = pair.ours?.content || "";
    const sep = document.createElement("span");
    sep.className = "diff-sep";
    sep.textContent = "↔";
    const azure = document.createElement("span");
    azure.className = "diff-token diff-token-azure";
    azure.textContent = pair.azure?.content || "";
    li.append(ours, sep, azure);
    diffMismatchListEl.appendChild(li);
  });

  const hasBox = (w) => Array.isArray(w?.polygon) && w.polygon.length >= 8;
  diffOursListEl.innerHTML = "";
  onlyOurs.forEach((w, idx) => {
    const li = document.createElement("li");
    li.className = "diff-row diff-row-ours";
    li.dataset.pairKey = `o-${idx}`;
    li.textContent = w.content || "";
    if (!hasBox(w)) {
      li.dataset.hasBox = "false";
      li.title = tr("diff_token_no_coords");
    }
    diffOursListEl.appendChild(li);
  });

  diffAzureListEl.innerHTML = "";
  onlyAzure.forEach((w, idx) => {
    const li = document.createElement("li");
    li.className = "diff-row diff-row-azure";
    li.dataset.pairKey = `a-${idx}`;
    li.textContent = w.content || "";
    if (!hasBox(w)) {
      li.dataset.hasBox = "false";
      li.title = "Keine Koordinaten verfügbar.";
    }
    diffAzureListEl.appendChild(li);
  });

  diffGroupsEl.querySelectorAll("li.diff-row").forEach((row) => {
    row.addEventListener("mouseenter", () => highlightDiffPair(row.dataset.pairKey, true));
    row.addEventListener("mouseleave", () => highlightDiffPair(row.dataset.pairKey, false));
  });
}

function highlightDiffPair(pairKey, on) {
  if (!pairKey) return;
  previewLayoutOverlayEl
    .querySelectorAll(`[data-pair-key="${pairKey}"]`)
    .forEach((el) => el.classList.toggle("is-highlighted", !!on));
  // Toggle a parent class so other boxes can be visually muted via CSS.
  previewLayoutOverlayEl.classList.toggle(
    "has-hover",
    !!previewLayoutOverlayEl.querySelector(".is-highlighted"),
  );
}

function _highlightRegion(regionIndex, on) {
  if (regionIndex == null) return;
  previewLayoutOverlayEl
    .querySelectorAll(`[data-region-index="${regionIndex}"]`)
    .forEach((el) => el.classList.toggle("is-highlighted", !!on));
  previewLayoutOverlayEl.classList.toggle(
    "has-hover",
    !!previewLayoutOverlayEl.querySelector(".is-highlighted"),
  );
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
    imgEl.alt = tr("layout_viz_alt", { index: index + 1 });
    imgEl.loading = "lazy";
    layoutVisualizationsEl.appendChild(imgEl);
  });
  layoutVisualizationsEl.classList.remove("hidden");
}

function renderLayoutPanel(layoutPages, visualizations, activePageIndex = null) {
  clearLayoutSidebar();
  if (layoutPages.length === 0 && (!Array.isArray(visualizations) || visualizations.length === 0)) {
    return;
  }

  const activeIndex = activePageIndex != null
    ? Math.max(0, Math.min(activePageIndex, layoutPages.length - 1))
    : currentPageIndex;
  const visiblePages = layoutPages.length > 0
    ? [{ page: layoutPages[activeIndex] || layoutPages[0], pageIndex: activeIndex }]
    : [];

  const regionCount = visiblePages.reduce(
    (total, entry) => total + (Array.isArray(entry.page?.regions) ? entry.page.regions.length : 0),
    0
  );
  const confidenceStats = collectLayoutConfidenceStats(
    visiblePages.map((entry) => entry.page)
  );
  const summaryParts = [];
  if (layoutPages.length > 0) {
    summaryParts.push(
      layoutPages.length > 1
        ? tr("layout_summary_page_multi", {
            current: activeIndex + 1,
            total: layoutPages.length,
          })
        : tr("layout_summary_pages_one", { count: layoutPages.length })
    );
    summaryParts.push(tr("layout_summary_regions", { count: regionCount }));
  }
  if (confidenceStats) {
    summaryParts.push(
      tr("layout_summary_confidence", {
        value: formatConfidence(confidenceStats.average),
      })
    );
    summaryParts.push(
      tr("layout_summary_scores", { count: confidenceStats.count })
    );
  }
  if (Array.isArray(visualizations) && visualizations.length > 0) {
    summaryParts.push(
      tr("layout_summary_visualizations", { count: visualizations.length })
    );
  }
  layoutSummaryEl.textContent = summaryParts.join(" | ");

  visiblePages.forEach(({ page, pageIndex }) => {
    const pageCardEl = document.createElement("article");
    pageCardEl.className = "layout-page-card";

    const pageTitleEl = document.createElement("p");
    pageTitleEl.className = "layout-page-title";
    pageTitleEl.textContent = tr("page_n", { n: page.page_number || pageIndex + 1 });
    pageCardEl.appendChild(pageTitleEl);

    const regions = Array.isArray(page.regions) ? page.regions : [];
    if (regions.length === 0) {
      const emptyEl = document.createElement("p");
      emptyEl.className = "layout-page-empty";
      emptyEl.textContent = tr("layout_no_regions");
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
      regionItemEl.dataset.regionIndex = String(regionIndex);
      regionItemEl.addEventListener("mouseenter", () =>
        _highlightRegion(regionIndex, true),
      );
      regionItemEl.addEventListener("mouseleave", () =>
        _highlightRegion(regionIndex, false),
      );

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

      if (Array.isArray(region.cells) && region.cells.length > 0) {
        const cellsWrap = document.createElement("details");
        cellsWrap.className = "layout-cells-wrap";
        const cellsSummary = document.createElement("summary");
        const headerCount = region.cells.filter((c) => c.is_header).length;
        const maxRow = Math.max(...region.cells.map((c) => c.row));
        const maxCol = Math.max(...region.cells.map((c) => c.column));
        const rows = maxRow + 1;
        const cols = maxCol + 1;
        let summaryText = `Tabellenstruktur: ${rows} Zeilen × ${cols} Spalten`;
        if (headerCount) summaryText += ` (${headerCount} Header-Zellen)`;
        cellsSummary.textContent = summaryText;
        cellsWrap.appendChild(cellsSummary);

        const grid = document.createElement("table");
        grid.className = "layout-cells-grid";
        const cellMap = new Map();
        for (const cell of region.cells) {
          cellMap.set(`${cell.row},${cell.column}`, cell);
        }
        for (let r = 0; r <= maxRow; r++) {
          const tr = document.createElement("tr");
          for (let c = 0; c <= maxCol; c++) {
            const cell = cellMap.get(`${r},${c}`);
            const td = document.createElement(cell && cell.is_header ? "th" : "td");
            if (cell) {
              const cb = Array.isArray(cell.bbox_2d) ? cell.bbox_2d : [];
              const bboxStr = cb.map((v) => Math.round(v)).join(", ");
              td.title = `Zeile ${r + 1}, Spalte ${c + 1} | bbox: ${bboxStr}`;
              td.textContent = cell.content || "";
              if (cell.row_span > 1 || cell.col_span > 1) {
                td.classList.add("layout-cell-span");
              }
            }
            tr.appendChild(td);
          }
          grid.appendChild(tr);
        }
        cellsWrap.appendChild(grid);
        regionItemEl.appendChild(cellsWrap);
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
      throw new Error(tr("token_limit_invalid"));
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
      throw new Error(tr("gif_frames_invalid"));
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
  const expertLayoutModelValue = String(payload.get("expert_layout_model") || "").trim();
  if (expertLayoutModelValue) {
    payload.set("expert_layout_model", expertLayoutModelValue);
  } else {
    payload.delete("expert_layout_model");
  }
  const expertLayoutThresholdValue = String(payload.get("expert_layout_threshold") || "").trim();
  if (expertLayoutThresholdValue) {
    payload.set("expert_layout_threshold", expertLayoutThresholdValue);
  } else {
    payload.delete("expert_layout_threshold");
  }
  const expertTextAnchorValue = String(payload.get("expert_text_anchor") || "").trim();
  if (expertTextAnchorValue === "true" || expertTextAnchorValue === "false") {
    payload.set("expert_text_anchor", expertTextAnchorValue);
  } else {
    payload.delete("expert_text_anchor");
  }
  const expertTextAnchorThresholdValue = String(
    payload.get("expert_text_anchor_threshold") || "",
  ).trim();
  if (expertTextAnchorThresholdValue) {
    payload.set("expert_text_anchor_threshold", expertTextAnchorThresholdValue);
  } else {
    payload.delete("expert_text_anchor_threshold");
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

function applyOcrResponse(data, { requestMode, requestTask, backendFallback }) {
  lastResponse = data;
  lastOcrApplyContext = { requestMode, requestTask, backendFallback };
  setAdvancedDirty(false);
  let displayText = data.text || tr("no_content");
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
    clearMarkdownPreview();
    clearDiffPanel();
    clearResultViewSwitch();
    tablePreviewBodyEl.innerHTML = "";
    tablePreviewWrapEl.classList.add("hidden");
    lastTableMatrices = [];
  } else {
    outputEl.textContent = displayText;
    // New OCR run → previous compare results no longer apply.
    lastDiff = null;
    lastMetrics = null;
    renderMetricsPanel(null);
    if (diffHeadingEl) diffHeadingEl.textContent = tr("diff_heading");
    if (diffTheirsLabelEl) diffTheirsLabelEl.textContent = tr("diff_theirs_default");
    if (diffGroupsEl) diffGroupsEl.classList.add("hidden");
    diffEmptyEl?.classList.remove("hidden");
    clearDiffPanel();
    renderMarkdownPreview(markdownPreview);
    renderTablePreview(tableMatrices);
    const layoutPages = normalizeLayoutPages(data.layout);

    pageImageDataUrls = data.page_images || [];
    currentPageIndex = 0;
    if (pageImageDataUrls.length > 0) {
      populatePageSelector(pageImageDataUrls.length);
      showPageImage(0);
    } else {
      pageSelectorWrapEl.classList.add("hidden");
      renderLayoutOverlay(layoutPages, 0);
      renderLayoutPanel(layoutPages, data.layout_visualizations, 0);
    }

    renderDiffOverlay(null);
    compareSummaryEl.classList.add("hidden");
    configurePlainResultViews({
      hasLayout:
        layoutPages.length > 0 ||
        (Array.isArray(data.layout_visualizations) && data.layout_visualizations.length > 0),
      hasMarkdown: markdownPreview.trim().length > 0,
      hasMetrics: !!lastMetrics,
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

  const warnings = (data.warnings || [])
    .map((w) => translateWarning(String(w)))
    .join(" | ");
  const backend = data.backend || backendFallback || "direct";
  metaEl.textContent =
    tr("meta_backend", { backend, model: data.model, latency: data.latency_ms }) +
    (warnings ? tr("meta_warnings", { warnings }) : "");

  void maybeFetchReferenceMetrics(data?.text || "");
}

async function maybeFetchReferenceMetrics(hypothesisText) {
  const refText = effectiveReferenceText();
  if (!hypothesisText || !refText || !refText.trim()) {
    // Referenz wurde geleert → Referenz-Block aus den Metriken entfernen.
    if (lastMetrics?.reference) {
      const { reference: _drop, ...rest } = lastMetrics;
      const hasOther = !!(rest.intrinsic || rest.comparison);
      lastMetrics = hasOther ? { ...rest, reference: null } : null;
      renderMetricsPanel(lastMetrics);
    }
    return;
  }
  try {
    const fd = new FormData();
    fd.append("text", hypothesisText);
    fd.append("reference_text", refText);
    const resp = await fetch(referenceMetricsEndpoint, { method: "POST", body: fd });
    if (!resp.ok) return;
    const data = await resp.json();
    const reference = data?.reference;
    if (!reference) return;
    // Wenn ein Compare-Lauf bereits Metriken gesetzt hat, ergänzen wir nur
    // den Referenz-Block; andernfalls bauen wir ein metrics-only-Objekt.
    lastMetrics = lastMetrics
      ? { ...lastMetrics, reference }
      : { intrinsic: null, comparison: null, reference };
    renderMetricsPanel(lastMetrics);
  } catch {
    /* netzwerkfehler stillschweigend ignorieren */
  }
}

function applyCompareResponse(data) {
  lastCompareResponse = data;
  lastDiff = data?.diff || null;
  lastMetrics = data?.metrics || null;
  const pageCount = Array.isArray(lastDiff?.pages) ? lastDiff.pages.length : 0;
  const matchedTotal = lastDiff?.matched_count ?? 0;
  const mismatchedTotal = lastDiff?.mismatched_count ?? 0;
  const onlyOursTotal = lastDiff?.only_ours_count ?? 0;
  const onlyTheirsTotal = lastDiff?.only_theirs_count ?? lastDiff?.only_azure_count ?? 0;

  const pageLabel =
    pageCount > 1 ? ` ${tr("compare_all_pages", { count: pageCount })}` : "";
  const theirsLabel = data?.engine?.label || "Andere";
  if (diffHeadingEl) diffHeadingEl.textContent = tr("diff_heading_engine", { engine: theirsLabel });
  if (diffTheirsLabelEl) diffTheirsLabelEl.textContent = tr("diff_theirs_named", { name: theirsLabel });
  const ourWarnings = Array.isArray(data?.our_warnings) ? data.our_warnings : [];
  const theirWarnings = Array.isArray(data?.their_warnings) ? data.their_warnings : [];
  const warningsHtml = [
    ...ourWarnings.map((w) => `<div class="compare-warning">⚠ Wir: ${escapeHtml(w)}</div>`),
    ...theirWarnings.map(
      (w) => `<div class="compare-warning">⚠ ${escapeHtml(theirsLabel)}: ${escapeHtml(w)}</div>`,
    ),
  ].join("");
  compareSummaryEl.innerHTML =
    `<span class="diff-legend-item diff-legend-matched">${tr("diff_summary_matched", { n: matchedTotal })}</span> | ` +
    `<span class="diff-legend-item diff-legend-mismatch">${tr("diff_summary_mismatch", { n: mismatchedTotal })}</span> | ` +
    `<span class="diff-legend-item diff-legend-ours">${tr("diff_summary_ours", { n: onlyOursTotal })}</span> | ` +
    `<span class="diff-legend-item diff-legend-azure">${tr("diff_summary_theirs", { name: escapeHtml(theirsLabel), n: onlyTheirsTotal })}</span>` +
    `<span class="compare-total-hint">${pageLabel}</span>` +
    warningsHtml;
  compareSummaryEl.classList.remove("hidden");

  renderMetricsPanel(lastMetrics);

  // Show the diff list groups now that we have data, and re-render the page
  // diff so the count badges + per-group lists pick up the fresh values
  // immediately (otherwise they stay at the previous run's stale "0" until
  // a tab switch triggers syncOverlayToActiveView).
  diffEmptyEl?.classList.add("hidden");
  if (diffGroupsEl) diffGroupsEl.classList.remove("hidden");
  const pageDiff = getActivePageDiff();
  renderDiffPanel(pageDiff);
  renderDiffOverlay(pageDiff);
}

function _fmtNumber(value, { digits = 3, percent = false } = {}) {
  if (value === null || value === undefined) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  if (percent) return `${(value * 100).toFixed(digits === 3 ? 1 : digits)} %`;
  return value.toFixed(digits);
}

function _metricRow(label, valueOurs, valueTheirs, { tooltip } = {}) {
  const helpHtml = tooltip
    ? ` <span class="metric-help" title="${escapeHtml(tooltip)}">ⓘ</span>`
    : "";
  const cell = (value) => {
    if (value === null || value === undefined) {
      return `<td class="metric-empty">—</td>`;
    }
    return `<td class="metric-value">${escapeHtml(String(value))}</td>`;
  };
  return `<tr><th>${escapeHtml(label)}${helpHtml}</th>${cell(valueOurs)}${cell(valueTheirs)}</tr>`;
}

function _renderIntrinsicTab(intrinsic) {
  if (!intrinsic) return `<p class="compare-reference-hint">${tr("metrics_no_data")}</p>`;
  const ours = intrinsic.ours || {};
  const theirs = intrinsic.theirs || {};
  const fmtConf = (v) => _fmtNumber(v, { digits: 3 });
  const fmtMs = (v) => (typeof v === "number" ? `${v} ms` : null);
  const rows = [
    _metricRow("Tokens", ours.tokens ?? null, theirs.tokens ?? null),
    _metricRow("Zeichen", ours.chars ?? null, theirs.chars ?? null),
    _metricRow("Word-Boxen", ours.word_box_count ?? null, theirs.word_box_count ?? null),
    _metricRow(
      "Ø Konfidenz",
      fmtConf(ours.avg_confidence),
      fmtConf(theirs.avg_confidence),
      {
        tooltip:
          tr("metrics_tooltip_avg_confidence"),
      },
    ),
    _metricRow("Latenz", fmtMs(ours.latency_ms), fmtMs(theirs.latency_ms)),
  ];
  return `
    <table class="metrics-table">
      <thead><tr><th>Metrik</th><th>Wir</th><th>Andere</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>
  `;
}

function _renderComparisonTab(comparison) {
  if (!comparison) return `<p class="compare-reference-hint">${tr("metrics_no_data")}</p>`;
  const fmt = (v) => _fmtNumber(v, { digits: 3 });
  const fmtPct = (v) => _fmtNumber(v, { percent: true });
  return `
    <p class="metrics-section-title">${tr("metrics_symmetric_title")}</p>
    <table class="metrics-table">
      <thead><tr><th>Metrik</th><th>Wert</th></tr></thead>
      <tbody>
        ${_singleMetricRow(
          "Δ Zeichen",
          fmt(comparison.delta_char),
          tr("metrics_tooltip_sym_char"),
        )}
        ${_singleMetricRow(
          "Δ Wörter",
          fmt(comparison.delta_word),
          tr("metrics_tooltip_sym_word"),
        )}
        ${_singleMetricRow(
          "Token-Jaccard",
          fmt(comparison.token_jaccard),
          "|A ∩ B| / |A ∪ B|, ordnungsunabhängig. Höher = mehr gemeinsame Tokens.",
        )}
      </tbody>
    </table>
    <p class="metrics-section-title">${tr("metrics_asymmetric_title", { name: "Other" })}</p>
    <table class="metrics-table">
      <thead><tr><th>Metrik</th><th>Wir vs Andere</th></tr></thead>
      <tbody>
        ${_singleMetricRow(
          "Token-Precision",
          fmtPct(comparison.token_precision),
          tr("metrics_tooltip_precision"),
        )}
        ${_singleMetricRow(
          "Token-Recall",
          fmtPct(comparison.token_recall),
          tr("metrics_tooltip_recall"),
        )}
        ${_singleMetricRow("Token-F1", fmtPct(comparison.token_f1))}
      </tbody>
    </table>
  `;
}

function _singleMetricRow(label, value, tooltip) {
  const helpHtml = tooltip
    ? ` <span class="metric-help" title="${escapeHtml(tooltip)}">ⓘ</span>`
    : "";
  if (value === null || value === undefined) {
    return `<tr><th>${escapeHtml(label)}${helpHtml}</th><td class="metric-empty">—</td></tr>`;
  }
  return `<tr><th>${escapeHtml(label)}${helpHtml}</th><td class="metric-value">${escapeHtml(String(value))}</td></tr>`;
}

function _renderReferenceTab(reference) {
  if (!reference) {
    return `<p class="compare-reference-hint">${tr("metrics_no_reference")}</p>`;
  }
  const ours = reference.ours || {};
  const theirs = reference.theirs;  // null wenn kein Compare-Lauf erfolgte
  const fmt = (v) => _fmtNumber(v, { digits: 3 });
  const fmtPct = (v) => _fmtNumber(v, { percent: true });
  const showTheirs = theirs && typeof theirs === "object";
  const tHead = showTheirs
    ? `<thead><tr><th>Metrik</th><th>Wir</th><th>Andere</th></tr></thead>`
    : `<thead><tr><th>Metrik</th><th>Wir</th></tr></thead>`;
  const row = (label, oursValue, theirsValue, opts) =>
    showTheirs ? _metricRow(label, oursValue, theirsValue, opts) : _singleMetricRow(label, oursValue, opts?.tooltip);
  const theirsRef = theirs || {};
  const rows = [
    row("CER", fmt(ours.cer), fmt(theirsRef.cer), {
      tooltip: tr("metrics_tooltip_cer"),
    }),
    row("WER", fmt(ours.wer), fmt(theirsRef.wer), {
      tooltip: tr("metrics_tooltip_wer"),
    }),
    row("Token-Precision", fmtPct(ours.token_precision), fmtPct(theirsRef.token_precision)),
    row("Token-Recall", fmtPct(ours.token_recall), fmtPct(theirsRef.token_recall)),
    row("Token-F1", fmtPct(ours.token_f1), fmtPct(theirsRef.token_f1)),
  ];
  return `
    <p class="metrics-section-title">${tr("metrics_reference_title", { tokens: reference.token_count ?? "?", chars: reference.char_count ?? "?" })}</p>
    <table class="metrics-table">
      ${tHead}
      <tbody>${rows.join("")}</tbody>
    </table>
  `;
}

function renderMetricsPanel(metrics) {
  const hasAny = !!(metrics && (metrics.intrinsic || metrics.comparison || metrics.reference));
  // Reveal/hide the Metriken result-view tab itself based on availability.
  if (resultViewMetricsBtnEl) {
    resultViewMetricsBtnEl.classList.toggle("hidden", !hasAny);
  }
  if (metricsEmptyEl) {
    metricsEmptyEl.classList.toggle("hidden", hasAny);
  }
  if (!hasAny) {
    metricsPanelEl?.classList.add("hidden");
    if (metricsContentEl) metricsContentEl.innerHTML = "";
    return;
  }
  metricsPanelEl?.classList.remove("hidden");

  const setTabState = (name, enabled, disabledTitle) => {
    const btn = metricsTabBtns.find((b) => b.dataset.metricsTab === name);
    if (!btn) return;
    if (enabled) {
      btn.removeAttribute("disabled");
      btn.removeAttribute("title");
    } else {
      btn.setAttribute("disabled", "");
      btn.setAttribute("title", disabledTitle || "");
    }
  };
  setTabState("intrinsic", !!metrics.intrinsic, tr("metrics_tab_requires_compare"));
  setTabState("comparison", !!metrics.comparison, tr("metrics_tab_requires_compare"));
  setTabState("reference", !!metrics.reference, tr("metrics_tab_requires_reference"));

  const enabledOrder = ["intrinsic", "comparison", "reference"].filter((name) => {
    const btn = metricsTabBtns.find((b) => b.dataset.metricsTab === name);
    return btn && !btn.hasAttribute("disabled");
  });
  if (!enabledOrder.includes(activeMetricsTab)) {
    activeMetricsTab = enabledOrder[0] || "intrinsic";
  }
  _renderActiveMetricsTab();
}

function _renderActiveMetricsTab() {
  if (!metricsContentEl || !lastMetrics) return;
  let html = "";
  if (activeMetricsTab === "intrinsic") html = _renderIntrinsicTab(lastMetrics.intrinsic);
  else if (activeMetricsTab === "comparison") html = _renderComparisonTab(lastMetrics.comparison);
  else if (activeMetricsTab === "reference") html = _renderReferenceTab(lastMetrics.reference);
  metricsContentEl.innerHTML = html;
  metricsTabBtns.forEach((btn) => {
    const active = btn.dataset.metricsTab === activeMetricsTab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
}

metricsTabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.hasAttribute("disabled")) return;
    activeMetricsTab = btn.dataset.metricsTab;
    _renderActiveMetricsTab();
  });
});

function effectiveReferenceText() {
  if (manualReferenceText && manualReferenceText.trim()) return manualReferenceText;
  if (referenceIgnoreEmbeddedEl?.checked) return "";
  return embeddedReferenceText || "";
}

function _syncReferenceTextarea() {
  if (!referenceTextEl) return;
  if (manualReferenceText && manualReferenceText.trim()) {
    referenceTextEl.value = manualReferenceText;
  } else if (embeddedReferenceText && !referenceIgnoreEmbeddedEl?.checked) {
    referenceTextEl.value = embeddedReferenceText;
  } else {
    referenceTextEl.value = "";
  }
}

function _setEmbeddedReference(text) {
  embeddedReferenceText = text || "";
  if (embeddedReferenceText) {
    referenceSourceBadgeEl?.classList.remove("hidden");
    referenceIgnoreWrapEl?.classList.remove("hidden");
    if (compareReferenceEl && !compareReferenceEl.open) compareReferenceEl.open = true;
  } else {
    referenceSourceBadgeEl?.classList.add("hidden");
    referenceIgnoreWrapEl?.classList.add("hidden");
    if (referenceIgnoreEmbeddedEl) referenceIgnoreEmbeddedEl.checked = false;
  }
  _syncReferenceTextarea();
}

async function maybeFetchEmbeddedPdfReference(file) {
  _setEmbeddedReference("");
  if (!file || file.type !== "application/pdf") return;
  try {
    const fd = new FormData();
    fd.append("file", file);
    const resp = await fetch(extractPdfTextEndpoint, { method: "POST", body: fd });
    if (!resp.ok) return;
    const data = await resp.json();
    if (data?.has_text_layer && data.text) {
      _setEmbeddedReference(String(data.text));
      if (typeof data.garbage_ratio === "number" && data.garbage_ratio > 0.3) {
        // Soft warning baked into the badge title; user can decide.
        referenceSourceBadgeEl?.setAttribute(
          "title",
          `Heuristik meldet ${Math.round(data.garbage_ratio * 100)} % unleserliche Zeichen — eventuell ignorieren.`,
        );
      } else {
        referenceSourceBadgeEl?.removeAttribute("title");
      }
    }
  } catch {
    /* network error → silent: user can still paste manually */
  }
}

let _referenceMetricsDebounce = 0;
function _scheduleReferenceMetricsRefresh() {
  // OCR muss vorher gelaufen sein — sonst gibt's keine Hypothese.
  const hyp = lastResponse?.text || "";
  if (!hyp) {
    // Wenn keine OCR-Antwort vorliegt, nur den Referenz-Block leeren.
    if (lastMetrics?.reference) {
      lastMetrics = { ...lastMetrics, reference: null };
      renderMetricsPanel(lastMetrics);
    }
    return;
  }
  window.clearTimeout(_referenceMetricsDebounce);
  _referenceMetricsDebounce = window.setTimeout(() => {
    void maybeFetchReferenceMetrics(hyp);
  }, 350);
}

referenceTextEl?.addEventListener("input", () => {
  manualReferenceText = referenceTextEl.value;
  _scheduleReferenceMetricsRefresh();
});

referenceClearBtnEl?.addEventListener("click", () => {
  manualReferenceText = "";
  if (referenceTextEl) referenceTextEl.value = "";
  if (embeddedReferenceText && !referenceIgnoreEmbeddedEl?.checked) {
    _syncReferenceTextarea();
  }
  _scheduleReferenceMetricsRefresh();
});

referenceFileEl?.addEventListener("change", async () => {
  const file = referenceFileEl.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    manualReferenceText = text;
    if (referenceTextEl) referenceTextEl.value = text;
  } catch {
    /* ignore */
  }
  referenceFileEl.value = "";
  _scheduleReferenceMetricsRefresh();
});

referenceIgnoreEmbeddedEl?.addEventListener("change", () => {
  _syncReferenceTextarea();
  _scheduleReferenceMetricsRefresh();
});

async function runOCR() {
  const selectedFile = currentFile();
  if (!selectedFile) {
    if (activeRequestController) {
      activeRequestController.abort();
      activeRequestController = null;
    }
    setLoading(false);
    metaEl.textContent = tr("meta_pick_file");
    clearOutput();
    setWorkspaceVisible(false);
    return;
  }

  if (activeRequestController) {
    activeRequestController.abort();
  }
  const controller = new AbortController();
  activeRequestController = controller;
  const payload = buildPayload();
  setLoading(true);
  clearOutput();
  setWorkspaceVisible(true);
  metaEl.textContent = tr("meta_running");

  try {
    const requestMode = String(payload.get("mode") || "plain");
    const requestTask = String(payload.get("task") || "");
    const response = await fetch(ocrEndpoint, {
      method: "POST",
      body: payload,
      signal: controller.signal,
    });
    if (!response.ok) {
      let detail;
      try { detail = (await response.json()).detail; } catch { detail = null; }
      throw new Error(detail || `HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json();
    applyOcrResponse(data, {
      requestMode,
      requestTask,
      backendFallback: String(payload.get("backend") || "direct"),
    });
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    metaEl.textContent = tr("meta_error", { message: error.message });
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
  void maybeFetchEmbeddedPdfReference(currentFile());
  void runOCR();
});
previewImageEl.addEventListener("load", () => {
  const layoutPages = normalizeLayoutPages(lastResponse?.layout);
  renderLayoutOverlay(layoutPages, currentPageIndex);
  if (previewZoomEl) previewZoomEl.classList.remove("hidden");
  applyPreviewZoom(1);
});

function applyPreviewZoom(scale) {
  const clamped = Math.min(3, Math.max(0.5, Number(scale) || 1));
  if (previewImageEl) {
    previewImageEl.style.transform = clamped === 1 ? "" : `scale(${clamped})`;
  }
  if (previewLayoutOverlayEl) {
    previewLayoutOverlayEl.style.transform = previewImageEl?.style.transform || "";
    previewLayoutOverlayEl.style.transformOrigin = "top center";
  }
  if (previewZoomSliderEl) previewZoomSliderEl.value = String(clamped);
  if (previewZoomValueEl) previewZoomValueEl.textContent = `${Math.round(clamped * 100)} %`;
}

previewZoomSliderEl?.addEventListener("input", () => {
  applyPreviewZoom(previewZoomSliderEl.value);
});
previewZoomFitBtnEl?.addEventListener("click", () => applyPreviewZoom(1));
previewZoomInBtnEl?.addEventListener("click", () => {
  const cur = Number(previewZoomSliderEl?.value || 1);
  applyPreviewZoom(cur + 0.1);
});
previewZoomOutBtnEl?.addEventListener("click", () => {
  const cur = Number(previewZoomSliderEl?.value || 1);
  applyPreviewZoom(cur - 0.1);
});
previewImageEl.addEventListener("error", () => {
  const file = currentFile();
  clearLayoutOverlay();
  previewImageStageEl.classList.add("hidden");
  previewImageEl.classList.add("hidden");
  previewImageEl.removeAttribute("src");
  previewEmptyEl.textContent = isTiffLikeFile(file)
    ? tr("preview_tiff_unsupported")
    : tr("preview_failed", { name: file?.type || file?.name || "?" });
  previewEmptyEl.classList.remove("hidden");
  if (previewUrl) {
    previewPdfLinkEl.href = previewUrl;
    previewPdfLinkEl.textContent = tr("open_file_new_tab");
    previewPdfLinkEl.classList.remove("hidden");
  }
});
modeEl.addEventListener("change", () => {
  toggleModeDependentFields();
});
advancedPanelEl.addEventListener("change", () => {
  setAdvancedDirty(true);
});
advancedPanelEl.addEventListener("input", () => {
  setAdvancedDirty(true);
});
advancedPanelEl
  .querySelectorAll("input, select, textarea")
  .forEach((el) => {
    el.addEventListener("change", () => setAdvancedDirty(true));
    el.addEventListener("input", () => setAdvancedDirty(true));
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

// Page selector for PDF previews
pageSelectorEl.addEventListener("change", () => {
  const index = parseInt(pageSelectorEl.value, 10);
  if (isNaN(index)) return;
  showPageImage(index);
});


// Compare with Azure endpoint
function _activeEngineName() {
  return compareEngineEl?.value?.trim() || "azure";
}

function _showEngineFields(name) {
  engineFieldGroups.forEach((group) => {
    const matches = group.dataset.engine === name;
    group.classList.toggle("hidden", !matches);
  });
  if (name === "local_models") {
    void _ensureLocalModelOptions();
  }
}

const _peerModelsEndpoint = `${appBasePath}/api/peer-models`;
const _defaultModelName = (document.body?.dataset.defaultModel || "").trim();
const _defaultInferenceProvider = (
  document.body?.dataset.defaultInferenceProvider || "ollama"
).trim();
const inferenceProviderEl = document.getElementById("inference_provider");
const modelEl = document.getElementById("model");
const modelSuggestionsEl = document.getElementById("model-suggestions");
const modelHintEl = document.getElementById("model-hint");
const INFERENCE_PROVIDER_KEY = "docread-inference-provider";
let _localModelsCacheByProvider = {};

function _modelsEndpointForProvider(providerId) {
  const provider = providerId || _defaultInferenceProvider;
  return `${appBasePath}/api/models?vision_only=true&provider=${encodeURIComponent(provider)}`;
}

async function _fetchLocalModels() {
  const providerId = inferenceProviderEl?.value?.trim() || _defaultInferenceProvider;
  if (_localModelsCacheByProvider[providerId]) {
    return _localModelsCacheByProvider[providerId];
  }
  try {
    const resp = await fetch(_modelsEndpointForProvider(providerId));
    if (!resp.ok) return [];
    const data = await resp.json();
    _localModelsCacheByProvider[providerId] = Array.isArray(data?.models) ? data.models : [];
  } catch {
    _localModelsCacheByProvider[providerId] = [];
  }
  return _localModelsCacheByProvider[providerId];
}

function _populateInferenceProviderSelect(providers, defaultProvider) {
  if (!inferenceProviderEl) return;
  inferenceProviderEl.innerHTML = "";
  for (const entry of providers) {
    const id = typeof entry === "string" ? entry : entry?.id;
    if (!id) continue;
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    inferenceProviderEl.appendChild(opt);
  }
  const saved = localStorage.getItem(INFERENCE_PROVIDER_KEY);
  const ids = providers.map((entry) => (typeof entry === "string" ? entry : entry?.id));
  inferenceProviderEl.value =
    saved && ids.includes(saved) ? saved : defaultProvider || ids[0] || _defaultInferenceProvider;
}

function _populateModelSuggestions(models) {
  if (!modelSuggestionsEl) return;
  modelSuggestionsEl.innerHTML = "";
  for (const name of models) {
    const opt = document.createElement("option");
    opt.value = name;
    modelSuggestionsEl.appendChild(opt);
  }
  if (modelHintEl) {
    const providerId = inferenceProviderEl?.value?.trim() || _defaultInferenceProvider;
    modelHintEl.textContent = models.length
      ? tr("models_vision_hint", { count: models.length, provider: providerId })
      : tr("models_vision_empty", { provider: providerId });
  }
}

async function _refreshModelSuggestions() {
  const models = await _fetchLocalModels();
  _populateModelSuggestions(models);
  if (modelEl && !modelEl.value.trim() && _defaultModelName) {
    modelEl.placeholder = _defaultModelName;
  }
}

async function _initInferenceControls() {
  if (!inferenceProviderEl) return;
  try {
    const resp = await fetch(`${appBasePath}/api/inference-providers`);
    if (!resp.ok) throw new Error(String(resp.status));
    const data = await resp.json();
    const providers = Array.isArray(data?.providers) ? data.providers : [];
    const defaultProvider = String(data?.default_provider || _defaultInferenceProvider);
    if (providers.length === 0) {
      _populateInferenceProviderSelect([defaultProvider], defaultProvider);
    } else {
      _populateInferenceProviderSelect(providers, defaultProvider);
    }
  } catch {
    _populateInferenceProviderSelect([_defaultInferenceProvider], _defaultInferenceProvider);
  }
  await _refreshModelSuggestions();
}

inferenceProviderEl?.addEventListener("change", () => {
  localStorage.setItem(INFERENCE_PROVIDER_KEY, inferenceProviderEl.value);
  void _refreshModelSuggestions();
  if (ourModelSelectEl || theirModelSelectEl) {
    _localModelsCacheByProvider = {};
    void _ensureLocalModelOptions();
  }
});

void _initInferenceControls();

function _populateModelSelect(selectEl, models, preferred) {
  if (!selectEl) return;
  selectEl.innerHTML = "";
  if (models.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = tr("models_none");
    opt.disabled = true;
    selectEl.appendChild(opt);
    return;
  }
  for (const name of models) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    selectEl.appendChild(opt);
  }
  if (preferred && models.includes(preferred)) {
    selectEl.value = preferred;
  }
}

async function _ensureLocalModelOptions() {
  if (ourModelSelectEl?.options.length && theirModelSelectEl?.options.length) return;
  const models = await _fetchLocalModels();
  _populateModelSelect(ourModelSelectEl, models, _defaultModelName);
  // Pick a different default for "their-model" so the comparison is non-trivial.
  const fallbackOther = models.find((n) => n !== _defaultModelName) || _defaultModelName;
  _populateModelSelect(theirModelSelectEl, models, fallbackOther);
}

let _peerModelsAbortController = null;

async function _refreshPeerModels() {
  if (!peerBaseUrlEl) return;
  const url = peerBaseUrlEl.value.trim();
  if (!url) {
    peerModelSelectEl?.classList.add("hidden");
    peerModelInputEl?.classList.remove("hidden");
    if (peerModelHintEl) {
      peerModelHintEl.textContent = "";
      peerModelHintEl.classList.add("hidden");
    }
    return;
  }
  if (_peerModelsAbortController) _peerModelsAbortController.abort();
  _peerModelsAbortController = new AbortController();
  if (peerModelHintEl) {
    peerModelHintEl.textContent = tr("peer_models_loading");
    peerModelHintEl.classList.remove("hidden");
  }
  try {
    const resp = await fetch(
      `${_peerModelsEndpoint}?url=${encodeURIComponent(url)}`,
      { signal: _peerModelsAbortController.signal },
    );
    if (!resp.ok) throw new Error(String(resp.status));
    const data = await resp.json();
    const models = Array.isArray(data?.models) ? data.models : [];
    if (models.length === 0) throw new Error("leer");
    _populateModelSelect(peerModelSelectEl, models, _defaultModelName);
    peerModelSelectEl?.classList.remove("hidden");
    peerModelInputEl?.classList.add("hidden");
    if (peerModelHintEl) {
      peerModelHintEl.textContent = tr("peer_models_loaded", { count: models.length });
      peerModelHintEl.classList.remove("hidden");
    }
  } catch (err) {
    if (err?.name === "AbortError") return;
    peerModelSelectEl?.classList.add("hidden");
    peerModelInputEl?.classList.remove("hidden");
    if (peerModelHintEl) {
      peerModelHintEl.textContent = tr("peer_models_unreachable");
      peerModelHintEl.classList.remove("hidden");
    }
  }
}

let _peerUrlDebounce = 0;
peerBaseUrlEl?.addEventListener("input", () => {
  window.clearTimeout(_peerUrlDebounce);
  _peerUrlDebounce = window.setTimeout(() => void _refreshPeerModels(), 350);
});
peerBaseUrlEl?.addEventListener("change", () => void _refreshPeerModels());

if (compareEngineEl) {
  _showEngineFields(_activeEngineName());
  compareEngineEl.addEventListener("change", () => _showEngineFields(_activeEngineName()));
}

function _appendEngineConfig(fd, engineName) {
  const get = (id) => document.getElementById(id)?.value?.trim() || "";
  if (engineName === "azure") {
    if (!get("azure-endpoint")) return "Azure-Endpunkt fehlt.";
    fd.append("azure_endpoint", get("azure-endpoint"));
    fd.append("azure_key", get("azure-key"));
  } else if (engineName === "local_models") {
    const our = ourModelSelectEl?.value?.trim() || "";
    const their = theirModelSelectEl?.value?.trim() || "";
    if (!our || !their) return tr("pick_both_models");
    fd.append("our_model", our);
    fd.append("their_model", their);
  } else if (engineName === "self_peer") {
    if (!get("peer-base-url")) return tr("peer_url_missing");
    fd.append("peer_base_url", get("peer-base-url"));
    if (get("peer-backend")) fd.append("peer_backend", get("peer-backend"));
    const peerModel = peerModelSelectEl && !peerModelSelectEl.classList.contains("hidden")
      ? peerModelSelectEl.value?.trim() || ""
      : peerModelInputEl?.value?.trim() || "";
    if (peerModel) fd.append("peer_model", peerModel);
  } else if (engineName === "google_vision") {
    if (!get("google-api-key")) return "Google-Vision-API-Key fehlt.";
    fd.append("google_api_key", get("google-api-key"));
  } else if (engineName === "plain_text") {
    if (!get("plain-text-url")) return "Plain-Text-Endpunkt-URL fehlt.";
    fd.append("plain_text_url", get("plain-text-url"));
    if (get("plain-text-method")) fd.append("plain_text_method", get("plain-text-method"));
    if (get("plain-text-field")) fd.append("plain_text_field", get("plain-text-field"));
    if (get("plain-text-auth-header"))
      fd.append("plain_text_auth_header", get("plain-text-auth-header"));
    if (get("plain-text-auth-value"))
      fd.append("plain_text_auth_value", get("plain-text-auth-value"));
  } else {
    return `Unbekannte Engine: ${engineName}`;
  }
  return null;
}

compareFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = currentFile();
  if (!file) {
    compareSummaryEl.textContent = tr("compare_no_image");
    compareSummaryEl.classList.remove("hidden");
    return;
  }

  const engineName = _activeEngineName();
  const fd = new FormData();
  fd.append("file", file);
  fd.append("engine", engineName);
  const configError = _appendEngineConfig(fd, engineName);
  if (configError) {
    compareSummaryEl.textContent = configError;
    compareSummaryEl.classList.remove("hidden");
    return;
  }

  compareSummaryEl.textContent = tr("compare_running");
  compareSummaryEl.classList.remove("hidden");
  lastDiff = null;
  lastMetrics = null;
  renderMetricsPanel(null);
  clearDiffPanel();
  renderDiffOverlay(null);

  const appendIfSet = (name, rawValue, { bool = false } = {}) => {
    const value = String(rawValue ?? "").trim();
    if (!value) return;
    if (bool && value !== "true" && value !== "false") return;
    fd.append(name, value);
  };
  appendIfSet("backend", document.getElementById("backend")?.value);
  appendIfSet("expert_enable_layout", document.getElementById("expert_enable_layout")?.value, {
    bool: true,
  });
  appendIfSet("expert_layout_model", document.getElementById("expert_layout_model")?.value);
  appendIfSet(
    "expert_layout_threshold",
    document.getElementById("expert_layout_threshold")?.value,
  );
  appendIfSet("expert_table_transformer", document.getElementById("expert_table_transformer")?.value, {
    bool: true,
  });
  appendIfSet("expert_per_region_ocr", document.getElementById("expert_per_region_ocr")?.value, {
    bool: true,
  });
  appendIfSet("expert_text_anchor", document.getElementById("expert_text_anchor")?.value, {
    bool: true,
  });
  appendIfSet(
    "expert_text_anchor_threshold",
    document.getElementById("expert_text_anchor_threshold")?.value,
  );
  appendIfSet("expert_word_detector", document.getElementById("expert_word_detector")?.value);
  appendIfSet(
    "expert_compare_include_detector_only",
    document.getElementById("expert_compare_include_detector_only")?.value,
    { bool: true },
  );
  const refText = effectiveReferenceText();
  if (refText && refText.trim()) {
    fd.append("reference_text", refText);
  }
  if (!fd.has("expert_enable_layout") && lastResponse?.analyzeResult?.pages?.[0]) {
    fd.append("expert_enable_layout", "true");
  }

  setLoading(true);
  try {
    const resp = await fetch(compareEndpoint, { method: "POST", body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      compareSummaryEl.textContent = tr("error_http", { detail: err.detail || resp.statusText });
      return;
    }
    const data = await resp.json();
    applyCompareResponse(data);
  } catch (err) {
    compareSummaryEl.textContent = tr("network_error", { message: err.message });
  } finally {
    setLoading(false);
  }
});

if (comparePresetBtn) {
  comparePresetBtn.addEventListener("click", () => {
    const endpoint = comparePresetBtn.dataset.endpoint || "";
    if (!endpoint) return;
    if (compareEngineEl) {
      compareEngineEl.value = "azure";
      _showEngineFields("azure");
    }
    azureEndpointEl.value = endpoint;
    compareFormEl.requestSubmit();
  });
}

if (comparePresetLayoutBtn) {
  comparePresetLayoutBtn.addEventListener("click", () => {
    const endpoint = comparePresetLayoutBtn.dataset.endpoint || "";
    if (!endpoint) return;
    if (compareEngineEl) {
      compareEngineEl.value = "azure";
      _showEngineFields("azure");
    }
    azureEndpointEl.value = endpoint;
    compareFormEl.requestSubmit();
  });
}

async function _loadExampleFile(slot) {
  const resp = await fetch(`${appBasePath}/api/examples/${slot}`);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || resp.statusText);
  }
  const blob = await resp.blob();
  const disposition = resp.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i);
  const filename = match ? decodeURIComponent(match[1]) : `example-${slot}`;
  const file = new File([blob], filename, { type: blob.type });
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileEl.files = transfer.files;
  setWorkspaceVisible(true);
  updatePreview();
  void maybeFetchEmbeddedPdfReference(file);
}

async function runExample(slot) {
  const presetEndpoint = comparePresetBtn?.dataset.endpoint || "";

  // Try the warm-at-boot cache first. Cache hit → instant: load the file
  // for the preview, then apply the cached OCR + compare responses.
  setLoading(true);
  try {
    let cached = null;
    try {
      const cachedResp = await fetch(`${appBasePath}/api/examples/${slot}/cached`);
      if (cachedResp.ok) {
        cached = await cachedResp.json();
      }
    } catch {
      // Network error reaching the cached endpoint; fall through to live.
    }

    if (cached?.ocr_response) {
      try {
        await _loadExampleFile(slot);
      } catch (err) {
        metaEl.textContent = tr("example_load_failed", { message: err.message });
        return;
      }
      applyOcrResponse(cached.ocr_response, {
        requestMode: "plain",
        requestTask: "ocr_text",
        backendFallback: cached.ocr_response.backend || "expert",
      });
      if (cached.compare_response) {
        applyCompareResponse(cached.compare_response);
      }
      return;
    }
  } finally {
    setLoading(false);
  }

  // Cache miss / not warmed yet → live path: fetch file, runOCR, compare.
  try {
    await _loadExampleFile(slot);
  } catch (err) {
    metaEl.textContent = tr("example_load_failed", { message: err.message });
    return;
  }
  await runOCR();
  if (presetEndpoint) {
    azureEndpointEl.value = presetEndpoint;
    compareFormEl.requestSubmit();
  }
}

document.querySelectorAll("button[data-example-slot]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const slot = Number(btn.dataset.exampleSlot);
    if (Number.isFinite(slot) && slot > 0) {
      void runExample(slot);
    }
  });
});

function refreshUiOnLocaleChange() {
  window.docreadApplyI18n?.();
  if (advancedToggleEl) {
    setAdvancedOpen(advancedToggleEl.getAttribute("aria-expanded") === "true");
  }
  if (lastResponse && lastOcrApplyContext) {
    applyOcrResponse(lastResponse, lastOcrApplyContext);
  }
  if (lastCompareResponse) {
    applyCompareResponse(lastCompareResponse);
  }
}

document.addEventListener("docread:locale-change", refreshUiOnLocaleChange);

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
