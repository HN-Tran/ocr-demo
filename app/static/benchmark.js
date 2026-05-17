"use strict";

const appBasePath = (document.body?.dataset.basePath || "").replace(/\/$/, "");
const fileEl = document.getElementById("bench-files");
const fileListEl = document.getElementById("bench-file-list");
const modelsEl = document.getElementById("bench-models");
const startBtn = document.getElementById("bench-start");
const statusEl = document.getElementById("bench-status");
const resultsEl = document.getElementById("bench-results");
const cancelBtn = document.getElementById("bench-cancel");
const csvLinkEl = document.getElementById("bench-csv-link");
const mlflowLinkEl = document.getElementById("bench-mlflow-link");
const aggEl = document.getElementById("bench-aggregate");
const tableBodyEl = document.getElementById("bench-table-body");

let pickedFiles = [];
let activeJobId = null;
let pollTimer = 0;
const expandedRows = new Set();

const themeToggleBtn = document.getElementById("bench-theme-toggle");
themeToggleBtn?.addEventListener("click", () => {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  if (isDark) {
    document.documentElement.removeAttribute("data-theme");
    localStorage.setItem("ocr-demo-theme", "light");
    themeToggleBtn.textContent = "☾";
  } else {
    document.documentElement.setAttribute("data-theme", "dark");
    localStorage.setItem("ocr-demo-theme", "dark");
    themeToggleBtn.textContent = "☀";
  }
});
if (document.documentElement.getAttribute("data-theme") === "dark" && themeToggleBtn) {
  themeToggleBtn.textContent = "☀";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Number(value).toFixed(digits);
}

function fmtMs(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${Math.round(value)} ms`;
}

async function loadModels() {
  try {
    const resp = await fetch(`${appBasePath}/api/models?vision_only=true`);
    if (!resp.ok) throw new Error(String(resp.status));
    const data = await resp.json();
    const models = Array.isArray(data?.models) ? data.models : [];
    if (models.length === 0) {
      modelsEl.innerHTML = `<p class="bench-empty">Keine Vision-Modelle in Ollama gefunden.</p>`;
      return;
    }
    modelsEl.innerHTML = models
      .map(
        (m) =>
          `<div class="bench-model-row">
            <span class="bench-model-name">${escapeHtml(m)}</span>
            <label><input type="checkbox" name="model" value="${escapeHtml(m)}::expert" /> Layout</label>
            <label><input type="checkbox" name="model" value="${escapeHtml(m)}::direct" /> Plain</label>
          </div>`,
      )
      .join("");
  } catch (err) {
    modelsEl.innerHTML = `<p class="bench-empty">Modellliste konnte nicht geladen werden: ${escapeHtml(err.message)}</p>`;
  }
}

fileEl.addEventListener("change", () => addFiles(fileEl.files || []));

const DROP_HINT = `<p class="bench-drop-hint">Bilder, PDFs, Word- oder ZIP-Dateien hier ablegen oder <label for="bench-files" class="bench-drop-label">auswählen</label></p>`;

function isZipFile(f) {
  return f.type === "application/zip" || f.name.toLowerCase().endsWith(".zip");
}

function renderFileList() {
  const hint = pickedFiles.length ? "" : DROP_HINT;
  const rows = pickedFiles
    .map((f, i) => {
      if (isZipFile(f)) {
        return `
        <div class="bench-file-row bench-file-row-zip" data-file-index="${i}">
          <div class="bench-file-name">${escapeHtml(f.name)} <span class="bench-zip-badge">ZIP</span></div>
          <div class="bench-zip-note">Enthaltene Dateien werden entpackt · Referenztexte aus gepaarten .txt-Dateien</div>
          <button type="button" class="bench-file-delete" data-delete-index="${i}" title="Entfernen">✕</button>
        </div>`;
      }
      return `
      <div class="bench-file-row" data-file-index="${i}">
        <div class="bench-file-name">${escapeHtml(f.name)}</div>
        <textarea data-ref-index="${i}" placeholder="Optionaler Referenztext für CER/WER…"></textarea>
        <button type="button" class="bench-file-delete" data-delete-index="${i}" title="Entfernen">✕</button>
      </div>`;
    })
    .join("");
  fileListEl.innerHTML = hint + rows;
  fileListEl.querySelectorAll(".bench-file-delete").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.deleteIndex, 10);
      pickedFiles.splice(idx, 1);
      renderFileList();
    });
  });
}

async function addFiles(files) {
  const incoming = Array.from(files);
  const existingNames = new Set(pickedFiles.map((f) => f.name));
  const newFiles = incoming.filter((f) => !existingNames.has(f.name));
  const startIdx = pickedFiles.length;
  pickedFiles.push(...newFiles);
  renderFileList();

  for (let i = startIdx; i < pickedFiles.length; i++) {
    const f = pickedFiles[i];
    if (f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf")) {
      try {
        const fd = new FormData();
        fd.append("file", f);
        const resp = await fetch(`${appBasePath}/api/benchmark/extract-text`, { method: "POST", body: fd });
        if (resp.ok) {
          const { text } = await resp.json();
          if (text) {
            const ta = fileListEl.querySelector(`textarea[data-ref-index="${i}"]`);
            if (ta) ta.value = text;
          }
        }
      } catch (_) {}
    }
  }
}

renderFileList();

fileListEl.addEventListener("dragover", (e) => {
  e.preventDefault();
  fileListEl.classList.add("bench-drop-active");
});
fileListEl.addEventListener("dragleave", () => fileListEl.classList.remove("bench-drop-active"));
fileListEl.addEventListener("drop", (e) => {
  e.preventDefault();
  fileListEl.classList.remove("bench-drop-active");
  if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
});

cancelBtn?.addEventListener("click", () => {
  resultsEl.classList.add("hidden");
  if (pollTimer) {
    window.clearInterval(pollTimer);
    pollTimer = 0;
  }
  activeJobId = null;
});

startBtn.addEventListener("click", async () => {
  if (pickedFiles.length === 0) {
    statusEl.textContent = "Erst Dateien wählen.";
    statusEl.classList.remove("hidden");
    return;
  }
  const models = Array.from(
    document.querySelectorAll('input[name="model"]:checked'),
  ).map((el) => el.value);
  const engines = Array.from(
    document.querySelectorAll('input[name="engine"]:checked'),
  ).map((el) => el.value);
  if (models.length + engines.length === 0) {
    statusEl.textContent = "Mindestens ein Modell oder eine Engine wählen.";
    statusEl.classList.remove("hidden");
    return;
  }

  const fd = new FormData();
  pickedFiles.forEach((f) => fd.append("files", f));
  pickedFiles.forEach((_, i) => {
    const ta = document.querySelector(`textarea[data-ref-index="${i}"]`);
    fd.append("references", ta?.value || "");
  });
  fd.append("models", models.join(","));
  fd.append("engines", engines.join(","));

  const get = (id) => document.getElementById(id)?.value?.trim() || "";
  const config = {
    azure_endpoint: get("bench-azure-endpoint"),
    azure_key: get("bench-azure-key"),
    peer_base_url: get("bench-peer-base-url"),
    peer_model: get("bench-peer-model"),
    google_api_key: get("bench-google-api-key"),
    plain_text_url: get("bench-plain-text-url"),
    plain_text_field: get("bench-plain-text-field"),
  };
  Object.entries(config).forEach(([k, v]) => v && fd.append(k, v));

  startBtn.disabled = true;
  statusEl.textContent = "Job wird angelegt…";
  statusEl.classList.remove("hidden");

  try {
    const resp = await fetch(`${appBasePath}/api/benchmark`, { method: "POST", body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    activeJobId = data.job_id;
    csvLinkEl.href = `${appBasePath}/api/benchmark/${activeJobId}/csv`;
    resultsEl.classList.remove("hidden");
    aggEl.innerHTML = "";
    tableBodyEl.innerHTML = "";
    startPolling();
  } catch (err) {
    statusEl.textContent = `Fehler: ${err.message}`;
  } finally {
    startBtn.disabled = false;
  }
});

function startPolling() {
  if (pollTimer) window.clearInterval(pollTimer);
  pollTimer = window.setInterval(pollOnce, 2000);
  void pollOnce();
}

async function pollOnce() {
  if (!activeJobId) return;
  try {
    const resp = await fetch(`${appBasePath}/api/benchmark/${activeJobId}`);
    if (!resp.ok) return;
    const job = await resp.json();
    renderJob(job);
    if (job.status === "done" || job.status === "failed") {
      window.clearInterval(pollTimer);
      pollTimer = 0;
    }
  } catch {
    /* network blip — try again next tick */
  }
}

function renderWordDiff(ref, hyp) {
  const rW = ref.trim().split(/\s+/).filter(Boolean);
  const hW = hyp.trim().split(/\s+/).filter(Boolean);
  const m = rW.length, n = hW.length;
  const dp = Array.from({length: m + 1}, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = rW[i-1] === hW[j-1] ? dp[i-1][j-1] + 1 : Math.max(dp[i-1][j], dp[i][j-1]);
  const ops = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && rW[i-1] === hW[j-1]) { ops.unshift({t: "eq", w: hW[j-1]}); i--; j--; }
    else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) { ops.unshift({t: "ins", w: hW[j-1]}); j--; }
    else { ops.unshift({t: "del", w: rW[i-1]}); i--; }
  }
  return ops.map(({t, w}) => {
    if (t === "eq") return `<span>${escapeHtml(w)}</span>`;
    if (t === "ins") return `<span class="bench-diff-ins">${escapeHtml(w)}</span>`;
    return `<span class="bench-diff-del">${escapeHtml(w)}</span>`;
  }).join(" ");
}

let _lastJob = null;

function renderJob(job) {
  _lastJob = job;
  const { progress, rows, aggregate, status, error, mlflow } = job;
  statusEl.textContent =
    error
      ? `Fehler: ${error}`
      : `${status} · ${progress.done}/${progress.total}`;

  if (mlflowLinkEl) {
    if (mlflow?.run_url) {
      mlflowLinkEl.href = mlflow.run_url;
      mlflowLinkEl.classList.remove("hidden");
    } else {
      mlflowLinkEl.classList.add("hidden");
    }
  }

  // Table
  const rowHtml = (rows || []).flatMap((row, idx) => {
    const warnings = (row.warnings || []).join(" | ");
    const cls = row.status === "running"
      ? "bench-status-running"
      : row.status === "error"
      ? "bench-status-error"
      : "";
    const hasDetail = row.status === "done" && row.text;
    const expanded = expandedRows.has(idx);
    const toggleBtn = hasDetail
      ? `<button class="bench-expand-btn" data-row-idx="${idx}" title="Text anzeigen">${expanded ? "▲" : "▼"}</button>`
      : "";
    const mainRow = `
      <tr>
        <td>${escapeHtml(row.file_name)}</td>
        <td>${escapeHtml(row.runner_label)}</td>
        <td class="${cls}">${escapeHtml(row.status)}</td>
        <td>${row.text_tokens || 0}</td>
        <td>${row.text_chars || 0}</td>
        <td>${fmtMs(row.latency_ms)}</td>
        <td>${fmt(row.cer)}</td>
        <td>${fmt(row.wer)}</td>
        <td>${fmt(row.token_f1)}</td>
        <td class="bench-warnings">${escapeHtml(row.error || warnings)}</td>
        <td>${toggleBtn}</td>
      </tr>`;
    if (!hasDetail || !expanded) return [mainRow];
    const diffHtml = row.reference
      ? `<div class="bench-detail-section">
           <div class="bench-detail-label">Diff (Referenz ↔ Erkannt)</div>
           <div class="bench-diff-words">${renderWordDiff(row.reference, row.text)}</div>
         </div>`
      : "";
    const detailRow = `
      <tr class="bench-detail-row">
        <td colspan="11">
          <div class="bench-detail">
            <div class="bench-detail-section">
              <div class="bench-detail-label">Erkannter Text</div>
              <pre class="bench-detail-pre">${escapeHtml(row.text)}</pre>
            </div>
            ${diffHtml}
          </div>
        </td>
      </tr>`;
    return [mainRow, detailRow];
  }).join("");
  tableBodyEl.innerHTML = rowHtml;
  tableBodyEl.querySelectorAll(".bench-expand-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.rowIdx, 10);
      if (expandedRows.has(idx)) expandedRows.delete(idx);
      else expandedRows.add(idx);
      if (_lastJob) renderJob(_lastJob);
    });
  });

  // Aggregate cards
  const per = aggregate?.per_runner || {};
  aggEl.innerHTML = Object.entries(per)
    .map(([label, stats]) => `
      <div class="bench-agg-card">
        <strong>${escapeHtml(label)}</strong>
        <dl>
          <dt>Erfolge</dt><dd>${stats.success_count}/${stats.doc_count}</dd>
          <dt>Ø CER</dt><dd>${fmt(stats.mean_cer)}</dd>
          <dt>Ø WER</dt><dd>${fmt(stats.mean_wer)}</dd>
          <dt>Ø F1</dt><dd>${fmt(stats.mean_token_f1)}</dd>
          <dt>Ø Latenz</dt><dd>${fmtMs(stats.mean_latency_ms)}</dd>
        </dl>
      </div>`)
    .join("");
}

void loadModels();
