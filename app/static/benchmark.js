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
          `<label><input type="checkbox" name="model" value="${escapeHtml(m)}" /> ${escapeHtml(m)}</label>`,
      )
      .join("");
  } catch (err) {
    modelsEl.innerHTML = `<p class="bench-empty">Modellliste konnte nicht geladen werden: ${escapeHtml(err.message)}</p>`;
  }
}

fileEl.addEventListener("change", async () => {
  pickedFiles = Array.from(fileEl.files || []);
  fileListEl.innerHTML = pickedFiles
    .map(
      (f, i) => `
      <div class="bench-file-row">
        <div class="bench-file-name">${escapeHtml(f.name)}</div>
        <textarea data-ref-index="${i}" placeholder="Optionaler Referenztext für CER/WER…"></textarea>
      </div>`,
    )
    .join("");

  for (let i = 0; i < pickedFiles.length; i++) {
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

function renderJob(job) {
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
  tableBodyEl.innerHTML = (rows || [])
    .map((row) => {
      const warnings = (row.warnings || []).join(" | ");
      const cls = row.status === "running"
        ? "bench-status-running"
        : row.status === "error"
        ? "bench-status-error"
        : "";
      return `
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
        </tr>`;
    })
    .join("");

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
