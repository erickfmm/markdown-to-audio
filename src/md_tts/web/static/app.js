"use strict";

/* ------------------------------------------------------------------ */
/* i18n                                                                */
/* ------------------------------------------------------------------ */

const I18N = {
  es: {
    newConversion: "Nueva conversión",
    dropzoneHint: "Arrastra archivos .md aquí o haz clic para elegir",
    dropzoneFormats: "Puedes seleccionar varios: conversión por lotes",
    or: "o pega texto",
    docName: "Nombre del documento",
    pasteText: "Texto Markdown",
    engine: "Motor TTS",
    language: "Idioma",
    pauseMs: "Pausa entre párrafos (ms)",
    workers: "Workers",
    device: "Dispositivo",
    auto: "auto",
    convert: "Convertir a audio",
    sending: "Enviando…",
    jobs: "Trabajos",
    noJobs: "Sin trabajos todavía. Convierte un documento para empezar.",
    queued: "En cola",
    running: "Procesando",
    done: "Listo",
    errorState: "Error",
    delete: "Eliminar",
    download: "Descargar",
    fragments: "fragmentos",
    formErrorFiles: "Selecciona al menos un archivo o pega texto.",
    formErrorName: "Escribe un nombre para el documento pegado.",
    backendUp: "API en línea",
    backendDown: "API sin conexión",
    sendFailed: "No se pudo enviar el trabajo",
    emptyDrop: "Suelta los archivos aquí",
  },
  en: {
    newConversion: "New conversion",
    dropzoneHint: "Drop .md files here or click to choose",
    dropzoneFormats: "You can select several: batch conversion",
    or: "or paste text",
    docName: "Document name",
    pasteText: "Markdown text",
    engine: "TTS engine",
    language: "Language",
    pauseMs: "Pause between paragraphs (ms)",
    workers: "Workers",
    device: "Device",
    auto: "auto",
    convert: "Convert to audio",
    sending: "Sending…",
    jobs: "Jobs",
    noJobs: "No jobs yet. Convert a document to get started.",
    queued: "Queued",
    running: "Processing",
    done: "Done",
    errorState: "Error",
    delete: "Delete",
    download: "Download",
    fragments: "fragments",
    formErrorFiles: "Select at least one file or paste some text.",
    formErrorName: "Write a name for the pasted document.",
    backendUp: "API online",
    backendDown: "API offline",
    sendFailed: "Could not submit the job",
    emptyDrop: "Drop files here",
  },
};

let lang = localStorage.getItem("mdtts-lang") || "es";

function t(key) {
  return (I18N[lang] && I18N[lang][key]) || I18N.es[key] || key;
}

function applyLang() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.documentElement.lang = lang;
  document.getElementById("lang-toggle").textContent = lang === "es" ? "EN" : "ES";
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

const $ = (id) => document.getElementById(id);

function fmtDuration(seconds) {
  if (seconds == null) return "";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

function fmtTime(epoch) {
  if (!epoch) return "";
  return new Date(epoch * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

async function fetchJson(url, options) {
  const resp = await fetch(url, options);
  let data = null;
  try {
    data = await resp.json();
  } catch (_) {
    /* respuesta no-JSON */
  }
  if (!resp.ok) {
    const detail = data && (data.detail || data.error);
    throw new Error(detail ? String(detail) : `HTTP ${resp.status}`);
  }
  return data;
}

/* ------------------------------------------------------------------ */
/* Estado                                                              */
/* ------------------------------------------------------------------ */

let jobs = [];
let pollTimer = null;
let sending = false;

/* ------------------------------------------------------------------ */
/* Salud del backend                                                   */
/* ------------------------------------------------------------------ */

async function checkBackend() {
  const el = $("backend-status");
  try {
    await fetchJson("/api/health");
    el.dataset.state = "up";
    el.querySelector(".label").textContent = t("backendUp");
  } catch (_) {
    el.dataset.state = "down";
    el.querySelector(".label").textContent = t("backendDown");
  }
}

/* ------------------------------------------------------------------ */
/* Motores                                                             */
/* ------------------------------------------------------------------ */

async function loadEngines() {
  try {
    const data = await fetchJson("/api/engines");
    const select = $("engine");
    select.innerHTML = "";
    data.engines.forEach((engine) => {
      const option = document.createElement("option");
      option.value = engine.id;
      option.textContent = engine.language_aware
        ? engine.id
        : `${engine.id} (⚠ ${t("language")})`;
      select.appendChild(option);
    });
  } catch (_) {
    /* la UI de salud ya informa del fallo */
  }
}

/* ------------------------------------------------------------------ */
/* Trabajos                                                            */
/* ------------------------------------------------------------------ */

async function refreshJobs() {
  try {
    const data = await fetchJson("/api/jobs");
    jobs = data.jobs || [];
  } catch (_) {
    return; // mantiene la última vista conocida
  }
  renderJobs();
  schedulePoll();
}

function schedulePoll() {
  const active = jobs.some((job) => job.state === "queued" || job.state === "running");
  clearTimeout(pollTimer);
  pollTimer = setTimeout(refreshJobs, active ? 1500 : 5000);
}

function jobNode(job) {
  const li = document.createElement("li");
  li.className = "job";
  li.dataset.id = job.id;

  const head = document.createElement("div");
  head.className = "job-head";

  const name = document.createElement("span");
  name.className = "job-name";
  name.textContent = job.name;
  head.appendChild(name);

  const badge = document.createElement("span");
  badge.className = `badge ${job.state}`;
  badge.textContent = t(
    job.state === "error" ? "errorState" : job.state === "queued" ? "queued" : job.state
  );
  head.appendChild(badge);

  const meta = document.createElement("span");
  meta.className = "job-meta";
  const bits = [job.options.engine, fmtTime(job.created_at)];
  if (job.total > 0) bits.push(`${job.done}/${job.total} ${t("fragments")}`);
  if (job.state === "done" && job.duration != null) bits.push(fmtDuration(job.duration));
  meta.textContent = bits.join(" · ");
  head.appendChild(meta);

  li.appendChild(head);

  if (job.state === "running" || job.state === "queued") {
    const progress = document.createElement("div");
    progress.className = "progress" + (job.total > 0 ? "" : " indeterminate");
    const fill = document.createElement("div");
    fill.className = "fill";
    if (job.total > 0) fill.style.width = `${Math.round((job.done / job.total) * 100)}%`;
    progress.appendChild(fill);
    li.appendChild(progress);
  }

  if (job.state === "error" && job.error) {
    const error = document.createElement("p");
    error.className = "job-error";
    error.textContent = job.error;
    li.appendChild(error);
  }

  if (job.state === "done" && job.audio_url) {
    const actions = document.createElement("div");
    actions.className = "job-actions";

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = job.audio_url;
    actions.appendChild(audio);

    const download = document.createElement("a");
    download.className = "btn link";
    download.href = job.audio_url;
    download.download = job.filename || `${job.name}.wav`;
    download.textContent = t("download");
    actions.appendChild(download);

    li.appendChild(actions);
  }

  if (job.state !== "running") {
    const del = document.createElement("button");
    del.type = "button";
    del.className = "btn danger";
    del.textContent = `✕ ${t("delete")}`;
    del.addEventListener("click", () => deleteJob(job.id));
    (li.querySelector(".job-actions") || li).appendChild(del);
  }

  return li;
}

function renderJobs() {
  const list = $("job-list");
  list.innerHTML = "";
  $("empty-jobs").hidden = jobs.length > 0;
  jobs.forEach((job) => list.appendChild(jobNode(job)));
}

async function deleteJob(id) {
  try {
    await fetchJson(`/api/jobs/${id}`, { method: "DELETE" });
  } catch (_) {
    /* ignorado: el polling reconciliará el estado */
  }
  refreshJobs();
}

/* ------------------------------------------------------------------ */
/* Formulario                                                          */
/* ------------------------------------------------------------------ */

function selectedFiles() {
  return Array.from($("file-input").files || []);
}

function renderFileList() {
  const files = selectedFiles();
  const ul = $("file-list");
  ul.innerHTML = "";
  ul.hidden = files.length === 0;
  files.forEach((file) => {
    const li = document.createElement("li");
    li.textContent = `📄 ${file.name}`;
    ul.appendChild(li);
  });
}

function showFormError(message) {
  const el = $("form-error");
  if (!message) {
    el.hidden = true;
    el.textContent = "";
  } else {
    el.hidden = false;
    el.textContent = message;
  }
}

async function submit() {
  if (sending) return;
  showFormError("");

  const files = selectedFiles();
  const text = $("doc-text").value.trim();
  const common = {
    engine: $("engine").value || "mms",
    language: $("language").value,
    pause_ms: Number($("pause-ms").value) || 500,
    device: $("device").value || null,
    workers: Number($("workers").value) || 1,
  };

  let resp;
  try {
    if (files.length > 0) {
      const form = new FormData();
      files.forEach((file) => form.append("files", file, file.name));
      Object.entries(common).forEach(([key, value]) => {
        if (value !== null) form.append(key, String(value));
      });
      resp = await fetchJson("/api/jobs", { method: "POST", body: form });
    } else if (text) {
      const name = $("doc-name").value.trim();
      if (!name) {
        showFormError(t("formErrorName"));
        return;
      }
      resp = await fetchJson("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, content: text, ...common }),
      });
    } else {
      showFormError(t("formErrorFiles"));
      return;
    }
  } catch (err) {
    showFormError(`${t("sendFailed")}: ${err.message}`);
    return;
  }

  // Limpiar entrada y refrescar
  $("file-input").value = "";
  $("doc-text").value = "";
  $("doc-name").value = "";
  renderFileList();
  if (resp && resp.jobs) jobs = resp.jobs.concat(jobs);
  renderJobs();
  schedulePoll();
}

/* ------------------------------------------------------------------ */
/* Eventos e inicio                                                    */
/* ------------------------------------------------------------------ */

function initDropzone() {
  const zone = $("dropzone");
  const input = $("file-input");

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") input.click();
  });
  input.addEventListener("change", renderFileList);

  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    })
  );
  zone.addEventListener("drop", (event) => {
    input.files = event.dataTransfer.files;
    renderFileList();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  applyLang();
  $("lang-toggle").addEventListener("click", () => {
    lang = lang === "es" ? "en" : "es";
    localStorage.setItem("mdtts-lang", lang);
    applyLang();
    checkBackend();
    renderJobs();
  });

  $("submit").addEventListener("click", submit);

  initDropzone();
  loadEngines();
  checkBackend();
  setInterval(checkBackend, 10000);
  refreshJobs();
});
