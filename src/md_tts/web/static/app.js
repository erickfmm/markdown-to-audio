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
    downloadMp3: "⬇ MP3",
    cancel: "Cancelar",
    cancelJob: "⏹ Cancelar",
    canceling: "Cancelando…",
    canceled: "Cancelado",
    resume: "▶ Reanudar",
    mp3Hint: "Convierte a MP3 (192 kbps) y descarga. La primera vez tarda unos segundos; luego queda en caché.",
    fragments: "fragmentos",
    formErrorFiles: "Selecciona al menos un archivo o pega texto.",
    formErrorName: "Escribe un nombre para el documento pegado.",
    backendUp: "API en línea",
    backendDown: "API sin conexión",
    sendFailed: "No se pudo enviar el trabajo",
    emptyDrop: "Suelta los archivos aquí",
    qwenModel: "Modelo Qwen3",
    qwenSpeaker: "Voz (speaker)",
    qwenInstructStyle: "Instrucción de estilo (opcional)",
    qwenInstructDesign: "Descripción de la voz (obligatorio)",
    qwenInstructHelpStyle: "Cómo leer el texto. Ej.: «Lée todo con un tono muy alegre y a ritmo rápido».",
    qwenInstructHelpDesign: "Describe la voz a crear: género, edad, timbre, emoción, ritmo. Ej.: «Voz masculina joven, tenor, timbre cálido, tono tranquilo de documental».",
    qwenInstructPlaceholderStyle: "Ej.: Habla con alegría y a un ritmo rápido",
    qwenInstructPlaceholderDesign: "Ej.: Voz femenina adulta, timbre cálido, tono sereno de narradora",
    qwenRefAudio: "Audio de referencia para clonar",
    qwenRefAudioHelp: "Sube un mp3/wav (ideal 3–10 s, una sola voz, sin música) o graba tu voz desde el navegador.",
    record: "Grabar",
    stopRecord: "Detener",
    qwenRefText: "Transcripción del audio (recomendado)",
    qwenRefTextHelp: "Qué dice el audio, en texto. Mejora mucho la calidad del clon.",
    qwenRefTextPlaceholder: "Escribe aquí lo que dice el audio…",
    refUploading: "Subiendo audio de referencia…",
    refUploadFailed: "No se pudo subir el audio de referencia",
    micError: "No se pudo acceder al micrófono (requiere localhost o HTTPS)",
    refRequired: "Sube o graba un audio de referencia para clonar la voz.",
    instructRequired: "Escribe una descripción de la voz para el diseño.",
    engineUnavailable: "no disponible",
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
    downloadMp3: "⬇ MP3",
    cancel: "Cancel",
    cancelJob: "⏹ Cancel",
    canceling: "Canceling…",
    canceled: "Canceled",
    resume: "▶ Resume",
    mp3Hint: "Converts to MP3 (192 kbps) and downloads. The first time takes a few seconds; then it's cached.",
    fragments: "fragments",
    formErrorFiles: "Select at least one file or paste some text.",
    formErrorName: "Write a name for the pasted document.",
    backendUp: "API online",
    backendDown: "API offline",
    sendFailed: "Could not submit the job",
    emptyDrop: "Drop files here",
    qwenModel: "Qwen3 model",
    qwenSpeaker: "Voice (speaker)",
    qwenInstructStyle: "Style instruction (optional)",
    qwenInstructDesign: "Voice description (required)",
    qwenInstructHelpStyle: "How to read the text. E.g. 'Read everything in a very happy tone, fast pace.'",
    qwenInstructHelpDesign: "Describe the voice to create: gender, age, timbre, emotion, pace. E.g. 'Young male voice, tenor range, warm timbre, calm documentary tone.'",
    qwenInstructPlaceholderStyle: "E.g. Speak cheerfully at a fast pace",
    qwenInstructPlaceholderDesign: "E.g. Adult female voice, warm timbre, calm narrator tone",
    qwenRefAudio: "Reference audio to clone",
    qwenRefAudioHelp: "Upload an mp3/wav (ideally 3–10 s, a single voice, no music) or record your voice from the browser.",
    record: "Record",
    stopRecord: "Stop",
    qwenRefText: "Audio transcript (recommended)",
    qwenRefTextHelp: "What the audio says, in text. Greatly improves clone quality.",
    qwenRefTextPlaceholder: "Type here what the audio says…",
    refUploading: "Uploading reference audio…",
    refUploadFailed: "Could not upload the reference audio",
    micError: "Could not access the microphone (localhost or HTTPS required)",
    refRequired: "Upload or record a reference audio to clone the voice.",
    instructRequired: "Write a voice description for voice design.",
    engineUnavailable: "not installed",
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
  updateQwenPanel(); // refresca etiquetas dinámicas del panel Qwen
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
  const date = new Date(epoch * 1000);
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (sameDay) return time;
  return date.toLocaleDateString([], { day: "numeric", month: "short" }) + " " + time;
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

let engineMeta = {};

async function loadEngines() {
  try {
    const data = await fetchJson("/api/engines");
    engineMeta = {};
    const select = $("engine");
    select.innerHTML = "";
    (data.engines || []).forEach((engine) => {
      engineMeta[engine.id] = engine;
      const option = document.createElement("option");
      option.value = engine.id;
      option.disabled = engine.available === false;
      let label = engine.id;
      if (!engine.language_aware) label += ` (⚠ ${t("language")})`;
      if (engine.available === false) label += ` — ${t("engineUnavailable")}`;
      option.textContent = label;
      select.appendChild(option);
    });
    // Si el engine seleccionado quedó deshabilitado, salta al primero disponible.
    const selected = select.selectedOptions[0];
    if (selected && selected.disabled) {
      const firstOk = [...select.options].find((o) => !o.disabled);
      if (firstOk) select.value = firstOk.value;
    }
    updateQwenPanel();
  } catch (_) {
    /* la UI de salud ya informa del fallo */
  }
}

/* ------------------------------------------------------------------ */
/* Panel Qwen3 (modelo / speaker / instruct / clonación)               */
/* ------------------------------------------------------------------ */

function qwenMeta() {
  const meta = engineMeta[$("engine").value];
  return meta && meta.variants && meta.variants.length > 0 ? meta : null;
}

function updateQwenPanel() {
  const meta = qwenMeta();
  $("qwen-options").hidden = !meta;
  if (!meta) return;

  // Tamaño del modelo (1.7b / 0.6b según el engine)
  const modelSel = $("qwen-model");
  const prevModel = modelSel.value;
  modelSel.innerHTML = "";
  meta.variants.forEach((variant) => {
    const option = document.createElement("option");
    option.value = variant;
    option.textContent = variant.toUpperCase();
    modelSel.appendChild(option);
  });
  modelSel.value = meta.variants.includes(prevModel)
    ? prevModel
    : meta.default_variant || meta.variants[0];

  // Speaker premium (solo qwen3-customvoice)
  const speakerField = $("qwen-speaker-field");
  const speakerSel = $("qwen-speaker");
  const speakers = meta.speakers || [];
  speakerField.hidden = speakers.length === 0;
  if (speakers.length > 0) {
    const prevSpeaker = speakerSel.value;
    speakerSel.innerHTML = "";
    speakers.forEach((speaker) => {
      const option = document.createElement("option");
      option.value = speaker;
      option.textContent = speaker;
      speakerSel.appendChild(option);
    });
    speakerSel.value = speakers.includes(prevSpeaker)
      ? prevSpeaker
      : meta.default_speaker || speakers[0];
  }

  // Instrucción: estilo (customvoice) o diseño de voz (voicedesign).
  // Solo customvoice 1.7b soporta instruct (0.6b la rechaza en el backend).
  const isDesign = $("engine").value === "qwen3-voicedesign";
  const supportsInstructNow =
    meta.supports_instruct && !(isDesign === false && modelSel.value === "0.6b");
  $("qwen-instruct-field").hidden = !supportsInstructNow;
  if (supportsInstructNow) {
    $("qwen-instruct-label").textContent = isDesign ? t("qwenInstructDesign") : t("qwenInstructStyle");
    $("qwen-instruct-help").textContent = isDesign ? t("qwenInstructHelpDesign") : t("qwenInstructHelpStyle");
    $("qwen-instruct").placeholder = isDesign
      ? t("qwenInstructPlaceholderDesign")
      : t("qwenInstructPlaceholderStyle");
  }

  // Clonación de voz (solo qwen3-clone)
  $("qwen-clone-box").hidden = !meta.supports_voice_clone;
}

/* ------------------------------------------------------------------ */
/* Audio de referencia: subida y grabación desde el navegador          */
/* ------------------------------------------------------------------ */

let voiceRef = null; // { id, duration_s, audio_url, ... }

async function uploadVoiceRef(file) {
  showFormError("");
  const status = $("qwen-ref-status");
  status.hidden = false;
  status.textContent = t("refUploading");
  try {
    const form = new FormData();
    form.append("audio", file, file.name);
    const refText = $("qwen-ref-text").value.trim();
    if (refText) form.append("ref_text", refText);
    const data = await fetchJson("/api/voice-references", { method: "POST", body: form });
    voiceRef = data.voice_reference;
    const preview = $("qwen-ref-preview");
    preview.hidden = false;
    preview.src = voiceRef.audio_url;
    status.textContent = `✓ ${file.name} · ${voiceRef.duration_s}s`;
  } catch (err) {
    voiceRef = null;
    $("qwen-ref-preview").hidden = true;
    status.textContent = "";
    status.hidden = true;
    showFormError(`${t("refUploadFailed")}: ${err.message}`);
  }
}

let mediaRecorder = null;
let recordStream = null;

async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === "undefined") {
    showFormError(t("micError"));
    return;
  }
  try {
    recordStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (_) {
    showFormError(t("micError"));
    return;
  }

  const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
  mediaRecorder = new MediaRecorder(recordStream, mimeType ? { mimeType } : undefined);
  const chunks = [];
  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) chunks.push(event.data);
  };
  mediaRecorder.onstop = () => {
    recordStream.getTracks().forEach((track) => track.stop());
    recordStream = null;
    const btn = $("qwen-record");
    btn.classList.remove("recording");
    btn.innerHTML = `🎤 <span data-i18n="record"></span>`;
    btn.querySelector("[data-i18n]").textContent = t("record");
    const type = mediaRecorder.mimeType || "audio/webm";
    const blob = new Blob(chunks, { type });
    const ext = type.includes("webm") ? "webm" : type.includes("ogg") ? "ogg" : "audio";
    uploadVoiceRef(new File([blob], `grabacion-${Date.now()}.${ext}`, { type }));
  };
  mediaRecorder.start();
  const btn = $("qwen-record");
  btn.classList.add("recording");
  btn.textContent = `⏹ ${t("stopRecord")}`;
}

function collectQwenFields() {
  const meta = qwenMeta();
  if (!meta) return {};
  const instructVisible = !$("qwen-instruct-field").hidden;
  const cloneVisible = !$("qwen-clone-box").hidden;
  const fields = {
    qwen_model: $("qwen-model").value,
    qwen_speaker: $("qwen-speaker-field").hidden ? null : $("qwen-speaker").value,
    qwen_instruct: instructVisible && $("qwen-instruct").value.trim() ? $("qwen-instruct").value.trim() : null,
    voice_ref_id: cloneVisible && voiceRef ? voiceRef.id : null,
    qwen_ref_text: cloneVisible && $("qwen-ref-text").value.trim() ? $("qwen-ref-text").value.trim() : null,
  };
  Object.keys(fields).forEach((key) => {
    if (fields[key] === null || fields[key] === undefined) delete fields[key];
  });
  return fields;
}

function validateQwenFields(qwenFields) {
  const meta = qwenMeta();
  if (!meta) return null;
  if (meta.requires_instruct && !qwenFields.qwen_instruct) return t("instructRequired");
  if (meta.requires_voice_reference && !qwenFields.voice_ref_id) return t("refRequired");
  return null;
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
  // queued/running ya cubren el caso cancel_requested (el job sigue activo
  // hasta que el worker marca canceled); el polling rápido (1.5 s) refresca
  // el badge "Cancelando…" sin demora.
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
  if (job.state === "running" && job.cancel_requested) {
    badge.classList.add("canceling");
    badge.textContent = t("canceling");
  } else {
    badge.textContent = t(
      job.state === "error" ? "errorState" : job.state === "queued" ? "queued" : job.state
    );
  }
  head.appendChild(badge);

  const meta = document.createElement("span");
  meta.className = "job-meta";
  const engineLabel = job.options && job.options.engine ? job.options.engine : "";
  const bits = engineLabel ? [engineLabel, fmtTime(job.created_at)] : [fmtTime(job.created_at)];
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

    if (job.mp3_url) {
      const downloadMp3 = document.createElement("a");
      downloadMp3.className = "btn link";
      downloadMp3.href = job.mp3_url;
      downloadMp3.download = (job.filename || `${job.name}.wav`).replace(/\.wav$/i, ".mp3");
      downloadMp3.textContent = t("downloadMp3");
      downloadMp3.title = t("mp3Hint");
      actions.appendChild(downloadMp3);
    }

    li.appendChild(actions);
  }

  if (job.state !== "running" || job.cancel_requested) {
    const del = document.createElement("button");
    del.type = "button";
    del.className = "btn danger";
    del.textContent = `✕ ${t("delete")}`;
    del.addEventListener("click", () => deleteJob(job.id));
    (li.querySelector(".job-actions") || li).appendChild(del);
  }

  // Cancelar: en cola (efecto inmediato) o en ejecución (cooperativa).
  if (job.state === "queued" || job.state === "running") {
    const actions = document.createElement("div");
    actions.className = "job-actions";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "btn warn";
    cancel.textContent = t("cancelJob");
    cancel.title = t("cancel");
    cancel.disabled = job.cancel_requested === true;
    cancel.addEventListener("click", () => cancelJob(job.id));
    actions.appendChild(cancel);
    li.appendChild(actions);
  }

  // Reanudar: solo jobs cancelados (los fragmentos quedaron en disco).
  if (job.state === "canceled") {
    const actions = document.createElement("div");
    actions.className = "job-actions";
    const resume = document.createElement("button");
    resume.type = "button";
    resume.className = "btn link";
    resume.textContent = t("resume");
    resume.addEventListener("click", () => resumeJob(job.id));
    actions.appendChild(resume);
    li.appendChild(actions);
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

async function cancelJob(id) {
  try {
    await fetchJson(`/api/jobs/${id}/cancel`, { method: "POST" });
  } catch (_) {
    /* ignorado: el polling reconciliará el estado */
  }
  refreshJobs();
}

async function resumeJob(id) {
  try {
    await fetchJson(`/api/jobs/${id}/resume`, { method: "POST" });
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
  const qwenFields = collectQwenFields();
  const qwenError = validateQwenFields(qwenFields);
  if (qwenError) {
    showFormError(qwenError);
    return;
  }
  const common = {
    engine: $("engine").value || "mms",
    language: $("language").value,
    pause_ms: Number($("pause-ms").value) || 500,
    device: $("device").value || null,
    workers: Number($("workers").value) || 1,
    ...qwenFields,
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

  $("engine").addEventListener("change", updateQwenPanel);
  // La visibilidad del instruct depende del tamaño elegido (0.6b no lo soporta)
  $("qwen-model").addEventListener("change", updateQwenPanel);
  $("qwen-ref-file").addEventListener("change", () => {
    const file = ($("qwen-ref-file").files || [])[0];
    if (file) uploadVoiceRef(file);
  });
  $("qwen-record").addEventListener("click", toggleRecording);

  initDropzone();
  loadEngines();
  checkBackend();
  setInterval(checkBackend, 10000);
  refreshJobs();
});
