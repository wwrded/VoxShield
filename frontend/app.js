/*
 * VoxShield dashboard.
 *
 * Talks to the REST API, renders the assessment, and - importantly - surfaces
 * the model's provenance. If the loaded checkpoint has not been trained on a
 * real speech corpus the interface says so before it shows a single score.
 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const el = {
    statusDot: $("statusDot"), statusText: $("statusText"),
    modelBanner: $("modelBanner"), modelBannerTitle: $("modelBannerTitle"),
    modelBannerText: $("modelBannerText"),
    profileSelect: $("profileSelect"), profileHint: $("profileHint"),
    presetButtons: $("presetButtons"), presetNote: $("presetNote"),
    dropZone: $("dropZone"), fileInput: $("audioFileInput"), limitsText: $("limitsText"),
    micBtn: $("micBtn"), micBtnText: $("micBtnText"), micTimer: $("micTimer"),
    playbackCard: $("playbackCard"), fileName: $("currentFileName"),
    audio: $("audioElement"), waveform: $("waveformCanvas"),
    analyzeBtn: $("analyzeBtn"), analyzeBtnText: $("analyzeBtnText"),
    errorText: $("errorText"),
    emptyState: $("emptyState"), results: $("resultsContent"),
    latency: $("latencyBadge"),
    verdictBanner: $("verdictBanner"), verdictTitle: $("verdictTitle"),
    verdictSubtitle: $("verdictSubtitle"), actionTag: $("actionTag"),
    riskValue: $("riskValue"), riskTier: $("riskTier"), riskBarFill: $("riskBarFill"),
    thresholdValue: $("thresholdValue"), marginValue: $("marginValue"),
    scorerValue: $("scorerValue"),
    durationValue: $("durationValue"), windowsValue: $("windowsValue"),
    peakWindowValue: $("peakWindowValue"), windowCanvas: $("windowCanvas"),
    recHeader: $("recHeader"), recBody: $("recBody"),
    spectrogram: $("spectrogramImg"),
    ciaC: $("ciaC"), ciaI: $("ciaI"), ciaA: $("ciaA"),
    evidenceList: $("evidenceList"), caveatBox: $("caveatBox"),
    feedbackCard: $("feedbackCard"), feedbackConsent: $("feedbackConsent"),
    feedbackToken: $("feedbackToken"), feedbackGenuine: $("feedbackGenuine"),
    feedbackSynthetic: $("feedbackSynthetic"), feedbackStatus: $("feedbackStatus"),
  };

  let feedbackConfig = { enabled: false, token_required: false };
  let lastScore = null;          // model score of the result on screen
  let lastAnalysed = null;       // {file} or {sampleId} that produced it

  const RECORD_SECONDS = 5;

  let selectedFile = null;      // File or Blob chosen for upload
  let selectedSampleId = null;  // bundled sample id, mutually exclusive
  let profiles = [];
  let recording = false;

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function showError(message) {
    el.errorText.textContent = message;
    el.errorText.hidden = false;
  }

  function clearError() {
    el.errorText.hidden = true;
    el.errorText.textContent = "";
  }

  function setBusy(busy) {
    el.analyzeBtn.disabled = busy || (!selectedFile && !selectedSampleId);
    el.analyzeBtnText.textContent = busy ? "Analysing…" : "Analyse recording";
  }

  async function request(url, options) {
    const response = await fetch(url, options);
    let body = null;
    try { body = await response.json(); } catch (_) { /* non-JSON error page */ }
    if (!response.ok) {
      throw new Error((body && body.detail) || `Request failed (${response.status})`);
    }
    return body;
  }

  // ------------------------------------------------------------------
  // Start-up
  // ------------------------------------------------------------------
  async function loadHealth() {
    try {
      const health = await request("/api/health");
      const model = health.model;

      el.statusDot.classList.add("online");
      el.statusText.textContent = model.weights_loaded
        ? `CNN loaded · ${model.parameters.toLocaleString()} parameters`
        : "No checkpoint · acoustic baseline only";

      const mb = Math.round(health.limits.max_upload_bytes / (1024 * 1024));
      el.limitsText.textContent =
        `${health.limits.allowed_extensions.join(", ")} · up to ${mb} MB · ` +
        `${health.limits.max_audio_seconds}s maximum`;

      // The provenance banner is the honest-reporting requirement made visible.
      if (!model.trustworthy) {
        el.modelBanner.hidden = false;
        el.modelBannerTitle.textContent =
          model.status === "DEMO_SYNTHETIC"
            ? "Demonstration model — not trained on speech"
            : "No validated model loaded";
        el.modelBannerText.textContent = model.caveats.join(" ");
      }
    } catch (err) {
      el.statusText.textContent = "API unreachable";
      showError(err.message);
    }
  }

  async function loadProfiles() {
    const data = await request("/api/profiles");
    profiles = data.profiles;
    el.profileSelect.innerHTML = "";
    profiles.forEach((profile) => {
      const option = document.createElement("option");
      option.value = profile.key;
      option.textContent = profile.display_name;
      el.profileSelect.appendChild(option);
    });
    updateProfileHint();
  }

  function updateProfileHint() {
    const profile = profiles.find((p) => p.key === el.profileSelect.value);
    el.profileHint.textContent = profile ? profile.description : "";
  }

  async function loadSamples() {
    const data = await request("/api/samples");
    el.presetNote.textContent = data.note || "";
    el.presetButtons.innerHTML = "";

    data.samples.forEach((sample) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "preset-btn";
      const dot = document.createElement("span");
      dot.className = "preset-indicator";
      const label = document.createElement("span");
      label.textContent = sample.name;
      button.append(dot, label);
      button.addEventListener("click", () => selectSample(sample, button));
      el.presetButtons.appendChild(button);
    });
  }

  // ------------------------------------------------------------------
  // Input selection
  // ------------------------------------------------------------------
  function selectSample(sample, button) {
    clearError();
    selectedSampleId = sample.id;
    selectedFile = null;

    document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");

    el.fileName.textContent = sample.name;
    // Playback comes from the dedicated GET route. v1 pointed this at the
    // POST-only prediction endpoint, so the player always failed.
    el.audio.src = `/api/samples/${encodeURIComponent(sample.id)}/audio`;
    el.playbackCard.style.display = "";
    drawWaveformFromUrl(el.audio.src);
    setBusy(false);
  }

  function selectFile(file) {
    clearError();
    selectedFile = file;
    selectedSampleId = null;
    document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));

    el.fileName.textContent = file.name || "recording.wav";
    const url = URL.createObjectURL(file);
    el.audio.src = url;
    el.playbackCard.style.display = "";
    drawWaveformFromUrl(url);
    setBusy(false);
  }

  el.fileInput.addEventListener("change", (event) => {
    if (event.target.files && event.target.files[0]) selectFile(event.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((type) =>
    el.dropZone.addEventListener(type, (e) => {
      e.preventDefault();
      el.dropZone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    el.dropZone.addEventListener(type, (e) => {
      e.preventDefault();
      el.dropZone.classList.remove("dragover");
    })
  );
  el.dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) selectFile(file);
  });

  // ------------------------------------------------------------------
  // Waveform preview
  // ------------------------------------------------------------------
  async function drawWaveformFromUrl(url) {
    const canvas = el.waveform;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    try {
      const buffer = await (await fetch(url)).arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const decoded = await audioCtx.decodeAudioData(buffer);
      const data = decoded.getChannelData(0);
      audioCtx.close();

      const step = Math.max(1, Math.floor(data.length / canvas.width));
      const mid = canvas.height / 2;
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < canvas.width; x += 1) {
        let peak = 0;
        for (let i = 0; i < step; i += 1) {
          const value = Math.abs(data[x * step + i] || 0);
          if (value > peak) peak = value;
        }
        ctx.moveTo(x, mid - peak * mid);
        ctx.lineTo(x, mid + peak * mid);
      }
      ctx.stroke();
    } catch (_) {
      /* Preview only; a decode failure here must not block analysis. */
    }
  }

  // ------------------------------------------------------------------
  // Microphone capture
  // ------------------------------------------------------------------
  function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (offset, text) => {
      for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
    };
    writeString(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);          // PCM
    view.setUint16(22, 1, true);          // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i += 1, offset += 2) {
      const clamped = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    }
    return new Blob([view], { type: "audio/wav" });
  }

  el.micBtn.addEventListener("click", async () => {
    if (recording) return;
    clearError();

    let stream;
    try {
      // Browsers apply noise suppression, echo cancellation and automatic gain by
      // default. Noise suppression re-synthesises the voice, and the processing
      // traces it leaves can look like a vocoder to the detector, so all three
      // are turned off: the model should hear the microphone, not the filter.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (err) {
      showError("Microphone access was refused.");
      return;
    }

    recording = true;
    el.micBtn.classList.add("recording");

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    const chunks = [];

    processor.onaudioprocess = (event) => {
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    source.connect(processor);
    processor.connect(audioCtx.destination);

    let remaining = RECORD_SECONDS;
    el.micTimer.textContent = `${remaining}s`;
    const ticker = setInterval(() => {
      remaining -= 1;
      el.micTimer.textContent = remaining > 0 ? `${remaining}s` : "";
    }, 1000);

    setTimeout(() => {
      clearInterval(ticker);
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((track) => track.stop());

      const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const merged = new Float32Array(total);
      let offset = 0;
      chunks.forEach((chunk) => { merged.set(chunk, offset); offset += chunk.length; });

      const blob = encodeWav(merged, audioCtx.sampleRate);
      audioCtx.close();

      const file = new File([blob], "microphone_capture.wav", { type: "audio/wav" });
      selectFile(file);

      recording = false;
      el.micBtn.classList.remove("recording");
      el.micTimer.textContent = "";
    }, RECORD_SECONDS * 1000);
  });

  // ------------------------------------------------------------------
  // Analysis
  // ------------------------------------------------------------------
  el.analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile && !selectedSampleId) return;
    clearError();
    setBusy(true);

    try {
      const form = new FormData();
      form.append("profile", el.profileSelect.value);

      let result;
      if (selectedSampleId) {
        result = await request(
          `/api/samples/${encodeURIComponent(selectedSampleId)}/analyze`,
          { method: "POST", body: form }
        );
      } else {
        form.append("file", selectedFile, selectedFile.name || "recording.wav");
        result = await request("/api/analyze", { method: "POST", body: form });
      }
      lastAnalysed = selectedSampleId ? { sampleId: selectedSampleId } : { file: selectedFile };
      render(result);
    } catch (err) {
      showError(err.message);
    } finally {
      setBusy(false);
    }
  });

  function render(result) {
    const risk = result.risk_assessment;

    el.emptyState.style.display = "none";
    el.results.style.display = "";
    el.latency.textContent = `Latency: ${result.latency_ms} ms`;

    el.verdictTitle.textContent = risk.verdict;
    el.verdictSubtitle.textContent =
      `${risk.risk_tier} risk · ${risk.profile_display_name}`;
    el.verdictBanner.style.borderColor = risk.color;
    el.actionTag.textContent = risk.action_code.replace(/_/g, " ");
    el.actionTag.style.background = risk.color;

    el.riskValue.textContent = `${risk.risk_score}%`;
    el.riskValue.style.color = risk.color;
    el.riskTier.textContent = risk.classification_label.replace(/_/g, " ");
    el.riskBarFill.style.width = `${risk.risk_score}%`;
    el.riskBarFill.style.background = risk.color;

    el.thresholdValue.textContent = risk.decision_threshold.toFixed(3);
    el.marginValue.textContent = `${risk.margin}%`;
    el.scorerValue.textContent =
      result.scorer === "cnn" ? "CNN classifier" : "Acoustic baseline";

    el.durationValue.textContent = `${result.audio.duration_seconds}s`;
    el.windowsValue.textContent = result.audio.windows_scored;

    const windows = result.audio.per_window_probabilities || [];
    if (windows.length) {
      const peak = Math.max.apply(null, windows);
      el.peakWindowValue.textContent = `${(peak * 100).toFixed(1)}%`;
    } else {
      el.peakWindowValue.textContent = "--";
    }
    drawWindowScores(windows, risk.decision_threshold, risk.color);

    el.recHeader.textContent = `Recommended action · ${risk.profile_display_name}`;
    el.recBody.textContent = risk.recommendation;

    el.spectrogram.src = result.spectrogram_image;

    el.ciaC.textContent = risk.cia_impact.confidentiality;
    el.ciaI.textContent = risk.cia_impact.integrity;
    el.ciaA.textContent = risk.cia_impact.availability;

    el.evidenceList.innerHTML = "";
    Object.entries(result.risk_assessment.evidence || {}).forEach(([key, value]) => {
      const row = document.createElement("div");
      row.className = "stat-row";
      const name = document.createElement("span");
      name.className = "stat-name";
      name.textContent = key.replace(/_/g, " ");
      const val = document.createElement("span");
      val.className = "stat-value";
      val.textContent = value;
      row.append(name, val);
      el.evidenceList.appendChild(row);
    });

    const caveats = risk.caveats || [];
    if (caveats.length) {
      el.caveatBox.hidden = false;
      el.caveatBox.textContent = caveats.join(" ");
    } else {
      el.caveatBox.hidden = true;
    }

    lastScore = risk.spoof_probability;
    resetFeedback();
  }

  // ------------------------------------------------------------------
  // Operator feedback
  // ------------------------------------------------------------------
  async function loadFeedbackConfig() {
    try {
      feedbackConfig = await request("/api/feedback/summary");
    } catch (_) {
      feedbackConfig = { enabled: false };
    }
    el.feedbackToken.hidden = !feedbackConfig.token_required;
  }

  function resetFeedback() {
    el.feedbackCard.hidden = !feedbackConfig.enabled;
    el.feedbackConsent.checked = false;
    el.feedbackStatus.textContent = "";
    el.feedbackStatus.className = "feedback-status";
    el.feedbackGenuine.disabled = false;
    el.feedbackSynthetic.disabled = false;
  }

  async function sendFeedback(label) {
    if (!lastAnalysed) return;
    if (!el.feedbackConsent.checked) {
      el.feedbackStatus.textContent = "Tick the consent box first. Without consent the recording is not kept.";
      el.feedbackStatus.className = "feedback-status err";
      return;
    }
    el.feedbackGenuine.disabled = true;
    el.feedbackSynthetic.disabled = true;
    el.feedbackStatus.className = "feedback-status";
    el.feedbackStatus.textContent = "Sending…";
    try {
      let file = lastAnalysed.file;
      let name = file && file.name ? file.name : "recording.wav";
      if (lastAnalysed.sampleId) {
        const audio = await fetch(`/api/samples/${encodeURIComponent(lastAnalysed.sampleId)}/audio`);
        file = await audio.blob();
        name = lastAnalysed.sampleId;
      }
      const form = new FormData();
      form.append("file", file, name);
      form.append("label", label);
      form.append("consent", "true");
      if (lastScore !== null) form.append("model_score", String(lastScore));
      const headers = {};
      if (feedbackConfig.token_required) headers["X-Feedback-Token"] = el.feedbackToken.value;
      const body = await request("/api/feedback", { method: "POST", body: form, headers });
      el.feedbackStatus.textContent =
        `Saved as ${label === "genuine" ? "a real voice" : "synthetic"}. ${body.pending} correction(s) waiting for the next fine-tune.`;
      el.feedbackStatus.className = "feedback-status ok";
    } catch (err) {
      el.feedbackStatus.textContent = err.message;
      el.feedbackStatus.className = "feedback-status err";
      el.feedbackGenuine.disabled = false;
      el.feedbackSynthetic.disabled = false;
    }
  }

  el.feedbackGenuine.addEventListener("click", () => sendFeedback("genuine"));
  el.feedbackSynthetic.addEventListener("click", () => sendFeedback("synthetic"));

  function drawWindowScores(scores, threshold, color) {
    const canvas = el.windowCanvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!scores.length) return;

    const barWidth = canvas.width / scores.length;
    scores.forEach((score, index) => {
      const height = Math.max(2, score * (canvas.height - 8));
      ctx.fillStyle = score >= threshold ? color : "#334155";
      ctx.fillRect(index * barWidth + 1, canvas.height - height, Math.max(1, barWidth - 2), height);
    });

    const y = canvas.height - threshold * (canvas.height - 8);
    ctx.strokeStyle = "#94a3b8";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  el.profileSelect.addEventListener("change", updateProfileHint);

  // ------------------------------------------------------------------
  (async function init() {
    await loadHealth();
    await loadFeedbackConfig();
    try {
      await loadProfiles();
      await loadSamples();
    } catch (err) {
      showError(err.message);
    }
  })();
})();
