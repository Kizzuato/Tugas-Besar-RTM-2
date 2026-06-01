const canvases = {
  main: document.getElementById("mainCanvas"),
  original: document.getElementById("originalCanvas"),
  normalized: document.getElementById("normalizedCanvas"),
  gray: document.getElementById("grayCanvas"),
  threshold: document.getElementById("thresholdCanvas"),
  edge: document.getElementById("edgeCanvas"),
  crop: document.getElementById("cropCanvas")
};

const els = {
  imageInput: document.getElementById("imageInput"),
  sampleSelect: document.getElementById("sampleSelect"),
  runButton: document.getElementById("runButton"),
  extractButton: document.getElementById("extractButton"),
  resetButton: document.getElementById("resetButton"),
  classifyButton: document.getElementById("classifyButton"),
  regionInput: document.getElementById("regionInput"),
  numberInput: document.getElementById("numberInput"),
  emptyState: document.getElementById("emptyState"),
  appStatus: document.getElementById("appStatus"),
  plateClass: document.getElementById("plateClass"),
  dominantColor: document.getElementById("dominantColor"),
  plateArea: document.getElementById("plateArea"),
  otsuValue: document.getElementById("otsuValue"),
  vehicleType: document.getElementById("vehicleType"),
  wheelType: document.getElementById("wheelType"),
  ruleUsed: document.getElementById("ruleUsed"),
  detectedText: document.getElementById("detectedText"),
  detectedLetters: document.getElementById("detectedLetters"),
  detectedDigits: document.getElementById("detectedDigits"),
  characterCount: document.getElementById("characterCount"),
  characterStrip: document.getElementById("characterStrip"),
  charStatus: document.getElementById("charStatus")
};

let selectedFile = null;
let selectedSample = "";

els.imageInput.addEventListener("change", event => {
  selectedFile = event.target.files[0] || null;
  selectedSample = "";
  els.sampleSelect.value = "";
  if (!selectedFile) return;
  setStatus(`Citra dipilih: ${selectedFile.name}`);
  resetOutputs(false);
});

els.sampleSelect.addEventListener("change", event => {
  selectedSample = event.target.value;
  selectedFile = null;
  els.imageInput.value = "";
  if (!selectedSample) return;
  setStatus(`Contoh dipilih: ${event.target.options[event.target.selectedIndex].text}`);
  resetOutputs(false);
});

els.runButton.addEventListener("click", () => processImage("classification"));
els.extractButton.addEventListener("click", () => processImage("extraction"));
els.resetButton.addEventListener("click", resetInterface);
els.classifyButton.addEventListener("click", classifyRegistration);

async function processImage(flow) {
  if (!selectedFile && !selectedSample) {
    setStatus("Pilih gambar terlebih dahulu");
    return;
  }

  setStatus("Memproses citra di backend Python...");
  setWorkflow(flow);

  const formData = new FormData();
  if (selectedFile) formData.append("image", selectedFile);
  if (selectedSample) formData.append("sample", selectedSample);
  formData.append("flow", flow);

  try {
    const response = await fetch("/api/process", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok || !payload.success) throw new Error(payload.message || "Pemrosesan gagal");
    renderResult(payload, flow);
    setStatus(flow === "extraction" ? "Alur ekstraksi selesai dari Python" : "Alur klasifikasi selesai dari Python");
  } catch (error) {
    setStatus(error.message);
  }
}

function renderResult(payload, flow) {
  els.emptyState.style.display = "none";
  drawImage(canvases.main, payload.images.main);
  drawImage(canvases.original, payload.images.original);
  drawImage(canvases.normalized, payload.images.normalized);
  drawImage(canvases.gray, payload.images.gray);
  drawImage(canvases.threshold, payload.images.threshold);
  drawImage(canvases.edge, payload.images.edge);
  drawImage(canvases.crop, payload.images.crop);

  els.plateClass.textContent = payload.analysis.plate_class || "-";
  els.dominantColor.textContent = payload.analysis.dominant_color || "-";
  els.plateArea.textContent = payload.analysis.plate_area || "-";
  els.otsuValue.textContent = payload.analysis.otsu_value ?? "-";

  renderRecognition(payload.recognition);
  renderClassification(payload.classification);
  setActiveStage(flow === "extraction" ? "crop" : "threshold");
}

function drawImage(canvas, src) {
  const ctx = canvas.getContext("2d");
  if (!src) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const image = new Image();
  image.onload = () => {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    ctx.drawImage(image, 0, 0);
  };
  image.src = src;
}

function renderRecognition(recognition) {
  els.detectedText.textContent = recognition.detected_text || "-";
  els.detectedLetters.textContent = recognition.detected_letters || "-";
  els.detectedDigits.textContent = recognition.detected_digits || "-";
  els.characterCount.textContent = recognition.character_count || "-";
  els.charStatus.textContent = recognition.character_count ? `${recognition.character_count} karakter` : "0 karakter";
  els.characterStrip.innerHTML = "";

  if (!recognition.characters || recognition.characters.length === 0) {
    els.characterStrip.innerHTML = "<div class=\"strip-empty\">Tidak ada kandidat karakter yang lolos filter Python.</div>";
    return;
  }

  recognition.characters.forEach(item => {
    const tile = document.createElement("div");
    tile.className = "char-tile";
    const image = document.createElement("img");
    image.src = item.image;
    image.alt = `Karakter ${item.character || "?"}`;
    const meta = document.createElement("div");
    meta.className = "char-meta";
    const score = typeof item.score === "number" ? item.score.toFixed(3) : "-";
    meta.innerHTML = `<strong>${item.character || "?"}</strong><span>${item.type || "karakter"} ${score}</span>`;
    tile.append(image, meta);
    els.characterStrip.appendChild(tile);
  });
}

function renderClassification(classification) {
  els.vehicleType.textContent = classification.vehicle_type || "-";
  els.wheelType.textContent = classification.wheel_category || "-";
  els.ruleUsed.textContent = classification.rule_used || "-";
}

async function classifyRegistration() {
  const region = els.regionInput.value;
  const number = els.numberInput.value;
  
  if (!number) {
    els.vehicleType.textContent = "Nomor belum diisi";
    els.wheelType.textContent = "-";
    els.ruleUsed.textContent = "-";
    return;
  }

  try {
    const response = await fetch("/api/classify-registration", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ region, number })
    });
    const payload = await response.json();
    if (!response.ok || !payload.success) throw new Error(payload.message || "Klasifikasi nomor gagal");
    
    // Update inputs with sanitized values from Python
    if (payload.sanitized) {
      els.regionInput.value = payload.sanitized.region;
      els.numberInput.value = payload.sanitized.number;
    }
    
    renderClassification(payload.classification);
  } catch (error) {
    setStatus(error.message);
  }
}


function setActiveStage(stage) {
  document.querySelectorAll(".stage-card").forEach(card => {
    card.classList.toggle("active", card.dataset.stage === stage);
  });
}

function setWorkflow(flow) {
  document.querySelectorAll(".workflow-step").forEach(step => {
    step.classList.toggle("active", step.dataset.flow === flow);
  });
}

function setStatus(text) {
  els.appStatus.textContent = text;
}

function resetOutputs(clearSelection = true) {
  if (clearSelection) {
    selectedFile = null;
    selectedSample = "";
    els.imageInput.value = "";
    els.sampleSelect.value = "";
  }
  els.emptyState.style.display = "grid";
  Object.values(canvases).forEach(canvas => canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height));
  els.plateClass.textContent = "Belum ada citra";
  els.dominantColor.textContent = "-";
  els.plateArea.textContent = "-";
  els.otsuValue.textContent = "-";
  els.detectedText.textContent = "-";
  els.detectedLetters.textContent = "-";
  els.detectedDigits.textContent = "-";
  els.characterCount.textContent = "-";
  els.charStatus.textContent = "Menunggu crop plat";
  els.characterStrip.innerHTML = "<div class=\"strip-empty\">Karakter hasil segmentasi akan muncul di sini.</div>";
  setActiveStage("original");
  setWorkflow("classification");
}

function resetInterface() {
  resetOutputs(true);
  els.vehicleType.textContent = "-";
  els.wheelType.textContent = "-";
  els.ruleUsed.textContent = "-";
  setStatus("Siap memproses citra");
}
