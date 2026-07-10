const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzoneText");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const statusLine = document.getElementById("status");
const serverBadge = document.getElementById("serverBadge");
const fileRecord = document.getElementById("fileRecord");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const fileType = document.getElementById("fileType");
const resultCard = document.getElementById("resultCard");
const resultTitle = document.getElementById("resultTitle");
const resultSubtitle = document.getElementById("resultSubtitle");
const referenceWrap = document.getElementById("referenceWrap");
const referenceImage = document.getElementById("referenceImage");
const confidenceBlock = document.getElementById("confidenceBlock");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceLabel = document.getElementById("confidenceLabel");
const confidenceMeter = document.getElementById("confidenceMeter");
const emptyRecord = document.getElementById("emptyRecord");
const metaList = document.getElementById("metaList");
const otherMatches = document.getElementById("otherMatches");

// ── Updated endpoint to connect to your FastAPI ──
const analyzeEndpoint =
  window.location.protocol === "file:"
    ? "http://127.0.0.1:8000/identify"
    : "/identify";

let selectedFile = null;
let previewUrl = null;
let isAnalyzing = false;

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function setStatus(message, tone = "neutral") {
  statusLine.textContent = message;
  statusLine.dataset.tone = tone;

  if (tone === "loading") {
    serverBadge.textContent = "Checking";
    serverBadge.dataset.tone = "loading";
  } else if (tone === "error") {
    serverBadge.textContent = "Need image";
    serverBadge.dataset.tone = "error";
  } else if (tone === "success") {
    serverBadge.textContent = "Match found";
    serverBadge.dataset.tone = "success";
  } else {
    serverBadge.textContent = "Waiting";
    serverBadge.dataset.tone = "neutral";
  }
}

function setBusy(nextValue) {
  isAnalyzing = nextValue;
  analyzeBtn.disabled = isAnalyzing || !selectedFile;
  clearBtn.disabled = isAnalyzing || !selectedFile;
  analyzeBtn.classList.toggle("is-loading", isAnalyzing);
}

function resetResultPanel() {
  resultCard.classList.add("result-panel-empty");
  resultCard.classList.remove("has-reference");
  resultTitle.textContent = "No specimen loaded";
  resultSubtitle.textContent = "Run analysis to create a plant record.";
  referenceWrap.hidden = true;
  confidenceBlock.hidden = true;
  emptyRecord.hidden = false;
  confidenceMeter.style.width = "0%";
  metaList.replaceChildren();
  otherMatches.replaceChildren();
}

function showLoadingResult() {
  resultCard.classList.remove("result-panel-empty");
  resultCard.classList.remove("has-reference");
  resultTitle.textContent = "Analyzing";
  resultSubtitle.textContent = "Identifying plant and retrieving medicinal information...";
  referenceWrap.hidden = true;
  confidenceBlock.hidden = true;
  emptyRecord.hidden = true;
  metaList.replaceChildren();
  otherMatches.replaceChildren();
}

function clearSelection() {
  selectedFile = null;
  imageInput.value = "";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  preview.removeAttribute("src");
  preview.hidden = true;
  dropzoneText.hidden = false;
  fileRecord.hidden = true;
  fileName.textContent = "-";
  fileSize.textContent = "-";
  fileType.textContent = "-";
  setBusy(false);
  resetResultPanel();
  setStatus("Insert an image to begin.");
}

function chooseFile(file) {
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    clearSelection();
    setStatus("Please choose an image file.", "error");
    return;
  }

  selectedFile = file;

  if (previewUrl) URL.revokeObjectURL(previewUrl);

  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  dropzoneText.hidden = true;
  fileRecord.hidden = false;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileType.textContent = file.type || "image";
  setBusy(false);
  setStatus(`Image ready: ${file.name}`);
}

function createMetaItem(label, value) {
  const item = document.createElement("div");
  item.className = "meta-item";

  const title = document.createElement("strong");
  title.textContent = label;

  const body = document.createElement("div");
  body.textContent = value;

  item.append(title, body);
  return item;
}

function createLinkItem(label, url, text) {
  let parsedUrl;
  try {
    parsedUrl = new URL(url);
  } catch {
    return null;
  }

  const item = document.createElement("div");
  item.className = "meta-item";

  const title = document.createElement("strong");
  title.textContent = label;

  const body = document.createElement("div");
  const link = document.createElement("a");
  link.className = "meta-link";
  link.href = parsedUrl.href;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = text;

  body.append(link);
  item.append(title, body);
  return item;
}

// ── Map your FastAPI response to the existing UI ──────────
function mapApiResponseToUI(data) {
  const report = data.report || {};
  const confidencePercent = Math.round((data.confidence || 0) * 100);

  return {
    best_match: {
      name: report.plant_name || data.ml_prediction || "Unknown",
      confidence: confidencePercent,
      confidence_label: confidencePercent >= 75 ? "high" : confidencePercent >= 50 ? "moderate" : "low",
      metadata: {
        scientific_name: report.latin_name || "",
        common_name: report.plant_name || "",
        nepali_name: report.nepali_name || "",
        medicinal_uses: report.medicinal_uses || "",
        traditional_use: report.traditional_use || "",
        safety: report.safety || "",
        safety_note: report.safety_note || "",
        location_in_nepal: report.location_in_nepal || "",
        source: report.source || (data.report_source === "knowledge_base" ? "Wikipedia via knowledge base" : "General knowledge"),
        observation_url: null,
        image_url: null,
      },
    },
    other_matches: [],
  };
}

function renderMetadata(metadata) {
  const items = [];

  const fields = [
    { label: "Scientific name", value: metadata.scientific_name },
    { label: "Common name", value: metadata.common_name },
    { label: "Nepali name", value: metadata.nepali_name },
    { label: "Medicinal uses", value: metadata.medicinal_uses },
    { label: "Traditional use in Nepal", value: metadata.traditional_use },
    { label: "Safety", value: metadata.safety },
    { label: "Safety note", value: metadata.safety_note },
    { label: "Found in Nepal", value: metadata.location_in_nepal },
    { label: "Source", value: metadata.source },
  ];

  fields.forEach(({ label, value }) => {
    if (value && value.trim() !== "") {
      items.push(createMetaItem(label, value));
    }
  });

  if (metadata.observation_url) {
    const linkItem = createLinkItem("Observation", metadata.observation_url, "Open observation");
    if (linkItem) items.push(linkItem);
  }

  if (metadata.image_url) {
    const linkItem = createLinkItem("Reference image", metadata.image_url, "Open image");
    if (linkItem) items.push(linkItem);
  }

  metaList.replaceChildren(
    ...(items.length > 0
      ? items
      : [createMetaItem("Info", "No additional metadata was returned for this match.")])
  );
}

function renderOtherMatches(matches) {
  otherMatches.replaceChildren();

  if (!Array.isArray(matches) || matches.length === 0) return;

  const heading = document.createElement("h3");
  heading.textContent = "Other possible matches";
  otherMatches.append(heading);

  matches.forEach((match) => {
    const row = document.createElement("div");
    row.className = "match-row";

    const rank = document.createElement("span");
    rank.className = "match-rank";
    rank.textContent = match.rank ? String(match.rank) : "-";

    const name = document.createElement("span");
    name.className = "match-name";
    name.textContent = match.name || "Unknown match";

    const score = document.createElement("span");
    score.className = "match-score";
    score.textContent =
      typeof match.confidence === "number"
        ? `${match.confidence}% ${match.confidence_label || ""}`.trim()
        : "n/a";

    row.append(rank, name, score);
    otherMatches.append(row);
  });
}

function renderResult(data) {
  // ── Map your FastAPI response to UI format ──
  const mapped = mapApiResponseToUI(data);

  const bestMatch = mapped.best_match || {};
  const metadata = bestMatch.metadata || {};
  const confidence = Number(bestMatch.confidence);
  const hasConfidence = Number.isFinite(confidence);
  const confidencePercent = hasConfidence ? Math.max(0, Math.min(100, confidence)) : 0;

  resultCard.classList.remove("result-panel-empty");
  resultTitle.textContent = bestMatch.name || "No match found";
  resultSubtitle.textContent = "Best match from Nepal medicinal plant knowledge base.";

  referenceWrap.hidden = true;
  referenceImage.removeAttribute("src");
  resultCard.classList.remove("has-reference");

  confidenceBlock.hidden = false;
  emptyRecord.hidden = true;
  confidenceValue.textContent = hasConfidence ? `Confidence: ${confidencePercent.toFixed(1)}%` : "Confidence: n/a";
  confidenceLabel.textContent = bestMatch.confidence_label || "unknown";
  confidenceMeter.style.width = `${confidencePercent}%`;

  renderMetadata(metadata);
  renderOtherMatches(mapped.other_matches);
}

// ── Main analyze function ─────────────────────────────────
async function analyzeSelectedFile() {
  if (!selectedFile || isAnalyzing) return;

  const formData = new FormData();
  // ── Changed key from "image" to "file" to match your FastAPI ──
  formData.append("file", selectedFile);

  setBusy(true);
  showLoadingResult();
  setStatus("Analyzing image...", "loading");

  try {
    const response = await fetch(analyzeEndpoint, {
      method: "POST",
      body: formData,
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : { error: await response.text() };

    if (!response.ok) {
      throw new Error(payload.detail || payload.error || "Analysis failed.");
    }

    // ── Handle low confidence response ──
    if (payload.success === false) {
      resultCard.classList.remove("result-panel-empty");
      resultTitle.textContent = "Plant not identified";
      resultSubtitle.textContent = payload.message || "Please try a clearer photo.";
      confidenceBlock.hidden = true;
      emptyRecord.hidden = true;
      metaList.replaceChildren(createMetaItem("Tip", "Try a clearer, well-lit photo of the plant."));
      otherMatches.replaceChildren();
      setStatus(payload.message || "Could not identify plant.", "error");
      return;
    }

    renderResult(payload);
    setStatus("Analysis complete.", "success");
  } catch (error) {
    resultCard.classList.remove("result-panel-empty");
    resultTitle.textContent = "Analysis failed";
    resultSubtitle.textContent = "The image could not be processed.";
    referenceWrap.hidden = true;
    resultCard.classList.remove("has-reference");
    confidenceBlock.hidden = true;
    emptyRecord.hidden = true;
    metaList.replaceChildren(createMetaItem("Error", error.message));
    otherMatches.replaceChildren();
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

// ── Event listeners ───────────────────────────────────────
imageInput.addEventListener("change", (event) => {
  chooseFile(event.target.files?.[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "dragend", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, () => {
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  chooseFile(event.dataTransfer.files?.[0]);
});

analyzeBtn.addEventListener("click", analyzeSelectedFile);
clearBtn.addEventListener("click", clearSelection);
referenceImage.addEventListener("error", () => {
  referenceWrap.hidden = true;
  resultCard.classList.remove("has-reference");
});