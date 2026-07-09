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

const analyzeEndpoint =
  window.location.protocol === "file:" ? "http://127.0.0.1:8000/analyze" : "analyze";

let selectedFile = null;
let previewUrl = null;
let isAnalyzing = false;

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "-";
  }

  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;

  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function setStatus(message, tone = "neutral") {
  statusLine.textContent = message;
  statusLine.dataset.tone = tone;

  if (tone === "loading") {
    serverBadge.textContent = "Analyzing";
    serverBadge.dataset.tone = "loading";
  } else if (tone === "error") {
    serverBadge.textContent = "Check";
    serverBadge.dataset.tone = "error";
  } else {
    serverBadge.textContent = "Ready";
    serverBadge.dataset.tone = tone === "success" ? "success" : "neutral";
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
  resultSubtitle.textContent = "Checking the local plant index.";
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
  setStatus("No image selected.");
}

function chooseFile(file) {
  if (!file) {
    return;
  }

  if (!file.type.startsWith("image/")) {
    clearSelection();
    setStatus("Please choose an image file.", "error");
    return;
  }

  selectedFile = file;

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  dropzoneText.hidden = true;
  fileRecord.hidden = false;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileType.textContent = file.type || "image";
  setBusy(false);
  setStatus(`Selected: ${file.name}`);
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

function renderMetadata(metadata) {
  const items = [];

  if (metadata.scientific_name) {
    items.push(createMetaItem("Scientific name", metadata.scientific_name));
  }

  if (metadata.common_name) {
    items.push(createMetaItem("Common name", metadata.common_name));
  }

  if (metadata.description) {
    items.push(createMetaItem("Description", metadata.description));
  }

  if (Array.isArray(metadata.taxonomy) && metadata.taxonomy.length > 0) {
    const taxonomy = metadata.taxonomy
      .filter((item) => item && item.label && item.value)
      .map((item) => `${item.label}: ${item.value}`)
      .join(" > ");

    if (taxonomy) {
      items.push(createMetaItem("Taxonomy", taxonomy));
    }
  }

  if (metadata.observation_url) {
    const linkItem = createLinkItem("Observation", metadata.observation_url, "Open observation");
    if (linkItem) {
      items.push(linkItem);
    }
  }

  if (metadata.image_url) {
    const linkItem = createLinkItem("Reference image", metadata.image_url, "Open image");
    if (linkItem) {
      items.push(linkItem);
    }
  }

  metaList.replaceChildren(
    ...(items.length > 0
      ? items
      : [createMetaItem("Info", "No additional metadata was returned for this match.")])
  );
}

function renderOtherMatches(matches) {
  otherMatches.replaceChildren();

  if (!Array.isArray(matches) || matches.length === 0) {
    return;
  }

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
  const bestMatch = data.best_match || {};
  const metadata = bestMatch.metadata || {};
  const confidence = Number(bestMatch.confidence);
  const hasConfidence = Number.isFinite(confidence);
  const confidencePercent = hasConfidence ? Math.max(0, Math.min(100, confidence)) : 0;

  resultCard.classList.remove("result-panel-empty");
  resultTitle.textContent = bestMatch.name || "No match found";
  resultSubtitle.textContent = "Best match from your local plant image index.";

  if (metadata.image_url) {
    referenceImage.src = metadata.image_url;
    referenceImage.alt = bestMatch.name ? `Reference image for ${bestMatch.name}` : "Reference plant";
    referenceWrap.hidden = false;
    resultCard.classList.add("has-reference");
  } else {
    referenceWrap.hidden = true;
    referenceImage.removeAttribute("src");
    resultCard.classList.remove("has-reference");
  }

  confidenceBlock.hidden = false;
  emptyRecord.hidden = true;
  confidenceValue.textContent = hasConfidence ? `Confidence: ${confidencePercent.toFixed(1)}%` : "Confidence: n/a";
  confidenceLabel.textContent = bestMatch.confidence_label || "unknown";
  confidenceMeter.style.width = `${confidencePercent}%`;

  renderMetadata(metadata);
  renderOtherMatches(data.other_matches);
}

async function analyzeSelectedFile() {
  if (!selectedFile || isAnalyzing) {
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);

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
      throw new Error(payload.error || "Analysis failed.");
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
