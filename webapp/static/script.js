const form = document.getElementById("upload-form");
const analyzeBtn = document.getElementById("analyze-btn");
const spinner = analyzeBtn.querySelector(".spinner");
const btnText = analyzeBtn.querySelector(".btn-text");

const emptyState = document.getElementById("empty-state");
const resultContent = document.getElementById("result-content");
const statusBanner = document.getElementById("status-banner");

const verdictBadge = document.getElementById("verdict-badge");
const probabilityFill = document.getElementById("probability-fill");
const probabilityLabel = document.getElementById("probability-label");
const reportText = document.getElementById("report-text");

const imageInput = document.getElementById("image");
const audioInput = document.getElementById("audio");
const imagePreview = document.getElementById("image-preview");

// ---- Show the chosen file name (and a thumbnail for images) next to each dropzone ----
function wireDropzone(input, { preview } = {}) {
    const dropzone = input.closest(".dropzone");
    const filenameEl = dropzone.querySelector('[data-role="filename"]');

    input.addEventListener("change", () => {
        const file = input.files[0];
        filenameEl.textContent = file ? file.name : "";

        if (preview && file) {
            const url = URL.createObjectURL(file);
            preview.src = url;
            preview.hidden = false;
            dropzone.querySelector('[data-role="placeholder-icon"]').hidden = true;
        }
    });
}

wireDropzone(imageInput, { preview: imagePreview });
wireDropzone(audioInput);

// ---- Submit handler ----
form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const imageFile = imageInput.files[0];
    const audioFile = audioInput.files[0];

    if (!imageFile || !audioFile) {
        showStatus("Please choose both an image file and an audio file.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("audio", audioFile);

    setLoading(true);
    showStatus("Analyzing image and audio, then drafting an inspection report…", "loading");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            showStatus(data.error || "Something went wrong. Please try again.", "error");
            return;
        }

        showResult(data);
        hideStatus();

    } catch (error) {
        showStatus("Could not reach the server. Make sure app.py is running.", "error");
    } finally {
        setLoading(false);
    }
});

function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    spinner.hidden = !isLoading;
    btnText.textContent = isLoading ? "Analyzing…" : "Analyze";
}

function showStatus(message, kind) {
    statusBanner.textContent = message;
    statusBanner.className = `status-banner ${kind}`;
    statusBanner.hidden = false;
}

function hideStatus() {
    statusBanner.hidden = true;
    statusBanner.textContent = "";
    statusBanner.className = "status-banner";
}

function showResult(data) {
    const isCrack = data.label.toLowerCase().includes("crack") &&
        !data.label.toLowerCase().includes("no crack");
    const percentage = (data.probability * 100).toFixed(1);

    verdictBadge.textContent = `${data.label} — ${percentage}% crack probability`;
    verdictBadge.className = `verdict-badge ${isCrack ? "crack" : "no-crack"}`;

    probabilityFill.style.width = `${percentage}%`;
    probabilityLabel.textContent = `${percentage}%`;

    reportText.textContent = data.report || "No report was generated for this analysis.";

    emptyState.hidden = true;
    resultContent.hidden = false;
}
