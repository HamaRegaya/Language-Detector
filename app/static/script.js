/**
 * Language Detector — Frontend Logic
 * Live-typing detection with debouncing, model switching, animated bars.
 */

const API_BASE = "";
let currentModel = "cnn";
let debounceTimer = null;

// ── DOM Elements ──
const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const clearBtn = document.getElementById("clearBtn");
const resultSection = document.getElementById("resultSection");
const loading = document.getElementById("loading");
const statusBadge = document.getElementById("statusBadge");
const statusText = document.getElementById("statusText");
const predictedLang = document.getElementById("predictedLang");
const confidence = document.getElementById("confidence");
const latency = document.getElementById("latency");
const modelUsed = document.getElementById("modelUsed");
const confidenceFill = document.getElementById("confidenceFill");
const probBars = document.getElementById("probBars");

// ── Health Check ──
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        statusBadge.classList.add("online");
        const models = [];
        if (data.models.nb) models.push("NB");
        if (data.models.cnn) models.push("CNN");
        statusText.textContent = `Online · ${models.join(" + ")}`;

        // If CNN not available, default to NB
        if (!data.models.cnn && data.models.nb) {
            switchModel("nb");
        }
    } catch {
        statusText.textContent = "Offline";
        statusBadge.classList.remove("online");
    }
}

// ── Model Switching ──
function switchModel(model) {
    currentModel = model;
    document.querySelectorAll(".model-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.model === model);
    });
    // Re-predict if there's text
    const text = textInput.value.trim();
    if (text.length >= 3) {
        predict(text);
    }
}

document.querySelectorAll(".model-btn").forEach(btn => {
    btn.addEventListener("click", () => switchModel(btn.dataset.model));
});

// ── Predict ──
async function predict(text) {
    if (text.length < 3) {
        resultSection.style.display = "none";
        return;
    }

    loading.style.display = "flex";
    resultSection.style.display = "none";

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, model: currentModel }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Update main prediction
        predictedLang.textContent = data.language;
        confidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        latency.textContent = `${data.latency_ms.toFixed(1)}ms`;
        modelUsed.textContent = data.model_used.toUpperCase();
        confidenceFill.style.width = `${data.confidence * 100}%`;

        // Build probability bars (sorted descending)
        const sorted = Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1]);
        const maxProb = sorted[0][1];

        probBars.innerHTML = sorted.map(([lang, prob], i) => `
            <div class="prob-row">
                <span class="prob-label">${lang}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill ${i === 0 ? 'top' : 'other'}"
                         style="width: ${(prob / maxProb) * 100}%"></div>
                </div>
                <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
            </div>
        `).join("");

        loading.style.display = "none";
        resultSection.style.display = "block";
    } catch (err) {
        loading.style.display = "none";
        console.error("Prediction failed:", err);
    }
}

// ── Live Typing with Debounce ──
textInput.addEventListener("input", () => {
    const text = textInput.value;
    charCount.textContent = `${text.length} / 512 chars`;

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        predict(text.trim());
    }, 300);
});

// ── Clear ──
clearBtn.addEventListener("click", () => {
    textInput.value = "";
    charCount.textContent = "0 / 512 chars";
    resultSection.style.display = "none";
    textInput.focus();
});

// ── Demo Buttons ──
document.querySelectorAll(".demo-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        textInput.value = btn.dataset.text;
        charCount.textContent = `${btn.dataset.text.length} / 512 chars`;
        predict(btn.dataset.text);
    });
});

// ── Init ──
checkHealth();
setInterval(checkHealth, 30000);
