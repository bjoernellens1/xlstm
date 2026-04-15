/**
 * xLSTM Chat – Frontend JavaScript
 *
 * Handles WebSocket communication, message rendering, model management,
 * and generation settings. Designed for high-performance real-time streaming.
 *
 * Architecture:
 *   - WebSocket for streaming token generation
 *   - REST API for model management and configuration
 *   - Event-driven UI updates with minimal DOM manipulation
 */

"use strict";

// ── State ───────────────────────────────────────────────────────────────────

const state = {
    messages: [],
    ws: null,
    isGenerating: false,
    modelLoaded: false,
    showPerformance: true,
};

// ── DOM Elements ────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const messagesEl = $("#messages");
const inputEl = $("#message-input");
const sendBtn = $("#send-btn");
const loadModelBtn = $("#load-model-btn");
const modelVariantEl = $("#model-variant");
const modelStatusEl = $("#model-status");
const modelInfoEl = $("#model-info");
const typingIndicator = $("#typing-indicator");
const tempSlider = $("#temperature");
const tempValue = $("#temp-value");
const tokensSlider = $("#max-tokens");
const tokensValue = $("#tokens-value");
const topkSlider = $("#top-k");
const topkValue = $("#topk-value");

// ── API Helpers ─────────────────────────────────────────────────────────────

const API_BASE = "/api/v1";

/**
 * Make an API request with error handling.
 * @param {string} endpoint - API path (relative to /api/v1/).
 * @param {object} options  - Fetch options.
 * @returns {Promise<object>} Parsed JSON response.
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const res = await fetch(url, {
        headers: { "Content-Type": "application/json", ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "API request failed");
    }
    return res.json();
}

// ── WebSocket Management ────────────────────────────────────────────────────

/**
 * Get the WebSocket URL for the current host.
 * @returns {string} WebSocket URL.
 */
function getWsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}${API_BASE}/chat/ws`;
}

/**
 * Ensure the WebSocket connection is open.
 * @returns {Promise<WebSocket>} Connected WebSocket.
 */
function ensureWebSocket() {
    return new Promise((resolve, reject) => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            resolve(state.ws);
            return;
        }

        const ws = new WebSocket(getWsUrl());
        state.ws = ws;

        ws.onopen = () => resolve(ws);
        ws.onerror = (e) => reject(new Error("WebSocket connection failed"));
        ws.onclose = () => {
            state.ws = null;
        };
    });
}

// ── Message Rendering ───────────────────────────────────────────────────────

/**
 * Create a message DOM element.
 * @param {string} role    - "user" or "assistant".
 * @param {string} content - Message text.
 * @returns {HTMLElement} Message element.
 */
function createMessageElement(role, content) {
    const msg = document.createElement("div");
    msg.className = `message message-${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "U" : "X";

    const contentEl = document.createElement("div");
    contentEl.className = "message-content";
    contentEl.textContent = content;

    msg.appendChild(avatar);
    msg.appendChild(contentEl);
    return msg;
}

/**
 * Clear the welcome message if present.
 */
function clearWelcome() {
    const welcome = messagesEl.querySelector(".welcome-message");
    if (welcome) welcome.remove();
}

/**
 * Scroll the messages container to the bottom.
 */
function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

/**
 * Safely append a labelled info line to a container using DOM API.
 * @param {HTMLElement} container - Parent element.
 * @param {string}      label    - Bold label text.
 * @param {string}      value    - Value text.
 */
function appendInfoLine(container, label, value) {
    if (container.childNodes.length > 0) {
        container.appendChild(document.createElement("br"));
    }
    const strong = document.createElement("strong");
    strong.textContent = label + ": ";
    container.appendChild(strong);
    container.appendChild(document.createTextNode(value));
}

/**
 * Add a message to the chat and render it.
 * @param {string} role    - "user" or "assistant".
 * @param {string} content - Message text.
 * @returns {HTMLElement} The created message element.
 */
function addMessage(role, content) {
    clearWelcome();
    state.messages.push({ role, content });
    const el = createMessageElement(role, content);
    messagesEl.appendChild(el);
    scrollToBottom();
    return el;
}

// ── Chat Logic ──────────────────────────────────────────────────────────────

/**
 * Send a message and stream the response via WebSocket.
 */
async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || state.isGenerating || !state.modelLoaded) return;

    inputEl.value = "";
    autoResizeInput();
    addMessage("user", text);

    state.isGenerating = true;
    sendBtn.disabled = true;
    typingIndicator.classList.remove("hidden");

    try {
        const ws = await ensureWebSocket();

        // Create assistant message placeholder
        const assistantEl = addMessage("assistant", "");
        const contentEl = assistantEl.querySelector(".message-content");
        let fullText = "";

        // Set up one-time message handler for this generation
        const handleMessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
                contentEl.textContent = `Error: ${data.error}`;
                cleanup();
                return;
            }

            if (data.done) {
                // Update the stored message with complete text
                state.messages[state.messages.length - 1].content = fullText;
                // Show performance metrics if available and enabled
                if (data.performance && state.showPerformance) {
                    showPerformanceMetrics(assistantEl, data.performance);
                }
                cleanup();
                return;
            }

            if (data.token !== undefined) {
                fullText += data.token;
                contentEl.textContent = fullText;
                scrollToBottom();
            }
        };

        const cleanup = () => {
            ws.removeEventListener("message", handleMessage);
            state.isGenerating = false;
            sendBtn.disabled = false;
            typingIndicator.classList.add("hidden");
        };

        ws.addEventListener("message", handleMessage);

        // Send the chat request
        ws.send(JSON.stringify({
            messages: state.messages.slice(0, -1), // Exclude empty assistant msg
            max_new_tokens: parseInt(tokensSlider.value),
            temperature: parseFloat(tempSlider.value),
            top_k: parseInt(topkSlider.value),
        }));

    } catch (err) {
        // Fallback to REST API if WebSocket fails
        console.warn("WebSocket failed, falling back to REST:", err.message);
        await sendMessageREST(text);
    }
}

/**
 * Fallback: send message via REST API (non-streaming).
 * @param {string} text - Already-added user message text.
 */
async function sendMessageREST(text) {
    try {
        const response = await apiRequest("/chat", {
            method: "POST",
            body: JSON.stringify({
                messages: state.messages.filter(m => m.content), // Exclude empty
                max_new_tokens: parseInt(tokensSlider.value),
                temperature: parseFloat(tempSlider.value),
                top_k: parseInt(topkSlider.value),
                stream: false,
            }),
        });

        // Update the last (empty) assistant message
        const lastAssistant = state.messages[state.messages.length - 1];
        lastAssistant.content = response.message.content;
        const lastEl = messagesEl.lastElementChild;
        lastEl.querySelector(".message-content").textContent = response.message.content;

        // Show performance metrics if available and enabled
        if (response.performance && state.showPerformance) {
            showPerformanceMetrics(lastEl, response.performance);
        }

    } catch (err) {
        const lastEl = messagesEl.lastElementChild;
        lastEl.querySelector(".message-content").textContent = `Error: ${err.message}`;
    } finally {
        state.isGenerating = false;
        sendBtn.disabled = false;
        typingIndicator.classList.add("hidden");
    }
}

// ── Model Management ────────────────────────────────────────────────────────

/**
 * Load the selected model variant.
 */
async function loadModel() {
    const variant = modelVariantEl.value;
    loadModelBtn.disabled = true;
    modelStatusEl.className = "status-badge status-loading";
    modelStatusEl.textContent = `Loading ${variant}...`;

    try {
        const info = await apiRequest("/models/load", {
            method: "POST",
            body: JSON.stringify({ variant }),
        });

        state.modelLoaded = true;
        modelStatusEl.className = "status-badge status-online";
        modelStatusEl.textContent = `${variant} model loaded`;
        sendBtn.disabled = false;

        // Show model info using safe DOM API
        modelInfoEl.classList.remove("hidden");
        modelInfoEl.textContent = "";
        appendInfoLine(modelInfoEl, "Parameters", info.parameter_count_human || "—");
        appendInfoLine(modelInfoEl, "Device", info.device || "—");
        appendInfoLine(modelInfoEl, "Vocab", info.vocab_size != null ? info.vocab_size.toLocaleString() : "—");

    } catch (err) {
        modelStatusEl.className = "status-badge status-offline";
        modelStatusEl.textContent = `Failed: ${err.message}`;
        state.modelLoaded = false;
    } finally {
        loadModelBtn.disabled = false;
    }
}

/**
 * Fetch and display current model status on page load.
 */
async function checkModelStatus() {
    try {
        const info = await apiRequest("/models/current");
        if (info.loaded) {
            state.modelLoaded = true;
            modelStatusEl.className = "status-badge status-online";
            modelStatusEl.textContent = `${info.variant} model loaded`;
            sendBtn.disabled = false;

            modelInfoEl.classList.remove("hidden");
            modelInfoEl.textContent = "";
            appendInfoLine(modelInfoEl, "Parameters", info.parameter_count_human || "—");
            appendInfoLine(modelInfoEl, "Device", info.device || "—");
            appendInfoLine(modelInfoEl, "Vocab", info.vocab_size != null ? info.vocab_size.toLocaleString() : "—");
        }
    } catch {
        // Server might not be ready yet
    }
}

// ── Input Handling ──────────────────────────────────────────────────────────

/**
 * Auto-resize the textarea based on content.
 */
function autoResizeInput() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + "px";
}

// ── Performance Metrics ─────────────────────────────────────────────────────

/**
 * Render a performance metrics bar below a message element.
 * Uses DOM API (not innerHTML) to avoid XSS risks.
 * @param {HTMLElement} messageEl  - The message DOM element.
 * @param {object}      perf      - Performance data from the server.
 */
function showPerformanceMetrics(messageEl, perf) {
    // Remove any existing metrics bar on this message
    const existing = messageEl.querySelector(".perf-metrics");
    if (existing) existing.remove();

    const bar = document.createElement("div");
    bar.className = "perf-metrics";

    /**
     * Create a single metric item element safely via DOM API.
     * @param {string} icon  - Emoji icon.
     * @param {*}      value - Numeric metric value.
     * @param {string} unit  - Unit label.
     * @returns {HTMLElement}
     */
    function createMetricItem(icon, value, unit) {
        const span = document.createElement("span");
        span.className = "perf-item";
        span.textContent = icon + " ";
        const strong = document.createElement("strong");
        strong.textContent = String(value);
        span.appendChild(strong);
        span.appendChild(document.createTextNode(" " + unit));
        return span;
    }

    if (perf.tokens_per_second !== undefined) {
        bar.appendChild(createMetricItem("⚡", perf.tokens_per_second, "tok/s"));
    }
    if (perf.packets_per_second !== undefined) {
        bar.appendChild(createMetricItem("📡", perf.packets_per_second, "packets/s"));
    }
    if (perf.tokens_generated !== undefined) {
        bar.appendChild(createMetricItem("📝", perf.tokens_generated, "tokens"));
    }
    if (perf.generation_time_s !== undefined) {
        bar.appendChild(createMetricItem("⏱️", perf.generation_time_s, "s"));
    }

    messageEl.appendChild(bar);
    scrollToBottom();
}

/**
 * Toggle performance metrics display on/off.
 */
function togglePerformance() {
    state.showPerformance = !state.showPerformance;
    const btn = $("#toggle-perf-btn");
    if (btn) {
        btn.textContent = state.showPerformance ? "📊 Metrics: ON" : "📊 Metrics: OFF";
        btn.classList.toggle("perf-on", state.showPerformance);
        btn.classList.toggle("perf-off", !state.showPerformance);
    }
    // Hide/show existing metrics bars
    document.querySelectorAll(".perf-metrics").forEach((el) => {
        el.style.display = state.showPerformance ? "" : "none";
    });
}

// ── Event Listeners ─────────────────────────────────────────────────────────

sendBtn.addEventListener("click", sendMessage);

inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

inputEl.addEventListener("input", autoResizeInput);

loadModelBtn.addEventListener("click", loadModel);

// Performance toggle
const perfBtn = $("#toggle-perf-btn");
if (perfBtn) perfBtn.addEventListener("click", togglePerformance);

// Slider value displays
tempSlider.addEventListener("input", () => {
    tempValue.textContent = tempSlider.value;
});
tokensSlider.addEventListener("input", () => {
    tokensValue.textContent = tokensSlider.value;
});
topkSlider.addEventListener("input", () => {
    topkValue.textContent = topkSlider.value;
});

// ── Initialization ──────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    checkModelStatus();
    inputEl.focus();
});
