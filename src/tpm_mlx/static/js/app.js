/**
 * TPM-MLX Playground Application Coordinator
 * Connects UI events, streaming text completion, image diffusion pipelines,
 * multi-session storage, and responsive sidebar.
 */

import { initTheme } from "./theme.js";
import { escapeHtml, formatMessageHtml, toggleThought } from "./markdown.js";
import { initLightbox, openImageLightbox, closeImageLightbox, copyImageLink, handleUnsupportedModify } from "./lightbox.js";
import { fetchModelsApi, loadModelApi, uploadImageApi, generateImageApi, streamChatCompletion } from "./api.js";
import {
    chatSessions,
    activeSessionId,
    getActiveSession,
    createNewSession,
    switchSession,
    deleteSession,
    clearAllSessions,
    exportCurrentChat,
    autoTitleSession,
    saveSessionsState,
    loadSessionsState,
    renderSessionsList,
    registerSessionCallbacks
} from "./sessions.js";

// Application State
let activeModel = "";
let activeDraftModel = null;
let activeImageModel = null;
let cachedModelsList = [];
let attachedReferenceImageUrl = null;

// DOM Elements
const newChatBtn = document.getElementById("new-chat-btn");
const headerNewChatBtn = document.getElementById("header-new-chat-btn");
const exportChatBtn = document.getElementById("export-chat-btn");
const clearSessionsBtn = document.getElementById("clear-sessions-btn");
const modelSelect = document.getElementById("model-select");
const draftSelect = document.getElementById("draft-select");
const customModelGroup = document.getElementById("custom-model-group");
const customModelInput = document.getElementById("custom-model-input");
const customDraftGroup = document.getElementById("custom-draft-group");
const customDraftInput = document.getElementById("custom-draft-input");
const draftTokensSelect = document.getElementById("draft-tokens-select");
const kvSizeInput = document.getElementById("kv-size-input");
const loadModelBtn = document.getElementById("load-model-btn");
const tempSlider = document.getElementById("temp-slider");
const tempVal = document.getElementById("temp-val");
const tokensInput = document.getElementById("tokens-input");
const reasoningCheckbox = document.getElementById("reasoning-checkbox");
const reasoningContainer = document.getElementById("reasoning-toggle-container");
const reasoningStateBadge = document.getElementById("reasoning-state-badge");
const statusBadge = document.getElementById("status-badge");
const statusText = document.getElementById("status-text");
const currentModelHeader = document.getElementById("current-model-header");
const headerSpeculationBadge = document.getElementById("header-speculation-badge");
const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const uploadBtn = document.getElementById("upload-btn");
const imageUploadInput = document.getElementById("image-upload-input");
const attachedImageContainer = document.getElementById("attached-image-container");
const attachedImageThumb = document.getElementById("attached-image-thumb");
const attachedImageName = document.getElementById("attached-image-name");
const sidebar = document.getElementById("sidebar");
const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
const sidebarCloseBtn = document.getElementById("sidebar-close-btn");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const headerStatusDot = document.getElementById("header-status-dot");
const headerStatusLabel = document.getElementById("header-status-label");
const autoExpandToggle = document.getElementById("auto-expand-toggle");
const directorDesc = document.getElementById("director-status-desc");

/**
 * Checks if current viewport is mobile (< 768px).
 * @returns {boolean}
 */
export function isMobileView() {
    return window.innerWidth <= 768;
}

/**
 * Updates UI server status indicators.
 * @param {string} status
 * @param {string} color
 * @param {string} text
 */
export function updateServerStatus(status, color, text) {
    if (statusText) statusText.textContent = text;
    const dot = statusBadge ? statusBadge.querySelector(".status-dot") : null;
    if (dot) dot.style.backgroundColor = color;
    if (headerStatusDot) {
        headerStatusDot.style.backgroundColor = color;
        headerStatusDot.parentElement.title = `Server Status: ${text}`;
    }
    if (headerStatusLabel) {
        headerStatusLabel.textContent = text.replace("Server ", "").replace("Model ", "");
    }
    if (statusBadge) {
        if (color === "#ef4444") {
            statusBadge.style.borderColor = "rgba(239, 68, 68, 0.25)";
            statusBadge.style.color = "#ef4444";
        } else if (color === "#fbbf24") {
            statusBadge.style.borderColor = "rgba(251, 191, 36, 0.25)";
            statusBadge.style.color = "#fbbf24";
        } else {
            statusBadge.style.borderColor = "var(--status-connected-border)";
            statusBadge.style.color = "var(--status-connected)";
        }
    }
}

/**
 * Sidebar navigation handling
 */
export function toggleSidebar() {
    if (isMobileView()) {
        document.body.classList.toggle("sidebar-open");
        document.body.classList.remove("sidebar-collapsed");
    } else {
        const isCollapsed = document.body.classList.toggle("sidebar-collapsed");
        try {
            localStorage.setItem("tpm_mlx_sidebar_collapsed", isCollapsed ? "true" : "false");
        } catch(e) {}
    }
}

export function closeSidebar() {
    if (isMobileView()) {
        document.body.classList.remove("sidebar-open");
    } else {
        document.body.classList.add("sidebar-collapsed");
        try {
            localStorage.setItem("tpm_mlx_sidebar_collapsed", "true");
        } catch(e) {}
    }
}

export function openSidebar() {
    if (isMobileView()) {
        document.body.classList.add("sidebar-open");
    } else {
        document.body.classList.remove("sidebar-collapsed");
        try {
            localStorage.setItem("tpm_mlx_sidebar_collapsed", "false");
        } catch(e) {}
    }
}

/**
 * Checks if a model name refers to an image diffusion model.
 * @param {string} name
 * @returns {boolean}
 */
export function isImageModelName(name) {
    if (!name) return false;
    const lower = name.toLowerCase();
    return lower.includes("flux") || 
           lower.includes("diffusion") || 
           (lower.includes("bonsai") && lower.includes("image")) || 
           lower.includes("klein") ||
           lower.includes("sdxl") ||
           lower.includes("stable-diffusion") ||
           lower.includes("z-image") ||
           lower.includes("z_image") ||
           lower.includes("zimage");
}

/**
 * Strips organizational repo prefixes for clean, single-line mobile dropdown display.
 * @param {string} modelId 
 * @param {boolean} isActive 
 * @returns {string}
 */
export function formatModelDisplayLabel(modelId, isActive = false) {
    if (!modelId) return "";
    const parts = modelId.split("/");
    const shortName = parts.length > 1 ? parts.slice(1).join("/") : modelId;
    return shortName + (isActive ? " (active)" : "");
}

/**
 * Updates top header model name and speculation indicator badge.
 */
export function updateHeaderBadge(modelName, draftName, specMode, hasMtp, draftTokens, backend) {
    window._lastSpecMode = specMode;
    window._lastHasMtp = hasMtp;
    window._lastDraftTokens = draftTokens;
    window._lastBackend = backend;

    const isMobile = isMobileView();
    currentModelHeader.textContent = modelName ? (isMobile ? modelName.split("/").pop() : modelName) : "Select and Load a Model";
    currentModelHeader.title = modelName || "";
    
    const isImage = (backend === 'image') || isImageModelName(modelName);

    if (modelName) {
        headerSpeculationBadge.style.display = "inline-block";
        if (isImage) {
            headerSpeculationBadge.textContent = isMobile ? "🎨 Image Gen" : "🎨 Flow Matching Image Generator";
            headerSpeculationBadge.title = "Native Diffusion Flow Matching Image Generator on MLX";
            headerSpeculationBadge.style.background = "var(--card-bg-subtle)";
            headerSpeculationBadge.style.color = "var(--text-primary)";
            headerSpeculationBadge.style.borderColor = "var(--border-hover)";
        } else if (specMode === "mtp" || hasMtp) {
            const tokensText = draftTokens ? `${draftTokens} Draft Tokens` : "Active";
            headerSpeculationBadge.textContent = isMobile ? `⚡ MTP (${draftTokens || 3} tok)` : `⚡ MTP Speculation (${tokensText}) [${(backend || 'LLM').toUpperCase()}]`;
            headerSpeculationBadge.title = `MTP Speculative Decoding with ${draftTokens || 3} tokens lookahead`;
            headerSpeculationBadge.style.background = "var(--status-connected-bg)";
            headerSpeculationBadge.style.color = "var(--status-connected)";
            headerSpeculationBadge.style.borderColor = "var(--status-connected-border)";
        } else if (specMode === "draft" && draftName) {
            headerSpeculationBadge.textContent = isMobile ? "⚡ Draft Spec" : `⚡ Draft Speculation [${(backend || 'LLM').toUpperCase()}]`;
            headerSpeculationBadge.title = `Draft Speculation via ${draftName}`;
            headerSpeculationBadge.style.background = "var(--surface-hover)";
            headerSpeculationBadge.style.color = "var(--text-primary)";
            headerSpeculationBadge.style.borderColor = "var(--border-hover)";
        } else {
            headerSpeculationBadge.textContent = isMobile ? "Standard" : `Standard Decoding [${(backend || 'LLM').toUpperCase()}]`;
            headerSpeculationBadge.title = "Standard autoregressive single-token decoding";
            headerSpeculationBadge.style.background = "var(--card-bg-subtle)";
            headerSpeculationBadge.style.color = "var(--text-secondary)";
            headerSpeculationBadge.style.borderColor = "var(--border-color)";
        }
    } else {
        headerSpeculationBadge.style.display = "none";
    }
}

/**
 * Suggests matching companion draft assistant for base model.
 * @param {string} baseModelId
 */
export function autoSelectMatchingDraft(baseModelId) {
    if (!baseModelId || baseModelId === "__CUSTOM__") return;
    if (isImageModelName(baseModelId)) {
        draftSelect.value = "__NONE__";
        return;
    }
    const lower = baseModelId.toLowerCase();
    const cleanBase = lower.replace(/^.*?\//, "").replace(/-4bit|-8bit|-bf16|-fp16|-it|-instruct/g, "");
    
    // Find matching companion assistant option in draftSelect dynamically
    let matchIdx = -1;
    // Pass 1: exact match with preference for official bf16 assistant on Gemma
    for (let i = 0; i < draftSelect.options.length; i++) {
        const optVal = draftSelect.options[i].value;
        const optLower = optVal.toLowerCase();
        if (optVal === "__AUTO__" || optVal === "__NONE__" || optVal === "__CUSTOM__") continue;

        const cleanDraft = optLower.replace(/^.*?\//, "").replace(/-4bit|-8bit|-bf16|-fp16|-it|-instruct/g, "");
        const draftWithoutTag = cleanDraft.replace(/-mtp|-assistant|_mtp|_assistant/g, "");

        if (draftWithoutTag === cleanBase || cleanDraft.includes(cleanBase)) {
            if (lower.includes("gemma") && optLower.includes("bf16")) {
                matchIdx = i;
                break;
            }
            if (matchIdx === -1) matchIdx = i;
        }

        // Tokenized match: all base tokens present in draft name with mtp/assistant tag
        const baseTokens = cleanBase.split(/[-_]/).filter(t => t.length > 0);
        const draftTokens = optLower.split(/[-_/]/).filter(t => t.length > 0);
        const allBaseTokens = baseTokens.every(tok => draftTokens.includes(tok));
        const hasSpecTag = draftTokens.some(t => t.includes("mtp") || t.includes("assistant"));
        if (allBaseTokens && hasSpecTag) {
            if (lower.includes("gemma") && optLower.includes("bf16")) {
                matchIdx = i;
                break;
            }
            if (matchIdx === -1) matchIdx = i;
        }
    }

    if (matchIdx !== -1) {
        draftSelect.selectedIndex = matchIdx;
    } else {
        // Reset to None (Single-Token Baseline) when model has no companion assistant
        draftSelect.value = "__NONE__";
    }
}

/**
 * Adjusts visible controls when an image model vs text LLM is selected.
 */
export function handleModelSelectionChange() {
    let selectedVal = modelSelect.value;
    if (selectedVal === "__CUSTOM__") {
        selectedVal = customModelInput.value.trim();
    }
    const isImage = isImageModelName(selectedVal);
    
    const noticeEl = document.getElementById("image-model-notice");
    const tempContainer = tempSlider.closest(".control-group");
    const tokensContainer = tokensInput.closest(".control-group");
    const kvContainer = kvSizeInput.closest(".control-group");
    const lookaheadContainer = draftTokensSelect.closest(".control-group");
    const draftContainer = draftSelect.closest(".control-group");
    const paramsTitle = document.getElementById("parameters-section-title");
    const resContainer = document.getElementById("image-resolution-container");
    const dirContainer = document.getElementById("image-director-container");
    const reasoningParent = reasoningContainer ? reasoningContainer.closest(".control-group") : null;

    if (isImage) {
        // Hide MTP & text-generation controls completely (never show MTP for image models)
        if (draftContainer) draftContainer.style.display = "none";
        if (customDraftGroup) customDraftGroup.style.display = "none";
        if (lookaheadContainer) lookaheadContainer.style.display = "none";
        if (kvContainer) kvContainer.style.display = "none";
        if (paramsTitle) paramsTitle.style.display = "none";
        if (tempContainer) tempContainer.style.display = "none";
        if (tokensContainer) tokensContainer.style.display = "none";
        if (reasoningParent) reasoningParent.style.display = "none";

        draftSelect.value = "__NONE__";
        draftSelect.disabled = true;
        draftTokensSelect.disabled = true;
        kvSizeInput.disabled = true;
        tempSlider.disabled = true;
        tokensInput.disabled = true;
        reasoningCheckbox.disabled = true;

        if (noticeEl) noticeEl.style.display = "block";
        if (resContainer) resContainer.style.display = "flex";
        if (dirContainer) dirContainer.style.display = "block";
    } else {
        // Restore MTP & LLM controls
        if (draftContainer) {
            draftContainer.style.display = "flex";
            draftContainer.style.opacity = "1";
            draftContainer.style.filter = "none";
            draftContainer.style.pointerEvents = "auto";
        }
        if (customDraftGroup) {
            customDraftGroup.style.display = draftSelect.value === "__CUSTOM__" ? "flex" : "none";
        }
        if (lookaheadContainer) {
            lookaheadContainer.style.display = "flex";
            lookaheadContainer.style.opacity = "1";
            lookaheadContainer.style.filter = "none";
            lookaheadContainer.style.pointerEvents = "auto";
        }
        if (kvContainer) {
            kvContainer.style.display = "flex";
            kvContainer.style.opacity = "1";
            kvContainer.style.filter = "none";
            kvContainer.style.pointerEvents = "auto";
        }
        if (paramsTitle) {
            paramsTitle.style.display = "block";
            paramsTitle.style.opacity = "1";
        }
        if (tempContainer) {
            tempContainer.style.display = "flex";
            tempContainer.style.opacity = "1";
            tempContainer.style.filter = "none";
            tempContainer.style.pointerEvents = "auto";
        }
        if (tokensContainer) {
            tokensContainer.style.display = "flex";
            tokensContainer.style.opacity = "1";
            tokensContainer.style.filter = "none";
            tokensContainer.style.pointerEvents = "auto";
        }
        if (reasoningParent) {
            reasoningParent.style.display = "flex";
            reasoningParent.style.opacity = "1";
            reasoningParent.style.filter = "none";
            reasoningParent.style.pointerEvents = "auto";
        }

        draftSelect.disabled = false;
        draftTokensSelect.disabled = false;
        kvSizeInput.disabled = false;
        tempSlider.disabled = false;
        tokensInput.disabled = false;
        reasoningCheckbox.disabled = false;

        if (noticeEl) noticeEl.style.display = "none";
        if (resContainer) resContainer.style.display = "none";
        if (dirContainer) dirContainer.style.display = "none";
        autoSelectMatchingDraft(selectedVal);
    }
}

/**
 * Extracts resolution like 1024x1024, 16:9, 1:1 from user prompt.
 * @param {string} text
 * @param {string} fallback
 * @returns {string}
 */
export function extractResolution(text, fallback = "1024x1024") {
    if (!text) return fallback;
    const lower = text.toLowerCase();
    
    const dimMatch = lower.match(/\b(\d{3,4})\s*[xX×*]\s*(\d{3,4})\b/);
    if (dimMatch) {
        return `${dimMatch[1]}x${dimMatch[2]}`;
    }
    
    if (/\b16\s*:\s*9\b/.test(lower)) return "1024x576";
    if (/\b9\s*:\s*16\b/.test(lower)) return "576x1024";
    if (/\b4\s*:\s*3\b/.test(lower)) return "1024x768";
    if (/\b3\s*:\s*4\b/.test(lower)) return "768x1024";
    if (/\b1\s*:\s*1\b/.test(lower) || /\b(square|1k)\b/.test(lower)) return "1024x1024";

    return fallback;
}

/**
 * Synchronizes the reasoning state badge text and style.
 */
export function updateReasoningBadge() {
    if (!reasoningStateBadge) return;
    if (reasoningCheckbox.checked) {
        reasoningStateBadge.textContent = "ON";
        reasoningStateBadge.style.background = "var(--surface-hover)";
        reasoningStateBadge.style.color = "var(--text-primary)";
        reasoningStateBadge.style.borderColor = "var(--border-hover)";
    } else {
        reasoningStateBadge.textContent = "OFF";
        reasoningStateBadge.style.background = "var(--card-bg-subtle)";
        reasoningStateBadge.style.color = "var(--text-tertiary)";
        reasoningStateBadge.style.borderColor = "var(--border-color)";
    }
}

/**
 * Attaches a reference image for image-to-image editing or VLM vision understanding.
 * @param {string} url
 * @param {string} displayName
 */
export function attachReferenceImage(url, displayName = "Reference Image") {
    const modelLower = (activeModel || "").toLowerCase();
    const isFlux = modelLower.includes("flux");
    const isVlm = modelLower.includes("gemma") || modelLower.includes("qwen") || modelLower.includes("vl");

    if (!isFlux && !isVlm) {
        handleUnsupportedModify();
        return;
    }

    attachedReferenceImageUrl = url;
    attachedImageThumb.src = url;
    attachedImageName.textContent = displayName;
    attachedImageContainer.style.display = "inline-flex";
    userInput.placeholder = isFlux 
        ? "Describe changes to make to this image with FLUX.2 (e.g. change lighting, add object)..."
        : "Ask a question about this image with the VLM...";
    userInput.focus();
}

/**
 * Removes reference image.
 */
export function clearAttachedReferenceImage() {
    attachedReferenceImageUrl = null;
    attachedImageContainer.style.display = "none";
    attachedImageThumb.src = "";
    userInput.placeholder = "Type a message, or /image <prompt> to generate an image... (Enter to send, Shift+Enter for newline)";
}

/**
 * Builds HTML for a generated image card.
 */
export function buildImageCardHtml(item) {
    const promptSafe = escapeHtml(item.imagePrompt || "");
    const revisedSafe = escapeHtml(item.revisedPrompt || "");
    const altText = revisedSafe || promptSafe || "Generated Image";
    const supportsEdit = item.supportsEdit !== undefined ? item.supportsEdit : true;

    return `
        <div class="image-result-card">
            <img src="${item.url}" class="generated-image" alt="${altText}" title="Click to view full screen" />
            ${revisedSafe && revisedSafe !== promptSafe ? `
                <div class="revised-prompt-tag">
                    <b>🪄 Visual Prompt:</b> ${revisedSafe}
                </div>
            ` : ''}
            <div class="image-actions">
                <button type="button" class="img-action-btn" onclick="openImageLightbox('${item.url}', this.closest('.image-result-card').querySelector('.generated-image').alt)">
                    🔍 Expand
                </button>
                <a href="${item.url}" download="generated_image.png" class="img-action-btn">
                    ⬇️ Download PNG
                </a>
                ${supportsEdit ? `
                    <button type="button" class="img-action-btn" onclick="attachReferenceImage('${item.url}', 'Generated Image')">
                        ✏️ Modify Image
                    </button>
                ` : `
                    <button type="button" class="img-action-btn" style="opacity: 0.55;" title="Image-to-Image editing is supported by FLUX.2 Klein models" onclick="handleUnsupportedModify()">
                        ✏️ Modify (FLUX only)
                    </button>
                `}
                <button type="button" class="img-action-btn" onclick="copyImageLink(this, '${item.url}')">
                    📋 Copy Link
                </button>
            </div>
        </div>
    `;
}

/**
 * Builds HTML for image generation metrics bar.
 */
export function buildImageMetricsHtml(metrics = {}, modelDisplayName = "") {
    const model = modelDisplayName || (metrics.model ? metrics.model.split("/").pop() : "");
    return `
        ${model ? `
        <div class="metric" data-tooltip="Model used for this generation run">
            <span class="metric-icon">🤖</span>
            <span class="metric-val" style="color: var(--primary);">${escapeHtml(model)}</span>
        </div>` : ''}
        ${metrics.generation_time_s !== undefined ? `
        <div class="metric" data-tooltip="Total Image Generation Latency">
            <span class="metric-icon">⏱️</span>
            <span>Latency:</span>
            <span class="metric-val">${metrics.generation_time_s}s</span>
        </div>` : ''}
        ${metrics.steps !== undefined ? `
        <div class="metric" data-tooltip="Denoising flow steps & step rate">
            <span class="metric-icon">🎨</span>
            <span>Steps:</span>
            <span class="metric-val">${metrics.steps} ${metrics.step_time_s !== undefined ? `(${metrics.step_time_s}s/step)` : ''}</span>
        </div>` : ''}
        ${metrics.width && metrics.height ? `
        <div class="metric" data-tooltip="Image Canvas Resolution">
            <span class="metric-icon">📐</span>
            <span>Size:</span>
            <span class="metric-val">${metrics.width}x${metrics.height}</span>
        </div>` : ''}
        ${metrics.peak_memory_gb ? `
        <div class="metric" data-tooltip="Peak Unified Memory during this generation run on Apple Silicon">
            <span class="metric-icon">💾</span>
            <span>Memory:</span>
            <span class="metric-val">${metrics.peak_memory_gb} GB</span>
        </div>` : ''}
    `;
}

/**
 * Builds HTML for LLM completion metrics bar.
 */
export function buildChatMetricsHtml(metrics = {}, usage = {}) {
    const promptTokens = (usage && usage.prompt_tokens) || metrics.prompt_tokens || 0;
    const genTokens = (usage && usage.completion_tokens) || metrics.generation_tokens || 0;
    
    let specBadge = "";
    if (metrics.speculation_mode && metrics.speculation_mode !== "none") {
        const accPercent = metrics.acceptance_rate !== undefined ? (metrics.acceptance_rate * 100).toFixed(1) : null;
        specBadge = `
            <div class="metric metric-accent" data-tooltip="Speculative Decoding (${metrics.speculation_mode.toUpperCase()}) Token Acceptance Rate">
                <span class="metric-icon">🎯</span>
                <span>Speculation:</span>
                <span class="metric-val">${metrics.speculation_mode.toUpperCase()}${accPercent ? ` (${accPercent}% acc)` : ''}</span>
            </div>
        `;
    }
    
    return `
        ${metrics.tps !== undefined ? `
        <div class="metric" data-tooltip="Generation Throughput (tokens generated per second during response)">
            <span class="metric-icon">⚡</span>
            <span>Speed:</span>
            <span class="metric-val">${metrics.tps} tok/s</span>
        </div>` : ''}
        ${metrics.ttft_ms !== undefined ? `
        <div class="metric" data-tooltip="Time-To-First-Token: Latency from request to emitting the first token">
            <span class="metric-icon">⏱️</span>
            <span>TTFT:</span>
            <span class="metric-val">${metrics.ttft_ms} ms</span>
        </div>` : ''}
        ${metrics.prompt_tps !== undefined ? `
        <div class="metric" data-tooltip="Prompt Ingestion Speed & Token Length">
            <span class="metric-icon">📥</span>
            <span>Prompt:</span>
            <span class="metric-val">${promptTokens ? promptTokens + ' tok @ ' : ''}${metrics.prompt_tps} tok/s</span>
        </div>` : ''}
        ${genTokens ? `
        <div class="metric" data-tooltip="Total Output Tokens Generated">
            <span class="metric-icon">📝</span>
            <span>Output:</span>
            <span class="metric-val">${genTokens} tokens</span>
        </div>` : ''}
        ${specBadge}
        ${metrics.peak_memory_gb ? `
        <div class="metric" data-tooltip="Peak Unified Memory (RAM+VRAM) on Apple Silicon">
            <span class="metric-icon">💾</span>
            <span>Memory:</span>
            <span class="metric-val">${metrics.peak_memory_gb} GB</span>
        </div>` : ''}
    `;
}

/**
 * Renders all messages of a given session in the chat view.
 * @param {string} sessionId
 */
export function renderSessionMessages(sessionId) {
    if (!chatContainer) return;
    chatContainer.innerHTML = "";

    const session = chatSessions.find(s => s.id === sessionId);
    if (!session || !session.messages || session.messages.length === 0) {
        const welcomeDiv = document.createElement("div");
        welcomeDiv.className = "message assistant";
        if (activeModel) {
            const isImg = isImageModelName(activeModel);
            welcomeDiv.innerHTML = isImg
                ? `Ready to generate visuals with <strong>${escapeHtml(activeModel)}</strong>! Type a prompt or enter <code>/image &lt;prompt&gt;</code>.`
                : `Ready for conversation with <strong>${escapeHtml(activeModel)}</strong>. Ask anything!`;
        } else {
            welcomeDiv.innerHTML = `Hello! Load a model from the sidebar to begin testing. TPM-MLX utilizes pre-allocated static KV caches, native MTP heads, and optimized zero-sync pipelines to achieve maximum tokens-per-second.`;
        }
        chatContainer.appendChild(welcomeDiv);
        return;
    }

    for (const msg of session.messages) {
        if (msg.role === "user") {
            const userMsgDiv = document.createElement("div");
            userMsgDiv.className = "message user";

            if (msg.referenceImage) {
                const refThumb = document.createElement("img");
                refThumb.src = msg.referenceImage;
                refThumb.style.maxWidth = "160px";
                refThumb.style.maxHeight = "160px";
                refThumb.style.borderRadius = "8px";
                refThumb.style.display = "block";
                refThumb.style.marginBottom = "8px";
                refThumb.style.border = "1px solid rgba(255, 255, 255, 0.2)";
                userMsgDiv.appendChild(refThumb);
            }

            const textSpan = document.createElement("span");
            let userText = "";
            if (typeof msg.content === "string") {
                userText = msg.content;
            } else if (Array.isArray(msg.content)) {
                const txtObj = msg.content.find(p => p.type === "text");
                userText = txtObj ? txtObj.text : "";
            }
            textSpan.textContent = userText;
            userMsgDiv.appendChild(textSpan);
            chatContainer.appendChild(userMsgDiv);
        } else if (msg.role === "assistant") {
            const assistantMsgDiv = document.createElement("div");
            assistantMsgDiv.className = "message assistant";

            const contentDiv = document.createElement("div");
            contentDiv.className = "message-body";

            if (msg.isImageGen && msg.imageUrl) {
                contentDiv.innerHTML = buildImageCardHtml({
                    url: msg.imageUrl,
                    imagePrompt: msg.imagePrompt || "",
                    revisedPrompt: msg.revisedPrompt || "",
                    supportsEdit: msg.supportsEdit !== undefined ? msg.supportsEdit : true
                });
            } else {
                contentDiv.innerHTML = formatMessageHtml(msg.content || "");
            }
            assistantMsgDiv.appendChild(contentDiv);

            if (msg.metrics) {
                const metricsDiv = document.createElement("div");
                metricsDiv.className = "metrics-bar";
                metricsDiv.innerHTML = msg.isImageGen
                    ? buildImageMetricsHtml(msg.metrics, msg.modelDisplayName)
                    : buildChatMetricsHtml(msg.metrics, msg.usage);
                assistantMsgDiv.appendChild(metricsDiv);
            }

            chatContainer.appendChild(assistantMsgDiv);
        }
    }

    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Fetches available models and populates dropdowns.
 */
export async function fetchModels() {
    try {
        const data = await fetchModelsApi();
        cachedModelsList = data.data || [];
        activeModel = data.active_model || "";
        activeDraftModel = data.active_draft_model || null;
        activeImageModel = data.active_image_model || null;

        modelSelect.innerHTML = "";
        draftSelect.innerHTML = `
            <option value="__AUTO__">✨ Auto-Detect Companion</option>
            <option value="__NONE__">🚫 None (Single-Token)</option>
        `;
        
        const textOptGroup = document.createElement("optgroup");
        textOptGroup.label = "💬 Text & Vision LLMs";

        const imageOptGroup = document.createElement("optgroup");
        imageOptGroup.label = "🎨 Image Generation Models";
        
        const draftOptGroup = document.createElement("optgroup");
        draftOptGroup.label = "⚡ Available Draft / MTP Assistants";

        let activeFound = false;

        cachedModelsList.forEach(m => {
            const isImage = m.is_image || isImageModelName(m.id);
            const isDraft = !isImage && (m.is_draft || m.id.toLowerCase().includes("assistant") || m.id.toLowerCase().includes("-mtp"));
            
            if (isDraft) {
                const opt = document.createElement("option");
                opt.value = m.id;
                opt.title = m.id;
                const isDraftActive = !isImageModelName(activeModel) && (activeDraftModel === m.id || (m.active && m.active_type === "draft"));
                opt.textContent = formatModelDisplayLabel(m.id, isDraftActive);
                if (isDraftActive) opt.selected = true;
                draftOptGroup.appendChild(opt);
            } else if (isImage) {
                const opt = document.createElement("option");
                opt.value = m.id;
                opt.title = m.id;
                const isImgActive = (activeModel === m.id);
                opt.textContent = formatModelDisplayLabel(m.id, isImgActive);
                if (activeModel === m.id) {
                    opt.selected = true;
                    activeFound = true;
                }
                imageOptGroup.appendChild(opt);
            } else {
                const opt = document.createElement("option");
                opt.value = m.id;
                opt.title = m.id;
                const isTxtActive = (activeModel === m.id);
                opt.textContent = formatModelDisplayLabel(m.id, isTxtActive);
                if (activeModel === m.id) {
                    opt.selected = true;
                    activeFound = true;
                    if (m.max_kv_size) kvSizeInput.value = m.max_kv_size;
                }
                textOptGroup.appendChild(opt);
            }
        });

        const customBaseOpt = document.createElement("option");
        customBaseOpt.value = "__CUSTOM__";
        customBaseOpt.textContent = "✏️ Enter Custom Model HF ID...";
        modelSelect.appendChild(textOptGroup);
        modelSelect.appendChild(imageOptGroup);
        modelSelect.appendChild(customBaseOpt);

        const customDraftOpt = document.createElement("option");
        customDraftOpt.value = "__CUSTOM__";
        customDraftOpt.textContent = "✏️ Enter Custom Draft HF ID...";
        draftOptGroup.appendChild(customDraftOpt);
        draftSelect.appendChild(draftOptGroup);

        // Explicitly set modelSelect.value and draftSelect.value after DOM elements are appended
        if (activeModel) {
            modelSelect.value = activeModel;
        }
        if (isImageModelName(activeModel)) {
            draftSelect.value = "__NONE__";
        } else if (activeDraftModel) {
            draftSelect.value = activeDraftModel;
        } else {
            draftSelect.value = "__AUTO__";
        }

        handleModelSelectionChange();

        if (data.is_loading) {
            const elapsedStr = data.loading_elapsed_s ? ` (${data.loading_elapsed_s}s)` : "";
            updateServerStatus("loading", "#fbbf24", `Loading Model${elapsedStr}...`);
            loadModelBtn.disabled = true;
            loadModelBtn.textContent = "Loading Model...";
            currentModelHeader.textContent = `Loading ${data.loading_model || "Model"}...`;
            setTimeout(fetchModels, 2000);
        } else if (activeFound || activeModel) {
            loadModelBtn.disabled = false;
            loadModelBtn.textContent = "Load Model";
            updateHeaderBadge(activeModel, activeDraftModel, data.speculation_mode, data.has_mtp, data.num_draft_tokens, data.backend);
            userInput.disabled = false;
            sendBtn.disabled = false;
            const isImg = (data.backend === "image") || isImageModelName(activeModel);
            userInput.placeholder = isImg ? "Type an image prompt to generate..." : (isMobileView() ? "Type message or /image..." : "Type a message, or /image <prompt>... (Enter to send)");
            updateServerStatus("connected", "#10b981", "Server Connected");
        } else {
            loadModelBtn.disabled = false;
            loadModelBtn.textContent = "Load Model";
            currentModelHeader.textContent = "Select and Load a Model";
            updateServerStatus("connected", "#10b981", "Server Connected");
        }
    } catch (err) {
        console.error("Error loading models:", err);
        updateServerStatus("failed", "#ef4444", "Connection Failed");
    }
}

/**
 * Handles Model Loading submission
 */
async function handleLoadModel() {
    let selectedModel = modelSelect.value;
    if (selectedModel === "__CUSTOM__") {
        selectedModel = customModelInput.value.trim();
    }
    if (!selectedModel) {
        alert("Please select or enter a base model name.");
        return;
    }

    const isImage = isImageModelName(selectedModel);

    let selectedDraft = null;
    if (!isImage) {
        selectedDraft = draftSelect.value;
        if (selectedDraft === "__CUSTOM__") {
            selectedDraft = customDraftInput.value.trim();
        } else if (selectedDraft === "__NONE__") {
            selectedDraft = null;
        } else if (selectedDraft === "__AUTO__") {
            const lower = selectedModel.toLowerCase();
            const cleanBase = lower.replace(/^.*?\//, "").replace(/-4bit|-8bit|-bf16|-fp16|-it|-instruct/g, "");
            let autoFound = null;
            for (let i = 0; i < draftSelect.options.length; i++) {
                const optVal = draftSelect.options[i].value;
                const optLower = optVal.toLowerCase();
                if (optVal === "__AUTO__" || optVal === "__NONE__" || optVal === "__CUSTOM__") continue;
                const cleanDraft = optLower.replace(/^.*?\//, "").replace(/-4bit|-8bit|-bf16|-fp16|-it|-instruct/g, "");
                const draftWithoutTag = cleanDraft.replace(/-mtp|-assistant|_mtp|_assistant/g, "");
                if (draftWithoutTag === cleanBase || cleanDraft.includes(cleanBase)) {
                    autoFound = optVal;
                    break;
                }
                const baseTokens = cleanBase.split(/[-_]/).filter(t => t.length > 0);
                const draftTokens = optLower.split(/[-_/]/).filter(t => t.length > 0);
                if (baseTokens.every(tok => draftTokens.includes(tok)) && draftTokens.some(t => t.includes("mtp") || t.includes("assistant"))) {
                    autoFound = optVal;
                    break;
                }
            }
            selectedDraft = autoFound;
        }

        // Cross-family mismatch guard to prevent invalid drafter pairing crashes
        if (selectedDraft && selectedDraft !== "__NONE__" && selectedDraft !== "__AUTO__") {
            const baseLower = selectedModel.toLowerCase();
            const draftLower = selectedDraft.toLowerCase();
            if (baseLower.includes("qwen") && !draftLower.includes("qwen")) {
                console.warn("Mismatched draft assistant detected for Qwen base, resetting draft to null");
                selectedDraft = null;
            } else if (baseLower.includes("gemma") && !draftLower.includes("gemma")) {
                console.warn("Mismatched draft assistant detected for Gemma base, resetting draft to null");
                selectedDraft = null;
            } else if (baseLower.includes("bonsai") && (draftLower.includes("gemma") || draftLower.includes("qwen"))) {
                console.warn("Mismatched draft assistant detected for Bonsai base, resetting draft to null");
                selectedDraft = null;
            }
        }
    }

    const kvSize = parseInt(kvSizeInput.value) || 4096;
    const draftTokensVal = draftTokensSelect.value ? parseInt(draftTokensSelect.value) : null;

    loadModelBtn.disabled = true;
    loadModelBtn.textContent = "Loading Model...";
    updateServerStatus("loading", "#fbbf24", "Loading Model...");

    try {
        const res = await loadModelApi({
            model: selectedModel,
            draft_model: isImage ? null : selectedDraft,
            max_kv_size: isImage ? null : kvSize,
            num_draft_tokens: isImage ? null : draftTokensVal,
            enable_mtp: !isImage,
        });

        activeModel = res.model || selectedModel;
        activeDraftModel = res.draft_model;
        updateHeaderBadge(activeModel, activeDraftModel, res.speculation_mode, res.has_mtp, res.num_draft_tokens, res.backend);
        updateServerStatus("loaded", "#10b981", "Model Loaded");
        userInput.disabled = false;
        sendBtn.disabled = false;
        if (isMobileView()) closeSidebar();

        const isImg = res.is_image || (res.backend === "image") || isImageModelName(activeModel);
        userInput.placeholder = isImg 
            ? "Type an image prompt to generate... (e.g. 'A futuristic city in the clouds')" 
            : (isMobileView() ? "Type message or /image..." : "Type a message, or /image <prompt>... (Enter to send)");

        const currentSession = getActiveSession();
        if (currentSession) {
            currentSession.model = activeModel;
            if (isImg && currentSession.messages.length === 0) {
                currentSession.type = "image";
            }
            saveSessionsState();
            renderSessionsList();

            if (currentSession.messages.length === 0) {
                renderSessionMessages(currentSession.id);
            } else {
                const specInfo = isImg 
                    ? "for Flow Matching Image Generation" 
                    : (res.speculation_mode === "mtp" 
                        ? `with MTP Acceleration (${res.num_draft_tokens || 3} tokens)` 
                        : (res.draft_model ? `with draft assistant ${res.draft_model}` : "in standard single-token mode"));
                const noticeDiv = document.createElement("div");
                noticeDiv.className = "message assistant";
                noticeDiv.style.opacity = "0.85";
                noticeDiv.innerHTML = `Switched model to <strong>${escapeHtml(activeModel)}</strong> ${specInfo}.`;
                chatContainer.appendChild(noticeDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
    } catch (err) {
        alert("Error loading model: " + err.message);
        updateServerStatus("failed", "#ef4444", "Load Failed");
    } finally {
        loadModelBtn.disabled = false;
        loadModelBtn.textContent = "Load Model";
        fetchModels();
    }
}

/**
 * Handles sending user prompts and receiving stream responses / images
 */
export async function sendMessage() {
    const userText = userInput.value.trim();
    if (!userText || userInput.disabled) return;

    let session = getActiveSession();
    if (!session) {
        session = createNewSession(activeModel, false);
    }

    const userMsgDiv = document.createElement("div");
    userMsgDiv.className = "message user";
    
    const currentRefUrl = attachedReferenceImageUrl;
    if (currentRefUrl) {
        const refThumb = document.createElement("img");
        refThumb.src = currentRefUrl;
        refThumb.style.maxWidth = "160px";
        refThumb.style.maxHeight = "160px";
        refThumb.style.borderRadius = "8px";
        refThumb.style.display = "block";
        refThumb.style.marginBottom = "8px";
        refThumb.style.border = "1px solid rgba(255, 255, 255, 0.2)";
        userMsgDiv.appendChild(refThumb);
    }

    const textSpan = document.createElement("span");
    textSpan.textContent = userText;
    userMsgDiv.appendChild(textSpan);
    chatContainer.appendChild(userMsgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    userInput.value = "";
    userInput.style.height = "";
    userInput.disabled = true;
    sendBtn.disabled = true;

    const isCurrentImageModel = isImageModelName(activeModel) || (currentModelHeader.textContent && isImageModelName(currentModelHeader.textContent));
    const isExplicitImageCmd = userText.startsWith("/image") || userText.startsWith("/edit");
    const routeToImageEngine = isExplicitImageCmd || isCurrentImageModel;

    let userContentPayload = userText;
    if (!routeToImageEngine && currentRefUrl) {
        const fullUrl = currentRefUrl.startsWith("http") || currentRefUrl.startsWith("data:")
            ? currentRefUrl
            : `${window.location.origin}${currentRefUrl}`;
        userContentPayload = [
            { type: "text", text: userText },
            { type: "image_url", image_url: { url: fullUrl } }
        ];
        clearAttachedReferenceImage();
    }

    session.messages.push({
        role: "user",
        content: userContentPayload,
        referenceImage: currentRefUrl || null,
        timestamp: Date.now()
    });
    autoTitleSession(session, userText, routeToImageEngine);
    saveSessionsState();
    renderSessionsList();

    const assistantMsgDiv = document.createElement("div");
    assistantMsgDiv.className = "message assistant";
    
    const contentDiv = document.createElement("div");
    contentDiv.className = "message-body";
    
    const loaderDiv = document.createElement("div");
    loaderDiv.className = "typing-loader";
    loaderDiv.innerHTML = "<span></span><span></span><span></span>";
    contentDiv.appendChild(loaderDiv);
    
    const metricsDiv = document.createElement("div");
    metricsDiv.className = "metrics-bar";
    metricsDiv.style.display = "none";
    
    assistantMsgDiv.appendChild(contentDiv);
    assistantMsgDiv.appendChild(metricsDiv);
    chatContainer.appendChild(assistantMsgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    if (routeToImageEngine) {
        const isEdit = !!currentRefUrl || userText.startsWith("/edit");
        clearAttachedReferenceImage();

        let imagePrompt = userText.startsWith("/image") ? userText.slice(6).trim() : (userText.startsWith("/edit") ? userText.slice(5).trim() : userText.trim());
        if (!imagePrompt) imagePrompt = isEdit ? "Refine and enhance image details and lighting" : "A beautiful scenic view on Apple Silicon";

        loaderDiv.innerHTML = isEdit 
            ? `<span style="font-size: 13px; color: var(--text-secondary); display: inline-flex; align-items: center; gap: 8px;">✏️ <span>Modifying image with FLUX Apple Silicon MLX...</span></span>`
            : `<span style="font-size: 13px; color: var(--text-secondary); display: inline-flex; align-items: center; gap: 8px;">🎨 <span>Generating image with Apple Silicon MLX...</span></span>`;

        try {
            const recentContext = (session.messages || [])
                .slice(-6, -1)
                .map(m => `${m.role}: ${typeof m.content === 'string' ? m.content : JSON.stringify(m.content)}`)
                .join("\n");

            const resSelectEl = document.getElementById("image-resolution-select");
            const defaultRes = resSelectEl ? resSelectEl.value : "1024x1024";
            const finalResolution = extractResolution(imagePrompt, defaultRes);

            const isAutoExpand = autoExpandToggle ? autoExpandToggle.checked : false;
            const activeImg = (isImageModelName(activeModel) ? activeModel : null) || activeImageModel;
            const imgPayload = {
                prompt: imagePrompt,
                size: finalResolution,
                steps: 4,
                auto_expand: isAutoExpand,
                context: isAutoExpand ? recentContext : null,
                response_format: "url"
            };
            if (activeImg) imgPayload.model = activeImg;
            if (isEdit && currentRefUrl) {
                imgPayload.image = currentRefUrl;
                imgPayload.guidance = 2.5;
            }

            const imgData = await generateImageApi(imgPayload, isEdit);
            const item = imgData.data[0];
            const metrics = imgData.tpm_metrics || {};

            const supportsEdit = metrics.supports_editing !== undefined 
                ? metrics.supports_editing 
                : (metrics.model ? metrics.model.toLowerCase().includes("flux") : (activeModel || "").toLowerCase().includes("flux"));
            const modelDisplayName = (metrics.model || activeModel || "").split("/").pop();

            contentDiv.innerHTML = buildImageCardHtml({
                url: item.url,
                imagePrompt: imagePrompt,
                revisedPrompt: item.revised_prompt || "",
                supportsEdit: supportsEdit
            });

            metricsDiv.innerHTML = buildImageMetricsHtml(metrics, modelDisplayName);
            metricsDiv.style.display = "flex";
            chatContainer.scrollTop = chatContainer.scrollHeight;

            session.messages.push({
                role: "assistant",
                isImageGen: true,
                imageUrl: item.url,
                imagePrompt: imagePrompt,
                revisedPrompt: item.revised_prompt || "",
                supportsEdit: supportsEdit,
                modelDisplayName: modelDisplayName,
                metrics: metrics,
                timestamp: Date.now()
            });
            session.updatedAt = Date.now();
            saveSessionsState();
            renderSessionsList();

        } catch (err) {
            console.error("Image Generation Error:", err);
            contentDiv.innerHTML = "";
            const errSpan = document.createElement("span");
            errSpan.style.color = "#ef4444";
            errSpan.textContent = `Error generating image: ${err.message}`;
            contentDiv.appendChild(errSpan);
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            if (!isMobileView()) userInput.focus();
        }
        return;
    }

    // Text completion via SSE streaming
    const rollingMessages = (session.messages || [])
        .filter(m => !m.isImageGen)
        .slice(-10)
        .map(m => ({ role: m.role, content: m.content }));

    const payload = {
        model: activeModel,
        messages: rollingMessages,
        max_tokens: parseInt(tokensInput.value) || 4096,
        temperature: parseFloat(tempSlider.value),
        stream: true,
        reasoning: reasoningCheckbox.checked
    };

    let metricsRendered = false;

    try {
        contentDiv.innerHTML = ""; // Clear loader
        const result = await streamChatCompletion(
            payload,
            (accumulatedText) => {
                contentDiv.innerHTML = formatMessageHtml(accumulatedText);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            },
            (metrics, usage) => {
                if (!metricsRendered) {
                    metricsRendered = true;
                    metricsDiv.innerHTML = buildChatMetricsHtml(metrics, usage);
                    metricsDiv.style.display = "flex";
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }
        );

        session.messages.push({
            role: "assistant",
            content: result.text,
            metrics: result.metrics,
            usage: result.usage,
            timestamp: Date.now()
        });
        session.updatedAt = Date.now();
        saveSessionsState();
        renderSessionsList();

    } catch (err) {
        console.error("Inference Error:", err);
        contentDiv.innerHTML = "";
        const errSpan = document.createElement("span");
        errSpan.style.color = "#ef4444";
        errSpan.textContent = `Error: ${err.message}`;
        contentDiv.appendChild(errSpan);
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        if (!isMobileView()) userInput.focus();
    }
}

/**
 * Expose global window attributes for HTML string event handlers
 */
function registerWindowHandlers() {
    window.openImageLightbox = openImageLightbox;
    window.closeImageLightbox = closeImageLightbox;
    window.copyImageLink = copyImageLink;
    window.handleUnsupportedModify = handleUnsupportedModify;
    window.toggleThought = toggleThought;
    window.switchSession = switchSession;
    window.deleteSession = deleteSession;
    window.attachReferenceImage = attachReferenceImage;
    window.clearAttachedReferenceImage = clearAttachedReferenceImage;
}

/**
 * Initialize event listeners across playground DOM
 */
function initEventListeners() {
    // Theme & Lightbox
    initTheme();
    initLightbox();
    registerWindowHandlers();

    // Sidebar navigation
    if (sidebarToggleBtn) sidebarToggleBtn.addEventListener("click", toggleSidebar);
    if (sidebarCloseBtn) sidebarCloseBtn.addEventListener("click", closeSidebar);
    if (sidebarBackdrop) sidebarBackdrop.addEventListener("click", closeSidebar);

    window.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && isMobileView() && document.body.classList.contains("sidebar-open")) {
            closeSidebar();
        }
    });

    window.addEventListener("resize", () => {
        if (!isMobileView()) {
            document.body.classList.remove("sidebar-open");
        } else {
            document.body.classList.remove("sidebar-collapsed");
        }
        if (activeModel) {
            updateHeaderBadge(activeModel, activeDraftModel, window._lastSpecMode, window._lastHasMtp, window._lastDraftTokens, window._lastBackend);
        }
    });

    try {
        if (!isMobileView() && localStorage.getItem("tpm_mlx_sidebar_collapsed") === "true") {
            document.body.classList.add("sidebar-collapsed");
        }
    } catch(e) {}

    // Model selection dropdowns & custom inputs
    modelSelect.addEventListener("change", () => {
        if (modelSelect.value === "__CUSTOM__") {
            customModelGroup.style.display = "flex";
        } else {
            customModelGroup.style.display = "none";
        }
        handleModelSelectionChange();
    });

    if (customModelInput) {
        customModelInput.addEventListener("input", () => {
            handleModelSelectionChange();
        });
    }

    draftSelect.addEventListener("change", () => {
        if (draftSelect.value === "__CUSTOM__") {
            customDraftGroup.style.display = "flex";
        } else {
            customDraftGroup.style.display = "none";
        }
    });

    tempSlider.addEventListener("input", (e) => {
        tempVal.textContent = parseFloat(e.target.value).toFixed(1);
    });

    if (autoExpandToggle && directorDesc) {
        autoExpandToggle.addEventListener("change", () => {
            if (autoExpandToggle.checked) {
                directorDesc.innerHTML = 'Enriches short prompts with cinematography details. Toggle OFF for <b>Direct Mode</b> (minimal VRAM & max speed).';
                directorDesc.style.color = 'var(--text-secondary)';
            } else {
                directorDesc.innerHTML = '⚡ <b>Direct Mode (Minimal VRAM)</b>: Text LLM bypassed. Generates directly from raw prompt for lowest memory.';
                directorDesc.style.color = '#38bdf8';
            }
        });
    }

    if (uploadBtn && imageUploadInput) {
        uploadBtn.addEventListener("click", () => imageUploadInput.click());
        imageUploadInput.addEventListener("change", async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            uploadBtn.disabled = true;
            try {
                const data = await uploadImageApi(file);
                attachReferenceImage(data.url, file.name);
            } catch (err) {
                alert("Image upload failed: " + err.message);
            } finally {
                uploadBtn.disabled = false;
                imageUploadInput.value = "";
            }
        });
    }

    loadModelBtn.addEventListener("click", handleLoadModel);

    reasoningCheckbox.addEventListener("change", updateReasoningBadge);
    reasoningContainer.addEventListener("click", (e) => {
        if (e.target.closest(".switch")) return;
        reasoningCheckbox.checked = !reasoningCheckbox.checked;
        reasoningCheckbox.dispatchEvent(new Event("change"));
    });
    updateReasoningBadge();

    // Auto-growing textarea
    userInput.addEventListener("input", () => {
        userInput.style.height = "auto";
        userInput.style.height = Math.min(userInput.scrollHeight, 140) + "px";
    });

    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    sendBtn.addEventListener("click", sendMessage);

    // Session actions
    if (newChatBtn) newChatBtn.addEventListener("click", () => createNewSession(activeModel, true));
    if (headerNewChatBtn) headerNewChatBtn.addEventListener("click", () => createNewSession(activeModel, true));
    if (exportChatBtn) exportChatBtn.addEventListener("click", exportCurrentChat);
    if (clearSessionsBtn) clearSessionsBtn.addEventListener("click", clearAllSessions);

    // Global Cmd+K / Ctrl+K shortcut for New Chat
    window.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
            e.preventDefault();
            createNewSession(activeModel, true);
        }
    });

    // Session Switch / Render callback registration
    registerSessionCallbacks({
        onSwitch: (sessionId, isCurrent) => {
            renderSessionMessages(sessionId);
            if (isMobileView()) closeSidebar();
            if (userInput && !isMobileView()) userInput.focus();
        },
        onRender: () => {
            renderSessionsList();
        }
    });
}

// Bootstrap on DOM ready
document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    loadSessionsState(activeModel);
    fetchModels();
});
