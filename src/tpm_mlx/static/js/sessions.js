/**
 * TPM-MLX Persistent Multi-Session Chat & Visual History Store
 * Manages multi-chat storage in localStorage, auto-titling, session switching,
 * markdown exports, and sidebar sessions drawer rendering.
 */

import { escapeHtml } from "./markdown.js";

export const SESSIONS_STORAGE_KEY = "tpm_mlx_chat_sessions_v1";
export const MAX_SESSIONS = 40;

export let chatSessions = [];
export let activeSessionId = null;

// Callbacks registered by coordinator
let onSessionSwitchCallback = null;
let onSessionRenderCallback = null;

export function registerSessionCallbacks({ onSwitch, onRender }) {
    if (onSwitch) onSessionSwitchCallback = onSwitch;
    if (onRender) onSessionRenderCallback = onRender;
}

/**
 * Formats relative time (e.g., "Just now", "5m ago", "2h ago", "Yesterday").
 * @param {number} timestamp
 * @returns {string}
 */
export function formatRelativeTime(timestamp) {
    if (!timestamp) return "";
    const diffMs = Date.now() - timestamp;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHr = Math.floor(diffMin / 60);
    const diffDays = Math.floor(diffHr / 24);

    if (diffSec < 45) return "Just now";
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHr < 24) return `${diffHr}h ago`;
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays}d ago`;
    const d = new Date(timestamp);
    return `${d.getMonth() + 1}/${d.getDate()}`;
}

/**
 * Returns the currently active session object or null.
 * @returns {object | null}
 */
export function getActiveSession() {
    if (!activeSessionId) return null;
    return chatSessions.find(s => s.id === activeSessionId) || null;
}

/**
 * Persists chat sessions state into localStorage.
 */
export function saveSessionsState() {
    try {
        if (chatSessions.length > MAX_SESSIONS) {
            chatSessions = chatSessions.slice(0, MAX_SESSIONS);
        }
        const data = {
            activeSessionId,
            sessions: chatSessions
        };
        localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(data));
    } catch (err) {
        console.warn("Storage quota warning, pruning old sessions:", err);
        try {
            chatSessions = chatSessions.slice(0, 15);
            const data = { activeSessionId, sessions: chatSessions };
            localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(data));
        } catch (e) {
            console.error("Failed to save sessions to localStorage:", e);
        }
    }
}

/**
 * Loads session state from localStorage and restores or creates active session.
 * @param {string} currentActiveModel
 */
export function loadSessionsState(currentActiveModel = "") {
    try {
        const raw = localStorage.getItem(SESSIONS_STORAGE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw);
            if (parsed && Array.isArray(parsed.sessions)) {
                chatSessions = parsed.sessions;
                activeSessionId = parsed.activeSessionId || null;
            }
        }
    } catch (err) {
        console.warn("Could not load sessions from localStorage:", err);
        chatSessions = [];
        activeSessionId = null;
    }

    if (chatSessions.length === 0) {
        createNewSession(currentActiveModel, false);
    } else {
        if (!activeSessionId || !chatSessions.some(s => s.id === activeSessionId)) {
            activeSessionId = chatSessions[0].id;
        }
        renderSessionsList();
        if (onSessionSwitchCallback) {
            onSessionSwitchCallback(activeSessionId);
        }
    }
}

/**
 * Creates a brand new chat session.
 * @param {string} activeModel
 * @param {boolean} switchToIt
 * @returns {object}
 */
export function createNewSession(activeModel = "", switchToIt = true) {
    const isImg = (activeModel || "").toLowerCase().includes("flux") || 
                  (activeModel || "").toLowerCase().includes("image") || 
                  (activeModel || "").toLowerCase().includes("diffusion");
    
    const newSession = {
        id: "session_" + Date.now() + "_" + Math.random().toString(36).substring(2, 7),
        title: "New Chat",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        model: activeModel || "",
        type: isImg ? "image" : "chat",
        messages: []
    };

    chatSessions.unshift(newSession);

    if (switchToIt) {
        activeSessionId = newSession.id;
        saveSessionsState();
        renderSessionsList();
        if (onSessionSwitchCallback) {
            onSessionSwitchCallback(newSession.id);
        }
    } else {
        activeSessionId = newSession.id;
        saveSessionsState();
        renderSessionsList();
        if (onSessionSwitchCallback) {
            onSessionSwitchCallback(newSession.id);
        }
    }
    return newSession;
}

/**
 * Switches to a different chat session.
 * @param {string} sessionId
 */
export function switchSession(sessionId) {
    if (sessionId === activeSessionId) {
        if (onSessionSwitchCallback) onSessionSwitchCallback(sessionId, true);
        return;
    }
    const target = chatSessions.find(s => s.id === sessionId);
    if (!target) return;

    activeSessionId = sessionId;
    saveSessionsState();
    renderSessionsList();
    if (onSessionSwitchCallback) {
        onSessionSwitchCallback(sessionId, false);
    }
}

/**
 * Deletes a session by ID.
 * @param {string} sessionId
 * @param {Event} [event]
 */
export function deleteSession(sessionId, event) {
    if (event) {
        event.stopPropagation();
        event.preventDefault();
    }

    const idx = chatSessions.findIndex(s => s.id === sessionId);
    if (idx === -1) return;

    chatSessions.splice(idx, 1);

    if (chatSessions.length === 0) {
        createNewSession("", true);
    } else if (activeSessionId === sessionId) {
        activeSessionId = chatSessions[0].id;
        saveSessionsState();
        renderSessionsList();
        if (onSessionSwitchCallback) {
            onSessionSwitchCallback(activeSessionId, false);
        }
    } else {
        saveSessionsState();
        renderSessionsList();
    }
}

/**
 * Deletes all sessions after user confirmation.
 */
export function clearAllSessions() {
    if (chatSessions.length === 0) return;
    if (!confirm("Are you sure you want to delete all chat history? This cannot be undone.")) {
        return;
    }
    chatSessions = [];
    try {
        localStorage.removeItem(SESSIONS_STORAGE_KEY);
    } catch (e) {}
    createNewSession("", true);
}

/**
 * Exports the active chat session as a Markdown document.
 */
export function exportCurrentChat() {
    const session = getActiveSession();
    if (!session || !session.messages || session.messages.length === 0) {
        alert("Current chat session is empty. Nothing to export.");
        return;
    }

    let md = `# TPM-MLX Session: ${session.title || "Chat"}\n\n`;
    md += `- **Date**: ${new Date(session.createdAt || Date.now()).toLocaleString()}\n`;
    if (session.model) md += `- **Model**: ${session.model}\n`;
    md += `\n---\n\n`;

    for (const msg of session.messages) {
        const timeStr = msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : "";
        const roleName = msg.role === "user" ? "👤 User" : "🤖 Assistant";
        md += `### ${roleName} ${timeStr ? `(${timeStr})` : ""}\n\n`;

        if (msg.isImageGen) {
            const fullImgUrl = msg.imageUrl && (msg.imageUrl.startsWith("http") ? msg.imageUrl : `${window.location.origin}${msg.imageUrl}`);
            md += `![Generated Image](${fullImgUrl})\n\n`;
            if (msg.imagePrompt) md += `*Prompt*: ${msg.imagePrompt}\n\n`;
            if (msg.revisedPrompt) md += `*Visual Prompt*: ${msg.revisedPrompt}\n\n`;
            if (msg.metrics) {
                md += `*Metrics*: Latency: ${msg.metrics.generation_time_s}s | Steps: ${msg.metrics.steps} | Size: ${msg.metrics.width}x${msg.metrics.height}\n\n`;
            }
        } else {
            let text = "";
            if (typeof msg.content === "string") {
                text = msg.content;
            } else if (Array.isArray(msg.content)) {
                const txtObj = msg.content.find(p => p.type === "text");
                text = txtObj ? txtObj.text : JSON.stringify(msg.content);
            }
            md += `${text}\n\n`;
            if (msg.metrics && msg.metrics.tps !== undefined) {
                md += `*Metrics*: Speed: ${msg.metrics.tps} tok/s | TTFT: ${msg.metrics.ttft_ms} ms\n\n`;
            }
        }
        md += `---\n\n`;
    }

    const safeTitle = (session.title || "chat_history").replace(/[^a-z0-9_-]/gi, "_").toLowerCase();
    const filename = `${safeTitle}_${Date.now()}.md`;
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Automatically creates a concise title from the user prompt.
 * @param {object} session
 * @param {string} userPrompt
 * @param {boolean} isImage
 */
export function autoTitleSession(session, userPrompt, isImage) {
    if (!session || (session.title !== "New Chat" && session.title !== "Untitled")) return;
    let clean = (userPrompt || "").trim();
    if (isImage) {
        if (clean.startsWith("/image")) clean = clean.slice(6).trim();
        if (clean.startsWith("/edit")) clean = clean.slice(5).trim();
        session.type = "image";
        session.title = clean ? (clean.length > 32 ? clean.slice(0, 32).trim() + "…" : clean) : "Generated Image";
    } else {
        session.type = "chat";
        clean = clean.replace(/^#+\s*/, '').replace(/\n+/g, ' ').trim();
        if (clean.length > 36) {
            const sub = clean.slice(0, 36);
            const lastSpace = sub.lastIndexOf(' ');
            clean = (lastSpace > 20 ? sub.slice(0, lastSpace) : sub).trim() + "…";
        }
        session.title = clean || "New Chat";
    }
    session.updatedAt = Date.now();
    saveSessionsState();
    renderSessionsList();
}

/**
 * Renders the session list items in the sidebar.
 */
export function renderSessionsList() {
    const sessionsListEl = document.getElementById("sessions-list");
    const sessionsCountBadge = document.getElementById("sessions-count-badge");
    if (!sessionsListEl) return;

    if (sessionsCountBadge) {
        sessionsCountBadge.textContent = chatSessions.length;
    }

    if (chatSessions.length === 0) {
        sessionsListEl.innerHTML = `<div class="sessions-empty">No previous chats</div>`;
        return;
    }

    sessionsListEl.innerHTML = chatSessions.map(session => {
        const isActive = session.id === activeSessionId;
        const icon = session.type === "image" ? "🎨" : "💬";
        const relTime = formatRelativeTime(session.updatedAt || session.createdAt);
        const titleSafe = escapeHtml(session.title || "Untitled");
        return `
            <div class="session-item ${isActive ? 'active' : ''}" data-session-id="${session.id}" onclick="switchSession('${session.id}')">
                <span class="session-item-icon">${icon}</span>
                <div class="session-item-content">
                    <span class="session-item-title" title="${titleSafe}">${titleSafe}</span>
                    <span class="session-item-time">${relTime}</span>
                </div>
                <button type="button" class="session-item-delete" title="Delete Chat" onclick="deleteSession('${session.id}', event)">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
        `;
    }).join("");
}
