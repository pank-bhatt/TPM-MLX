/**
 * TPM-MLX API Client
 * Wraps backend HTTP and Server-Sent Event (SSE) streaming endpoints.
 */

export const API_URL = ""; // Relative path to match hosted server

/**
 * Fetches all available and active models from the server.
 * @returns {Promise<any>}
 */
export async function fetchModelsApi() {
    const resp = await fetch(`${API_URL}/v1/models`);
    if (!resp.ok) {
        throw new Error(`Failed to fetch models: HTTP ${resp.status}`);
    }
    return await resp.json();
}

/**
 * Requests backend to load a model and optional companion draft assistant.
 * @param {object} payload
 * @returns {Promise<any>}
 */
export async function loadModelApi(payload) {
    const resp = await fetch(`${API_URL}/v1/load_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    const res = await resp.json();
    if (!resp.ok) {
        throw new Error(res.detail || JSON.stringify(res));
    }
    return res;
}

/**
 * Uploads a reference image to the server for multimodal VLM or FLUX editing.
 * @param {File} file
 * @returns {Promise<{url: string, filename: string}>}
 */
export async function uploadImageApi(file) {
    const formData = new FormData();
    formData.append("file", file);
    const resp = await fetch(`${API_URL}/v1/upload_image`, {
        method: "POST",
        body: formData
    });
    if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Failed to upload image");
    }
    return await resp.json();
}

/**
 * Generates or edits an image via diffusion flow matching.
 * @param {object} payload
 * @param {boolean} isEdit
 * @returns {Promise<any>}
 */
export async function generateImageApi(payload, isEdit = false) {
    const endpoint = isEdit ? `${API_URL}/v1/images/edits` : `${API_URL}/v1/images/generations`;
    const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    if (!resp.ok) {
        const errObj = await resp.json();
        throw new Error(errObj.detail || "Image operation failed");
    }
    return await resp.json();
}

/**
 * Streams chat completion chunks via SSE reader.
 * @param {object} payload
 * @param {function(string): void} onDelta
 * @param {function(object, object): void} onMetrics
 * @returns {Promise<{text: string, metrics: object, usage: object}>}
 */
export async function streamChatCompletion(payload, onDelta, onMetrics) {
    const response = await fetch(`${API_URL}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        const errObj = await response.json();
        throw new Error(errObj.detail || `Server error: HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let accumulatedText = "";
    let lastMetrics = null;
    let lastUsage = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // Keep last incomplete line

        for (let line of lines) {
            line = line.trim();
            if (!line.startsWith("data: ")) continue;

            const dataStr = line.slice(6);
            if (dataStr === "[DONE]") continue;

            try {
                const chunk = JSON.parse(dataStr);
                if (chunk.choices && chunk.choices[0] && chunk.choices[0].delta && chunk.choices[0].delta.content) {
                    const delta = chunk.choices[0].delta.content;
                    accumulatedText += delta;
                    if (onDelta) onDelta(accumulatedText, delta);
                }

                if (chunk.tpm_metrics) {
                    lastMetrics = chunk.tpm_metrics;
                }
                if (chunk.usage) {
                    lastUsage = chunk.usage;
                }

                if (chunk.tpm_metrics && onMetrics) {
                    onMetrics(chunk.tpm_metrics, chunk.usage);
                }
            } catch (e) {
                console.warn("Could not parse stream JSON chunk:", e, dataStr);
            }
        }
    }

    return {
        text: accumulatedText,
        metrics: lastMetrics,
        usage: lastUsage
    };
}
