/**
 * TPM-MLX Image Lightbox & Studio Viewer
 * Full-screen modal display, zoom inspection, direct downloading,
 * and link copying for generated visuals.
 */

let modalEl = null;
let backdropEl = null;
let imageEl = null;
let captionEl = null;
let downloadBtnEl = null;
let openTabBtnEl = null;
let closeBtnEl = null;
let contentEl = null;

/**
 * Opens full-screen lightbox for an image.
 * @param {string} src
 * @param {string} caption
 */
export function openImageLightbox(src, caption = "") {
    if (!modalEl || !imageEl) {
        modalEl = document.getElementById("image-lightbox-modal");
        imageEl = document.getElementById("lightbox-image");
        captionEl = document.getElementById("lightbox-caption");
        downloadBtnEl = document.getElementById("lightbox-download-btn");
        openTabBtnEl = document.getElementById("lightbox-open-tab-btn");
        contentEl = document.getElementById("lightbox-content");
    }
    if (!modalEl || !imageEl) return;

    imageEl.src = src;
    imageEl.alt = caption || "Generated Image";
    imageEl.classList.remove("zoomed");
    if (contentEl) {
        contentEl.scrollTop = 0;
        contentEl.scrollLeft = 0;
    }

    if (captionEl) {
        captionEl.textContent = caption || "";
        captionEl.title = caption || "";
        captionEl.style.display = caption ? "block" : "none";
    }

    if (downloadBtnEl) {
        downloadBtnEl.href = src;
    }
    if (openTabBtnEl) {
        openTabBtnEl.href = src;
    }

    modalEl.classList.add("active");
    modalEl.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
}

/**
 * Closes the image lightbox modal.
 */
export function closeImageLightbox() {
    if (!modalEl) {
        modalEl = document.getElementById("image-lightbox-modal");
        imageEl = document.getElementById("lightbox-image");
    }
    if (!modalEl) return;

    modalEl.classList.remove("active");
    modalEl.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    if (imageEl) {
        imageEl.classList.remove("zoomed");
    }
}

/**
 * Copies the full URL of the generated image to clipboard.
 * @param {HTMLElement} btn
 * @param {string} relUrl
 */
export function copyImageLink(btn, relUrl) {
    const fullUrl = window.location.origin + relUrl;
    navigator.clipboard.writeText(fullUrl).then(() => {
        const origHtml = btn.innerHTML;
        btn.innerHTML = "✅ Link Copied!";
        btn.style.color = "var(--primary)";
        btn.style.borderColor = "var(--primary)";
        setTimeout(() => {
            btn.innerHTML = origHtml;
            btn.style.color = "";
            btn.style.borderColor = "";
        }, 2000);
    }).catch(() => {
        prompt("Direct Image Link:", fullUrl);
    });
}

/**
 * Displays notice when attempting image-to-image on an unsupported model.
 */
export function handleUnsupportedModify() {
    alert("Image modification (Image-to-Image editing) requires Qwen-Image-2.1 (e.g. OsaurusAI/Qwen-Image-2.1-Turbo-mflux-4bit) or FLUX.2 Klein (e.g. mlx-community/flux2-klein-9b-4bit). Please load one from the sidebar to edit images.");
}

let model3dModalEl = null;
let model3dBackdropEl = null;
let model3dViewerEl = null;
let model3dCaptionEl = null;
let model3dDownloadBtnEl = null;
let model3dCloseBtnEl = null;

export function open3DLightbox(glbUrl, caption = "🧊 Watertight 3D Geometry") {
    if (!model3dModalEl) {
        model3dModalEl = document.getElementById("model-3d-modal");
        model3dBackdropEl = document.getElementById("model-3d-backdrop");
        model3dViewerEl = document.getElementById("model-3d-viewer");
        model3dCaptionEl = document.getElementById("model-3d-caption");
        model3dDownloadBtnEl = document.getElementById("model-3d-download-btn");
        model3dCloseBtnEl = document.getElementById("model-3d-close-btn");
    }
    if (!model3dModalEl || !model3dViewerEl) return;

    model3dViewerEl.setAttribute("src", glbUrl);
    if (model3dCaptionEl) model3dCaptionEl.textContent = caption;
    if (model3dDownloadBtnEl) model3dDownloadBtnEl.href = glbUrl;

    model3dModalEl.style.display = "flex";
    model3dModalEl.classList.add("active");
    model3dModalEl.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
}

export function close3DLightbox() {
    if (!model3dModalEl) {
        model3dModalEl = document.getElementById("model-3d-modal");
        model3dViewerEl = document.getElementById("model-3d-viewer");
    }
    if (!model3dModalEl) return;
    model3dModalEl.classList.remove("active");
    model3dModalEl.setAttribute("aria-hidden", "true");
    model3dModalEl.style.display = "none";
    document.body.style.overflow = "";
    if (model3dViewerEl) {
        model3dViewerEl.removeAttribute("src");
    }
}

/**
 * Initializes lightbox DOM elements and event bindings.
 */
export function initLightbox() {
    modalEl = document.getElementById("image-lightbox-modal");
    backdropEl = document.getElementById("lightbox-backdrop");
    imageEl = document.getElementById("lightbox-image");
    captionEl = document.getElementById("lightbox-caption");
    downloadBtnEl = document.getElementById("lightbox-download-btn");
    openTabBtnEl = document.getElementById("lightbox-open-tab-btn");
    closeBtnEl = document.getElementById("lightbox-close-btn");
    contentEl = document.getElementById("lightbox-content");

    model3dModalEl = document.getElementById("model-3d-modal");
    model3dBackdropEl = document.getElementById("model-3d-backdrop");
    model3dViewerEl = document.getElementById("model-3d-viewer");
    model3dCaptionEl = document.getElementById("model-3d-caption");
    model3dDownloadBtnEl = document.getElementById("model-3d-download-btn");
    model3dCloseBtnEl = document.getElementById("model-3d-close-btn");

    if (backdropEl) backdropEl.addEventListener("click", closeImageLightbox);
    if (closeBtnEl) closeBtnEl.addEventListener("click", closeImageLightbox);

    if (model3dBackdropEl) model3dBackdropEl.addEventListener("click", close3DLightbox);
    if (model3dCloseBtnEl) model3dCloseBtnEl.addEventListener("click", close3DLightbox);

    if (contentEl) {
        contentEl.addEventListener("click", (e) => {
            if (e.target === contentEl) {
                closeImageLightbox();
            }
        });
    }

    if (imageEl) {
        imageEl.addEventListener("click", (e) => {
            e.stopPropagation();
            imageEl.classList.toggle("zoomed");
        });
    }

    window.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            if (modalEl && modalEl.classList.contains("active")) {
                closeImageLightbox();
            }
            if (model3dModalEl && model3dModalEl.classList.contains("active")) {
                close3DLightbox();
            }
        }
    });

    // Delegated click listener for all generated images
    document.addEventListener("click", (e) => {
        const imgTarget = e.target.closest(".generated-image");
        if (imgTarget && !e.target.closest("#image-lightbox-modal")) {
            const src = imgTarget.getAttribute("src");
            const caption = imgTarget.getAttribute("alt") || "";
            if (src) {
                openImageLightbox(src, caption);
            }
        }
    });

    // Expose on window for inline handlers
    window.open3DLightbox = open3DLightbox;
    window.close3DLightbox = close3DLightbox;
}
