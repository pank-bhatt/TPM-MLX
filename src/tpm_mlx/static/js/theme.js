/**
 * TPM-MLX Theme Engine
 * Manages Obsidian Dark and Porcelain Light glassmorphism themes
 */

const THEME_STORAGE_KEY = "tpm_mlx_theme";

/**
 * Returns saved theme or prefers-color-scheme setting.
 * @returns {"light" | "dark"}
 */
export function getPreferredTheme() {
    try {
        const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
        if (savedTheme === "light" || savedTheme === "dark") {
            return savedTheme;
        }
    } catch (e) {
        // LocalStorage might be restricted
    }
    return (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) ? "light" : "dark";
}

/**
 * Applies the specified theme to the document and updates UI icon states.
 * @param {"light" | "dark"} theme
 */
export function applyTheme(theme) {
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const themeIconSun = document.getElementById("theme-icon-sun");
    const themeIconMoon = document.getElementById("theme-icon-moon");

    if (theme === "light") {
        document.documentElement.setAttribute("data-theme", "light");
        if (themeIconSun) themeIconSun.style.display = "block";
        if (themeIconMoon) themeIconMoon.style.display = "none";
        if (themeToggleBtn) themeToggleBtn.title = "Switch to Dark Mode (Obsidian)";
    } else {
        document.documentElement.removeAttribute("data-theme");
        if (themeIconSun) themeIconSun.style.display = "none";
        if (themeIconMoon) themeIconMoon.style.display = "block";
        if (themeToggleBtn) themeToggleBtn.title = "Switch to Light Mode (Porcelain)";
    }
}

/**
 * Toggles between light and dark themes.
 */
export function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
    const newTheme = currentTheme === "light" ? "dark" : "light";
    try {
        localStorage.setItem(THEME_STORAGE_KEY, newTheme);
    } catch (e) {}
    applyTheme(newTheme);
}

/**
 * Initializes theme listeners and applies initial theme.
 */
export function initTheme() {
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", toggleTheme);
    }

    if (window.matchMedia) {
        window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
            try {
                if (!localStorage.getItem(THEME_STORAGE_KEY)) {
                    applyTheme(e.matches ? "light" : "dark");
                }
            } catch (err) {}
        });
    }

    applyTheme(getPreferredTheme());
}
