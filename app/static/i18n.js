/**
 * UI i18n: both locales embedded in window.__UI_CATALOG (no full page reload on toggle).
 */
(function () {
  const LOCALE_STORAGE_KEY = "docread-locale";
  const LOCALE_COOKIE = "app_locale";

  const catalog = window.__UI_CATALOG || {};
  const serverLocale = (window.__UI_LOCALE || "en").toLowerCase().startsWith("de")
    ? "de"
    : "en";
  const storedLocale = localStorage.getItem(LOCALE_STORAGE_KEY);
  let currentLocale =
    storedLocale === "en" || storedLocale === "de" ? storedLocale : serverLocale;
  let messages =
    catalog[currentLocale] || window.__UI_MESSAGES || catalog.en || catalog.de || {};

  window.__docreadLocale = currentLocale;
  document.documentElement.lang = currentLocale;

  function t(key, params) {
    let text = messages[key] ?? key;
    if (params) {
      for (const [name, value] of Object.entries(params)) {
        text = text.replaceAll(`{${name}}`, String(value));
      }
    }
    return text;
  }

  function applyDomTranslations() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (key) el.textContent = t(key);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      const key = el.getAttribute("data-i18n-placeholder");
      if (key) el.placeholder = t(key);
    });
    document.querySelectorAll("[data-i18n-title]").forEach((el) => {
      const key = el.getAttribute("data-i18n-title");
      if (key) el.title = t(key);
    });
    document.querySelectorAll("[data-i18n-aria-label]").forEach((el) => {
      const key = el.getAttribute("data-i18n-aria-label");
      if (key) el.setAttribute("aria-label", t(key));
    });
    const titleKey = document.querySelector("title")?.getAttribute("data-i18n");
    if (titleKey) document.title = t(titleKey);
  }

  function updateToggleButtons() {
    const enBtn = document.getElementById("locale-en-btn");
    const deBtn = document.getElementById("locale-de-btn");
    if (!enBtn || !deBtn) return;
    enBtn.setAttribute("aria-pressed", String(currentLocale === "en"));
    deBtn.setAttribute("aria-pressed", String(currentLocale === "de"));
  }

  function setLocale(next) {
    const code = next === "de" ? "de" : "en";
    if (code === currentLocale) return;
    const nextMessages = catalog[code];
    if (!nextMessages) return;

    currentLocale = code;
    messages = nextMessages;
    window.__docreadLocale = code;
    document.documentElement.lang = code;

    localStorage.setItem(LOCALE_STORAGE_KEY, code);
    const base = (document.body?.dataset.basePath || "").replace(/\/$/, "") || "";
    const path = base || "/";
    document.cookie = `${LOCALE_COOKIE}=${code};path=${path};max-age=31536000;SameSite=Lax`;

    applyDomTranslations();
    updateToggleButtons();
    document.dispatchEvent(
      new CustomEvent("docread:locale-change", { detail: { locale: code } }),
    );
  }

  function initLocaleToggle() {
    const enBtn = document.getElementById("locale-en-btn");
    const deBtn = document.getElementById("locale-de-btn");
    if (!enBtn || !deBtn) return;
    updateToggleButtons();
    enBtn.addEventListener("click", () => setLocale("en"));
    deBtn.addEventListener("click", () => setLocale("de"));
  }

  window.docreadT = t;
  window.docreadSetLocale = setLocale;
  window.docreadApplyI18n = applyDomTranslations;

  applyDomTranslations();
  initLocaleToggle();
})();
