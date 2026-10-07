/* Voice RAG — apply the saved theme before first paint (v5.0.0).
 * Without a saved choice the CSS follows prefers-color-scheme. */
(function () {
  'use strict';
  try {
    var saved = window.localStorage.getItem('voice_rag_theme');
    if (saved === 'light' || saved === 'dark') {
      document.documentElement.setAttribute('data-theme', saved);
    }
  } catch (e) {
    /* storage unavailable: follow the system preference */
  }
})();
