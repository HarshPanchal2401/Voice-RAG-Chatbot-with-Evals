/* Voice RAG — drag-to-resize side panels (v1.0).
 * Drag the handle on the inner edge of each sidebar; double-click resets.
 * Arrow keys resize when a handle is focused. Widths persist in localStorage. */
(function () {
  'use strict';
  var KEY = 'voice_rag_panel_widths';
  var LIMITS = { left: { min: 220, max: 440, def: 280 }, right: { min: 280, max: 560, def: 340 } };
  var MIN_MAIN = 480;
  var layout = document.querySelector('.app-layout');
  if (!layout) return;

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { return {}; }
  }
  function save(w) { try { localStorage.setItem(KEY, JSON.stringify(w)); } catch (e) { /* ignore */ } }

  var widths = load();
  function clamp(side, px) {
    var lim = LIMITS[side];
    var other = side === 'left' ? (widths.right || LIMITS.right.def) : (widths.left || LIMITS.left.def);
    var maxByViewport = window.innerWidth - other - MIN_MAIN;
    return Math.round(Math.max(lim.min, Math.min(lim.max, maxByViewport, px)));
  }
  function apply(side, px) {
    widths[side] = clamp(side, px);
    layout.style.setProperty(side === 'left' ? '--left-w' : '--right-w', widths[side] + 'px');
    var h = handles[side];
    if (h) h.setAttribute('aria-valuenow', String(widths[side]));
  }

  var handles = {};
  function makeHandle(side, panel) {
    if (!panel) return;
    var h = document.createElement('div');
    h.className = 'panel-resizer panel-resizer-' + side;
    h.setAttribute('role', 'separator');
    h.setAttribute('aria-orientation', 'vertical');
    h.setAttribute('aria-label', side === 'left' ? 'Resize sidebar' : 'Resize telemetry panel');
    h.setAttribute('aria-valuemin', String(LIMITS[side].min));
    h.setAttribute('aria-valuemax', String(LIMITS[side].max));
    h.tabIndex = 0;
    h.title = 'Drag to resize · double-click to reset';
    panel.appendChild(h);
    handles[side] = h;

    h.addEventListener('pointerdown', function (e) {
      if (e.button !== 0) return;
      e.preventDefault();
      h.setPointerCapture(e.pointerId);
      layout.classList.add('is-resizing');
      var startX = e.clientX;
      var startW = panel.getBoundingClientRect().width;
      function move(ev) {
        var dx = ev.clientX - startX;
        apply(side, side === 'left' ? startW + dx : startW - dx);
      }
      function up(ev) {
        h.releasePointerCapture(ev.pointerId);
        h.removeEventListener('pointermove', move);
        h.removeEventListener('pointerup', up);
        h.removeEventListener('pointercancel', up);
        layout.classList.remove('is-resizing');
        save(widths);
        window.dispatchEvent(new Event('resize')); // let the waveform canvas re-measure
      }
      h.addEventListener('pointermove', move);
      h.addEventListener('pointerup', up);
      h.addEventListener('pointercancel', up);
    });
    h.addEventListener('dblclick', function () {
      apply(side, LIMITS[side].def);
      save(widths);
      window.dispatchEvent(new Event('resize'));
    });
    h.addEventListener('keydown', function (e) {
      var step = e.shiftKey ? 48 : 16;
      var cur = widths[side] || LIMITS[side].def;
      var grow = side === 'left' ? 'ArrowRight' : 'ArrowLeft';
      var shrink = side === 'left' ? 'ArrowLeft' : 'ArrowRight';
      if (e.key === grow) apply(side, cur + step);
      else if (e.key === shrink) apply(side, cur - step);
      else if (e.key === 'Home') apply(side, LIMITS[side].def);
      else return;
      e.preventDefault();
      save(widths);
    });
  }

  makeHandle('left', document.getElementById('sidebar-left'));
  makeHandle('right', document.getElementById('telemetry-panel'));
  if (widths.left) apply('left', widths.left);
  if (widths.right) apply('right', widths.right);
  window.addEventListener('resize', function () {
    if (widths.left) apply('left', widths.left);
    if (widths.right) apply('right', widths.right);
  });
})();
