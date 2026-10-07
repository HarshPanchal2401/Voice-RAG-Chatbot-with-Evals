/* ==========================================================================
   Voice RAG Chatbot — "Studio" UI controller
   Talks to the v3 API: /health, /api/v1/sample-queries, /api/v1/query/text/stream (SSE),
   /api/v1/query/voice, /api/v1/voice/tts, /api/v1/feedback.
   ========================================================================== */
(function () {
  'use strict';

  // ------------------------------------------------------------------ icons
  const ICONS = {
    logo: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M8 10v1M12 8v5M16 10v1"/>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    chat: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    bulb: '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/>',
    eval: '<rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="m9 14 2 2 4-4"/>',
    settings: '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
    mic: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><path d="M12 19v3"/>',
    send: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
    moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    user: '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    bot: '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2M20 14h2M15 13v2M9 13v2"/>',
    play: '<polygon points="7 4 20 12 7 20 7 4"/>',
    pause: '<rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/>',
    vol: '<path d="M11 4.7a.7.7 0 0 0-1.2-.5L6.4 7.6A1.4 1.4 0 0 1 5.4 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.4a1.4 1.4 0 0 1 1 .4l3.4 3.4a.7.7 0 0 0 1.2-.5z"/><path d="M16 9a5 5 0 0 1 0 6"/><path d="M19.4 18.4a9 9 0 0 0 0-12.7"/>',
    mute: '<path d="M11 4.7a.7.7 0 0 0-1.2-.5L6.4 7.6A1.4 1.4 0 0 1 5.4 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.4a1.4 1.4 0 0 1 1 .4l3.4 3.4a.7.7 0 0 0 1.2-.5z"/><path d="m22 9-6 6M16 9l6 6"/>',
    copy: '<rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    chart: '<path d="M3 3v18h18"/><path d="M18 17V9M13 17V5M8 17v-3"/>',
    clock: '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    'check-circle': '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    chevron: '<path d="m9 18 6-6-6-6"/>',
    up: '<path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/>',
    down: '<path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"/>',
    book: '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
    plus: '<path d="M5 12h14M12 5v14"/>',
    clip: '<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>',
    x: '<path d="M18 6 6 18M6 6l12 12"/>',
    refresh: '<path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M8 16H3v5"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  };
  function svg(name) {
    return '<svg class="ic" viewBox="0 0 24 24" aria-hidden="true">' + (ICONS[name] || '') + '</svg>';
  }
  function hydrateIcons(root) {
    (root || document).querySelectorAll('i[data-icon]').forEach((el) => {
      const wrap = document.createElement('span');
      wrap.innerHTML = svg(el.dataset.icon);
      const s = wrap.firstChild;
      if (el.id) s.id = el.id;
      el.replaceWith(s);
    });
  }

  // ------------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  function el(tag, cls, html) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
    raw(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    setRaw(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (e) { /* ignore */ } },
  };
  const nowTime = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  function fmtMs(ms) {
    if (ms == null || isNaN(ms)) return '–';
    return ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(1) + ' s';
  }
  function fmtClock(sec) {
    if (!isFinite(sec) || sec < 0) sec = 0;
    const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
  }
  function b64ToBytes(b64) {
    const bin = atob(b64); const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function formatDetail(detail, fallback) {
    if (!detail) return fallback || 'Request failed.';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((d) => (d && d.msg ? (d.loc ? d.loc.slice(-1)[0] + ': ' : '') + d.msg : String(d))).join('; ');
    return detail.message || JSON.stringify(detail);
  }
  function toast(msg, kind) {
    const t = el('div', 'toast' + (kind === 'error' ? ' error' : ''));
    t.textContent = msg;
    $('toasts').appendChild(t);
    setTimeout(() => t.classList.add('hide'), kind === 'error' ? 5000 : 2800);
    setTimeout(() => t.remove(), kind === 'error' ? 5400 : 3200);
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); }
    catch (e) {
      const ta = el('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); } catch (e2) { /* ignore */ } ta.remove();
    }
    toast('Copied to clipboard');
  }

  // ------------------------------------------------------------------ i18n
  const L = {
    hi: {
      name: 'Hindi', short: 'HI', locale: 'hi-IN',
      title: '👋 नमस्ते, मैं आपका AI सहायक हूँ',
      sub: 'कृपया अपना सवाल पूछें या नीचे दिए गए नमूना प्रश्नों में से किसी एक को चुनें।',
      placeholder: 'अपना संदेश लिखें या आवाज़ का उपयोग करें… (Type your message or use voice)',
      short: 'अपना सवाल लिखें…',
    },
    gu: {
      name: 'Gujarati', short: 'GU', locale: 'gu-IN',
      title: '👋 નમસ્તે, હું તમારો AI સહાયક છું',
      sub: 'કૃપા કરીને તમારો પ્રશ્ન પૂછો અથવા નીચે આપેલા નમૂના પ્રશ્નોમાંથી એક પસંદ કરો.',
      placeholder: 'તમારો સંદેશ લખો અથવા અવાજનો ઉપયોગ કરો… (Type your message or use voice)',
      short: 'તમારો પ્રશ્ન લખો…',
    },
  };
  const mqNarrow = window.matchMedia('(max-width: 860px)');
  const placeholderText = () => (mqNarrow.matches ? L[S.lang].short : L[S.lang].placeholder);
  const LANG_LABEL = { hi: 'Hindi', gu: 'Gujarati', en: 'English' };

  // ------------------------------------------------------------------ state
  const KEYS = { lang: 'voice_rag_lang', key: 'voice_rag_api_key', theme: 'voice_rag_theme', settings: 'vr_settings', recent: 'vr_recent', stats: 'vr_stats', evals: 'vr_evals' };
  const S = {
    lang: (store.raw(KEYS.lang) === 'gu') ? 'gu' : 'hi',
    apiKey: store.raw(KEYS.key) || '',
    authRequired: false,
    settings: Object.assign({ voiceReply: false, evaluate: false, autoDetect: true, reranker: false, topK: 5 }, store.get(KEYS.settings, {})),
    history: [],
    busy: false,
    abort: null,
    samples: [],
    sampleToken: 0,
    pendingQueryId: null,
    recorder: null,
  };
  const saveSettings = () => store.set(KEYS.settings, S.settings);

  // Anonymous browser id: lets the server count voice uses per user (together with the IP).
  const CLIENT_ID = (() => {
    let id = store.raw('vr_client_id');
    if (!id || !/^[A-Za-z0-9_-]{8,64}$/.test(id)) {
      id = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : 'c' + Math.random().toString(36).slice(2) + Date.now().toString(36);
      store.setRaw('vr_client_id', id);
    }
    return id;
  })();

  // ------------------------------------------------------------------ API
  async function api(path, opts) {
    opts = opts || {};
    const headers = Object.assign({}, opts.headers || {});
    if (S.apiKey) headers['X-API-Key'] = S.apiKey;
    headers['X-Client-Id'] = CLIENT_ID;
    const res = await fetch(path, Object.assign({}, opts, { headers }));
    if (res.status === 401) {
      S.apiKey = ''; store.setRaw(KEYS.key, null);
      askApiKey('The server rejected the request. Please enter a valid API key.');
      throw new Error('API key required.');
    }
    if (!res.ok) {
      let detail = null;
      try { detail = (await res.json()).detail; } catch (e) { /* ignore */ }
      let msg = formatDetail(detail, 'Request failed (' + res.status + ').');
      if (res.status === 429 && !detail) msg = 'Too many requests. Please wait ' + (res.headers.get('Retry-After') || 'a few') + ' seconds.';
      if (res.status === 429 && /voice/i.test(msg)) refreshQuota();
      if (res.status === 413) msg = 'The audio file is too large.';
      throw new Error(msg);
    }
    return res;
  }

  async function checkHealth() {
    const chip = $('status-chip');
    try {
      const r = await fetch('/health');
      const d = await r.json();
      chip.dataset.state = 'online';
      $('status-text').textContent = 'Online';
      chip.title = (d.llm_model || '') + (d.reranker_backend ? ' · rerank: ' + d.reranker_backend : '');
      S.authRequired = !!d.auth_required;
      if (S.authRequired && !S.apiKey) askApiKey('This server requires an API key. It is stored in this browser only.');
    } catch (e) {
      chip.dataset.state = 'offline';
      $('status-text').textContent = 'Offline';
    }
  }

  // ------------------------------------------------------------------ language + theme
  function setLang(code, opts) {
    S.lang = code === 'gu' ? 'gu' : 'hi';
    store.setRaw(KEYS.lang, S.lang);
    const t = L[S.lang];
    document.documentElement.lang = S.lang;
    document.body.classList.toggle('lang-gu', S.lang === 'gu');
    document.body.classList.toggle('lang-hi', S.lang === 'hi');
    $('welcome-title').textContent = t.title;
    $('welcome-sub').textContent = t.sub;
    $('composer-input').placeholder = placeholderText();
    document.querySelectorAll('.seg-btn').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.lang === S.lang)));
    if (!opts || opts.load !== false) loadSamples();
  }

  const mqDark = window.matchMedia('(prefers-color-scheme: dark)');
  const effectiveTheme = () => {
    const t = document.documentElement.getAttribute('data-theme');
    return t === 'dark' || t === 'light' ? t : (mqDark.matches ? 'dark' : 'light');
  };
  function syncThemeIcon() {
    const dark = effectiveTheme() === 'dark';
    const btn = $('theme-btn');
    btn.innerHTML = svg(dark ? 'sun' : 'moon');
    btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  }
  function toggleTheme() {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    store.setRaw(KEYS.theme, next);
    syncThemeIcon();
  }

  // ------------------------------------------------------------------ samples
  async function loadSamples() {
    const token = ++S.sampleToken;
    const box = $('sample-chips');
    box.innerHTML = '';
    for (let i = 0; i < 6; i++) box.appendChild(el('span', 'chip skeleton', '&nbsp;'));
    try {
      const r = await api('/api/v1/sample-queries?lang=' + S.lang + '&count=40');
      const d = await r.json();
      if (token !== S.sampleToken) return;
      S.samples = d.queries || [];
      renderChips();
    } catch (e) {
      if (token !== S.sampleToken) return;
      box.innerHTML = '';
      box.appendChild(el('span', 'empty', esc(e.message || 'Could not load sample questions.')));
    }
  }
  function renderChips() {
    const box = $('sample-chips');
    box.innerHTML = '';
    const pool = S.samples.slice().sort(() => 0.5 - Math.random()).slice(0, 6);
    if (!pool.length) { box.appendChild(el('span', 'empty', 'No sample questions available.')); return; }
    pool.forEach((q) => {
      const b = el('button', 'chip');
      b.type = 'button';
      b.textContent = q.question;
      b.title = q.question + (q.query_type ? ' · ' + q.query_type : '');
      b.addEventListener('click', () => ask(q.question, q.query_id));
      box.appendChild(b);
    });
  }

  // ------------------------------------------------------------------ formatting
  const CITE_RE = /\[(?:સંદર્ભ|संदर्भ|Source|Context|Doc)\s*(\d+)\]/gi;
  function formatAnswer(text) {
    let h = esc(text || '');
    h = h.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    h = h.replace(CITE_RE, (m, n) => '<button type="button" class="cite" data-cite="' + n + '" aria-label="Source ' + n + '">' + n + '</button>');
    const blocks = h.split(/\n{2,}/).map((b) => {
      const lines = b.split('\n');
      if (lines.every((l) => /^\s*[-*•]\s+/.test(l))) return '<ul>' + lines.map((l) => '<li>' + l.replace(/^\s*[-*•]\s+/, '') + '</li>').join('') + '</ul>';
      return '<p>' + lines.join('<br>') + '</p>';
    });
    return blocks.join('');
  }

  // ------------------------------------------------------------------ audio
  const player = { audio: new Audio(), queue: [], playingMsg: null, msgAudios: new Set() };
  function stopAllAudio() {
    player.queue = [];
    try { player.audio.pause(); } catch (e) { /* ignore */ }
    player.msgAudios.forEach((a) => { try { a.pause(); } catch (e) { /* ignore */ } });
  }
  // Sentence-level TTS chunks arriving during streaming: play back-to-back.
  function enqueueChunk(bytes) {
    player.queue.push(URL.createObjectURL(new Blob([bytes], { type: 'audio/mpeg' })));
    if (player.audio.paused && player.audio.dataset.busy !== '1') playNextChunk();
  }
  function playNextChunk() {
    const url = player.queue.shift();
    if (!url) { player.audio.dataset.busy = '0'; return; }
    player.audio.dataset.busy = '1';
    player.audio.src = url;
    player.audio.onended = () => { URL.revokeObjectURL(url); playNextChunk(); };
    player.audio.onerror = () => { URL.revokeObjectURL(url); playNextChunk(); };
    player.audio.play().catch(() => { player.audio.dataset.busy = '0'; });
  }

  // Each answer gets a speaker icon button (no audio bar). Click = play/pause;
  // if no audio exists yet, it is synthesized on demand first.
  function updateSpeakBtn(msg) {
    const b = msg.speakBtn;
    if (!b) return;
    const playing = !!(msg.audio && !msg.audio.paused);
    b.classList.toggle('loading', !!msg.ttsLoading);
    b.classList.toggle('on', playing);
    b.disabled = !!msg.ttsLoading;
    b.innerHTML = msg.ttsLoading ? '<span class="spin" aria-hidden="true"></span>' : svg(playing ? 'pause' : 'vol');
    const label = msg.ttsLoading ? 'Generating audio…' : playing ? 'Pause audio' : 'Listen to answer';
    b.setAttribute('aria-label', label);
    b.title = label;
  }

  function buildPlayer(msg, blob, autoplay) {
    if (msg.audio) { try { msg.audio.pause(); } catch (e) { /* ignore */ } player.msgAudios.delete(msg.audio); }
    if (msg.audioUrl) URL.revokeObjectURL(msg.audioUrl);
    msg.audioUrl = URL.createObjectURL(blob);
    msg.els.player.hidden = true;
    const audio = new Audio(msg.audioUrl);
    audio.preload = 'auto';
    ['play', 'pause', 'ended'].forEach((ev) => audio.addEventListener(ev, () => updateSpeakBtn(msg)));
    audio.addEventListener('ended', () => { audio.currentTime = 0; });
    msg.audio = audio;
    player.msgAudios.add(audio);
    updateSpeakBtn(msg);
    if (autoplay) { stopAllAudio(); audio.play().catch(() => { /* autoplay blocked: user can click the speaker */ }); }
  }

  function toggleSpeak(msg) {
    if (msg.ttsLoading) return;
    if (!msg.audio) { synthesizeFor(msg); return; }
    if (msg.audio.paused) { stopAllAudio(); msg.audio.play().catch(() => toast('Could not play audio', 'error')); }
    else msg.audio.pause();
  }

  async function synthesizeFor(msg) {
    if (!msg.answer) return;
    msg.ttsLoading = true; updateSpeakBtn(msg);
    try {
      const r = await api('/api/v1/voice/tts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: msg.answer, language: msg.answerLang || S.lang }) });
      const d = await r.json();
      msg.ttsLoading = false;
      buildPlayer(msg, new Blob([b64ToBytes(d.audio_base64)], { type: 'audio/mpeg' }), true);
      refreshQuota();
    } catch (e) {
      msg.ttsLoading = false; updateSpeakBtn(msg);
      toast(e.message || 'Could not generate audio', 'error');
    }
  }

  // ------------------------------------------------------------------ messages
  function scrollToEnd(force) {
    const sc = $('chat-scroll');
    const near = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 160;
    if (force || near) sc.scrollTop = sc.scrollHeight;
  }

  function addUserMessage(text, extra) {
    const row = el('div', 'msg user');
    const col = el('div', 'msg-col');
    const bubble = el('div', 'bubble');
    bubble.textContent = text;
    const meta = el('div', 'msg-meta');
    meta.innerHTML = (extra && extra.voice ? '<span class="lang-tag">🎤 Voice</span>' : '') + '<span>' + nowTime() + '</span>';
    col.append(bubble, meta);
    row.append(el('div', 'avatar user', svg('user')), col);
    $('thread').appendChild(row);
    document.querySelector('.chat-card').classList.add('has-msgs');
    scrollToEnd(true);
    return { row, bubble };
  }

  function addBotMessage() {
    const row = el('div', 'msg bot');
    const col = el('div', 'msg-col');
    const bubble = el('div', 'bubble');
    const text = el('div', 'msg-text', '<span class="typing" aria-label="Thinking"><span></span><span></span><span></span></span>');
    const player = el('div', 'player'); player.hidden = true;
    bubble.append(text, player);
    const meta = el('div', 'msg-meta'); meta.innerHTML = '<span>' + nowTime() + '</span>';
    const notes = el('div', 'notes');
    const tools = el('div', 'msg-tools'); tools.hidden = true;
    const refs = el('div', 'refs'); refs.hidden = true;
    const insights = el('div', 'insights'); insights.hidden = true;
    const foot = el('div', 'msg-foot');
    foot.append(tools, meta);
    col.append(bubble, foot, notes, refs, insights);
    row.append(el('div', 'avatar bot', svg('bot')), col);
    $('thread').appendChild(row);
    scrollToEnd(true);
    const msg = { row, answer: '', sources: [], els: { text, player, meta, notes, tools, refs, insights } };
    text.addEventListener('click', (e) => {
      const c = e.target.closest('.cite');
      if (c) openSource(msg, parseInt(c.dataset.cite, 10));
    });
    return msg;
  }

  let rafPending = false;
  function renderStreaming(msg) {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      if (msg.final) return; // the final answer was already rendered
      msg.els.text.innerHTML = formatAnswer(msg.answer);
      const last = msg.els.text.lastElementChild;
      if (last) last.classList.add('caret');
      scrollToEnd();
    });
  }

  function addNote(msg, kind, html) {
    msg.els.notes.appendChild(el('div', 'note ' + kind, html));
  }

  function finalizeMessage(msg, data) {
    msg.final = true;
    msg.answer = (data.answer != null ? data.answer : msg.answer) || '';
    msg.answerLang = data.answer_language || data.language || S.lang;
    msg.traceId = data.trace_id || null;
    msg.sources = data.sources || [];
    msg.els.text.innerHTML = msg.answer ? formatAnswer(msg.answer) : '<p class="searched">No answer was returned.</p>';
    const lang = LANG_LABEL[msg.answerLang] || msg.answerLang;
    msg.els.meta.innerHTML = '';
    if (data.retrieval_query && data.query && data.retrieval_query.trim() !== data.query.trim()) {
      const sl = el('div', 'searched-line', svg('search') + '<span>Searched for: ' + esc(data.retrieval_query) + '</span>');
      sl.title = 'Your follow-up was rewritten into this standalone question before searching: ' + data.retrieval_query;
      msg.els.notes.prepend(sl);
    }
    if (data.no_answer) addNote(msg, 'warn', svg('info') + '<span>No answer was found in the indexed sources.</span>');
    renderTools(msg);
    msg.els.meta.appendChild(el('span', 'msg-time', nowTime()));
    msg.els.meta.title = 'Answer language: ' + lang;
    renderRefs(msg);
    if (data.latency) renderLatency(msg, data.latency);
  }

  function renderTools(msg) {
    const t = msg.els.tools;
    t.innerHTML = '';
    t.hidden = false;
    if (msg.sources.length) {
      const b = el('button', 'meta-link', svg('book') + '<span>' + msg.sources.length + ' sources</span><span class="chev">' + svg('chevron') + '</span>');
      b.type = 'button'; b.setAttribute('aria-expanded', 'false');
      b.addEventListener('click', () => {
        const open = msg.els.refs.hidden;
        msg.els.refs.hidden = !open; b.classList.toggle('on', open); b.setAttribute('aria-expanded', String(open));
      });
      msg.refsBtn = b;
      msg.els.meta.appendChild(b);
    }
    // Insights toggle: evaluation + latency stay hidden until opened
    const ib = el('button', 'meta-link insights-btn', svg('chart') + '<span>Insights</span><span class="q-dot" hidden></span><span class="chev">' + svg('chevron') + '</span>');
    ib.type = 'button'; ib.setAttribute('aria-expanded', 'false'); ib.hidden = !msg.ins;
    ib.addEventListener('click', () => setInsightsOpen(msg, msg.els.insights.hidden));
    msg.insightsBtn = ib;
    msg.els.meta.appendChild(ib);
    if (msg.evalState !== undefined) updateEvalChip(msg, msg.evalState);
    // Total latency chip, coloured by speed; opens the Latency tab
    const lc = el('button', 'tool-btn lat-chip', svg('clock') + '<span class="lat-v">–</span>');
    lc.type = 'button'; lc.hidden = true;
    lc.addEventListener('click', () => {
      const showingLat = !msg.els.insights.hidden && msg.ins && msg.ins.active === 'lat';
      setInsightsOpen(msg, !showingLat, 'lat');
    });
    msg.latChip = lc;
    t.appendChild(lc);
    if (msg.answer) {
      const sp = el('button', 'tool-btn icon-only speak-btn'); sp.type = 'button';
      sp.addEventListener('click', () => toggleSpeak(msg));
      msg.speakBtn = sp;
      t.appendChild(sp);
      updateSpeakBtn(msg);
    }
    const copy = el('button', 'tool-btn icon-only', svg('copy')); copy.type = 'button'; copy.setAttribute('aria-label', 'Copy answer');
    copy.addEventListener('click', () => copyText(msg.answer));
    t.appendChild(copy);
    if (msg.traceId) {
      [['up', 1, 'Good answer'], ['down', 0, 'Bad answer']].forEach(([ic, score, label]) => {
        const b = el('button', 'tool-btn icon-only', svg(ic)); b.type = 'button'; b.setAttribute('aria-label', label);
        b.addEventListener('click', async () => {
          t.querySelectorAll('[data-fb]').forEach((x) => { x.disabled = true; });
          b.classList.add('on');
          try {
            await api('/api/v1/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ trace_id: msg.traceId, score }) });
            toast('Thanks for the feedback');
          } catch (e) { toast(e.message, 'error'); }
        });
        b.dataset.fb = '1';
        t.appendChild(b);
      });
    }
  }

  function renderRefs(msg) {
    const box = msg.els.refs;
    box.innerHTML = '';
    msg.sources.forEach((s, i) => {
      const r = el('div', 'ref');
      r.dataset.n = String(i + 1);
      const head = el('div', 'ref-head');
      const scores = [];
      if (s.rerank_score != null) scores.push('rerank ' + Number(s.rerank_score).toFixed(2));
      if (s.dense_score != null) scores.push('dense ' + Number(s.dense_score).toFixed(2));
      head.innerHTML = '<span class="ref-num">[' + (i + 1) + ']</span>' + (s.title ? '<span>' + esc(s.title) + '</span>' : '') + '<span class="mono">' + esc(scores.join(' · ')) + '</span>';
      const tx = el('div', 'ref-text'); tx.textContent = s.text || '';
      r.append(head, tx);
      if ((s.text || '').length > 220) {
        const more = el('button', 'ref-more', 'Show more'); more.type = 'button';
        more.addEventListener('click', () => { const o = r.classList.toggle('open'); more.textContent = o ? 'Show less' : 'Show more'; });
        r.appendChild(more);
      }
      if (s.url && /^https?:\/\//i.test(s.url)) {
        const a = el('a', 'ref-more'); a.href = s.url; a.target = '_blank'; a.rel = 'noopener noreferrer'; a.textContent = 'Open source';
        r.appendChild(a);
      }
      box.appendChild(r);
    });
  }

  function openSource(msg, n) {
    if (!msg.sources.length) return;
    msg.els.refs.hidden = false;
    if (msg.refsBtn) { msg.refsBtn.classList.add('on'); msg.refsBtn.setAttribute('aria-expanded', 'true'); }
    const r = msg.els.refs.querySelector('.ref[data-n="' + n + '"]');
    if (r) { r.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); r.classList.add('flash'); setTimeout(() => r.classList.remove('flash'), 1400); }
  }

  // Latency colour bands (ms): green < 2.5 s, amber < 6 s, red otherwise.
  function latClass(ms) { return ms == null ? '' : ms < 2500 ? 'good' : ms < 6000 ? 'mid' : 'bad'; }

  // Collapsible, tabbed insights panel (Evaluation | Latency). Hidden until the user opens it.
  function ensureInsights(msg) {
    if (msg.ins) return msg.ins;
    const box = msg.els.insights;
    box.innerHTML = '';
    box.classList.add('insights-panel');
    const tabs = el('div', 'tabs'); tabs.setAttribute('role', 'tablist');
    const panes = {};
    const tabBtns = {};
    [['eval', 'chart', 'Evaluation'], ['lat', 'clock', 'Latency']].forEach(([key, ic, label]) => {
      const b = el('button', 'tab', svg(ic) + '<span>' + label + '</span>');
      b.type = 'button'; b.setAttribute('role', 'tab'); b.dataset.tab = key; b.hidden = true;
      b.addEventListener('click', () => { msg.ins.userPicked = true; showTab(msg, key); });
      tabs.appendChild(b); tabBtns[key] = b;
      const pane = el('div', 'tab-pane'); pane.setAttribute('role', 'tabpanel'); pane.hidden = true;
      panes[key] = pane;
    });
    box.append(tabs, panes.eval, panes.lat);
    msg.ins = { tabs: tabBtns, panes, active: null, userPicked: false };
    return msg.ins;
  }
  function showTab(msg, key) {
    const ins = ensureInsights(msg);
    ins.active = key;
    Object.keys(ins.panes).forEach((k) => {
      ins.panes[k].hidden = k !== key;
      ins.tabs[k].setAttribute('aria-selected', String(k === key));
      ins.tabs[k].classList.toggle('active', k === key);
    });
  }
  function insightPane(msg, key) {
    const ins = ensureInsights(msg);
    ins.tabs[key].hidden = false;
    if (!ins.active || (key === 'eval' && !ins.userPicked)) showTab(msg, key);
    if (msg.insightsBtn) msg.insightsBtn.hidden = false;
    return ins.panes[key];
  }
  function setInsightsOpen(msg, open, tab) {
    if (!msg.ins) return;
    msg.els.insights.hidden = !open;
    if (tab) { msg.ins.userPicked = true; showTab(msg, tab); }
    if (msg.insightsBtn) {
      msg.insightsBtn.classList.toggle('on', open);
      msg.insightsBtn.setAttribute('aria-expanded', String(open));
    }
  }

  function bar(v) {
    const pct = v == null ? 0 : Math.round(Math.max(0, Math.min(1, v)) * 100);
    const cls = v == null ? '' : v >= 0.8 ? 'good' : v >= 0.5 ? 'mid' : 'bad';
    return '<span class="bar"><span class="bar-fill ' + cls + '" style="width:' + pct + '%"></span></span>';
  }

  function renderEvalPending(msg) {
    insightPane(msg, 'eval').innerHTML = '<div class="pane-head"><span class="pane-title">' + svg('chart') + 'Evaluation Scores</span><span class="badge">Evaluating…</span></div><div class="insight-empty">Running DeepEval metrics on this answer.</div>';
    updateEvalChip(msg, 'pending');
  }

  function renderEval(msg, ev, evalMs) {
    const pane = insightPane(msg, 'eval');
    if (!ev) {
      pane.innerHTML = '<div class="pane-head"><span class="pane-title">' + svg('chart') + 'Evaluation Scores</span><span class="badge bad">Failed</span></div><div class="insight-empty">The evaluation could not be completed.</div>';
      updateEvalChip(msg, null);
      return;
    }
    const s = ev.scores || {};
    const rows = [['Context Recall', s.context_recall], ['Faithfulness', s.faithfulness], ['Answer Relevance', s.answer_relevance], ['Answer Correctness', s.answer_correctness], ['Hit Rate @K', s.hit_rate_at_k]]
      .filter(([, v]) => v !== undefined);
    const overall = s.overall_score;
    let badge = '<span class="badge">n/a</span>';
    if (overall != null) {
      const cls = overall >= 0.8 ? 'good' : overall >= 0.5 ? 'mid' : 'bad';
      const label = overall >= 0.8 ? 'Good' : overall >= 0.5 ? 'Fair' : 'Poor';
      badge = '<span class="badge plain">Overall Quality</span><span class="badge ' + cls + '">' + label + ' · ' + overall.toFixed(2) + '</span>';
    }
    let html = '<div class="pane-head"><span class="pane-title">' + svg('chart') + 'Evaluation Scores</span><span class="badges">' + badge + '</span></div>';
    html += rows.map(([k, v]) => '<div class="metric"><span class="metric-name">' + k + '</span>' + bar(v) +
      (v == null ? '<span class="metric-val na">n/a</span>' : '<span class="metric-val">' + Number(v).toFixed(2) + '</span>') + '</div>').join('');
    if (ev.failed_metrics && ev.failed_metrics.length) html += '<div class="note err" style="margin-top:8px">Failed: ' + esc(ev.failed_metrics.join(', ')) + '</div>';
    if (ev.warning) html += '<div class="insight-empty" style="margin-top:8px">' + esc(ev.warning) + '</div>';
    if (evalMs) html += '<div class="insight-empty" style="margin-top:8px">Evaluated in ' + fmtMs(evalMs) + (ev.is_golden ? ' · golden question' : '') + '</div>';
    pane.innerHTML = html;
    updateEvalChip(msg, overall);
    const log = store.get(KEYS.evals, []);
    log.push({ t: Date.now(), lang: msg.answerLang, overall: overall, s });
    store.set(KEYS.evals, log.slice(-200));
  }

  function updateEvalChip(msg, overall) {
    msg.evalState = overall;
    const b = msg.insightsBtn;
    if (!b) return;
    const dot = b.querySelector('.q-dot');
    if (!dot) return;
    dot.className = 'q-dot ' + (overall === 'pending' ? 'pending' : overall == null ? 'bad' : overall >= 0.8 ? 'good' : overall >= 0.5 ? 'mid' : 'bad');
    dot.hidden = false;
    dot.title = overall === 'pending' ? 'Evaluating…' : overall == null ? 'Evaluation failed' : 'Overall quality ' + overall.toFixed(2);
  }

  function renderLatency(msg, lat) {
    msg.latency = lat;
    const r = lat.retrieval || {};
    const rows = [
      ['Speech-to-text', lat.stt_ms],
      ['Query rewrite', r.condense_ms],
      ['Retrieval', r.retrieval_total_ms != null ? r.retrieval_total_ms - (r.rerank_ms || 0) : null],
      ['Reranking', r.rerank_ms],
      ['TTFT', lat.ttft_ms],
      ['Generation', lat.llm_ms],
      ['Speech synthesis', lat.tts_ms != null ? lat.tts_ms : msg.ttsMs],
    ].filter(([, v]) => v != null && !isNaN(v));
    const total = lat.total_ms;
    const max = Math.max.apply(null, rows.map(([, v]) => v).concat([1]));
    insightPane(msg, 'lat').innerHTML = '<div class="pane-head"><span class="pane-title">' + svg('clock') + 'Latency Breakdown</span><span class="badge ' + latClass(total) + '">' + fmtMs(total) + '</span></div>' +
      rows.map(([k, v]) => '<div class="metric"><span class="metric-name">' + k + '</span><span class="bar"><span class="bar-fill lat" style="width:' + Math.max(2, Math.round((v / max) * 100)) + '%"></span></span><span class="metric-val">' + fmtMs(v) + '</span></div>').join('') +
      '<div class="lat-row total"><span>Total</span><span class="v ' + latClass(total) + '">' + fmtMs(total) + '</span></div>';
    if (msg.latChip) {
      msg.latChip.hidden = total == null;
      msg.latChip.className = 'tool-btn lat-chip ' + latClass(total);
      msg.latChip.querySelector('.lat-v').textContent = fmtMs(total);
      msg.latChip.title = 'Total response time ' + fmtMs(total) + ' (green under 2.5 s, amber under 6 s, red slower). Click for the breakdown.';
    }
  }

  // ------------------------------------------------------------------ voice quota
  async function refreshQuota() {
    const box = $('voice-quota');
    try {
      const r = await api('/api/v1/voice/quota');
      const q = await r.json();
      S.quota = q;
      if (!q.enabled) { box.hidden = true; return; }
      box.hidden = false;
      const left = q.remaining == null ? q.limit : q.remaining;
      const pct = q.limit ? Math.round((left / q.limit) * 100) : 0;
      box.className = 'quota' + (left === 0 || q.global_exhausted ? ' empty' : left <= 1 ? ' low' : '');
      const done = left === 0 || q.global_exhausted;
      let note = q.lifetime ? (done ? 'Voice limit used. You can still type your questions.' : 'Free voice uses (total, not daily)') : (q.window_hours === 24 ? 'today' : 'per ' + q.window_hours + ' h');
      if (done && q.reset_in_seconds) {
        const h = Math.floor(q.reset_in_seconds / 3600), m = Math.round((q.reset_in_seconds % 3600) / 60);
        note = 'resets in ' + (h ? h + ' h ' : '') + m + ' min';
      }
      box.innerHTML = '<div class="quota-row"><span>Voice uses left</span><strong>' + (q.global_exhausted ? 0 : left) + ' / ' + q.limit + '</strong></div>' +
        '<div class="quota-bar"><span style="width:' + (q.global_exhausted ? 0 : pct) + '%"></span></div><div class="quota-note">' + esc(note) + '</div>';
      const mic = $('mic-btn');
      mic.title = done ? 'Voice limit used — type your question instead' : left + ' voice uses left';
      mic.classList.toggle('locked', done);
      mic.setAttribute('aria-disabled', String(done));
      $('attach-btn').classList.toggle('locked', done);
      $('opt-voice').disabled = done;
      if (done) { $('opt-voice').checked = false; S.settings.voiceReply = false; saveSettings(); }
    } catch (e) { box.hidden = true; }
  }
  const quotaBlocked = () => !!(S.quota && S.quota.enabled && (S.quota.remaining === 0 || S.quota.global_exhausted));

  // ------------------------------------------------------------------ stats + recent
  function recordQuery(question, lang, ok, totalMs) {
    const st = store.get(KEYS.stats, { count: 0, ok: 0, ms: 0, timed: 0 });
    st.count += 1; if (ok) st.ok += 1;
    if (totalMs) { st.ms += totalMs; st.timed += 1; }
    store.set(KEYS.stats, st);
    const rec = store.get(KEYS.recent, []);
    rec.unshift({ q: question, lang: lang, t: Date.now() });
    store.set(KEYS.recent, rec.slice(0, 50));
    renderStats(); renderRecent();
  }
  function renderStats() {
    const st = store.get(KEYS.stats, { count: 0, ok: 0, ms: 0, timed: 0 });
    $('stat-total').textContent = st.count.toLocaleString();
    $('stat-avg').textContent = st.timed ? fmtMs(st.ms / st.timed) : '–';
    $('stat-success').textContent = st.count ? ((st.ok / st.count) * 100).toFixed(1) + '%' : '–';
  }
  function recentRow(r) {
    const b = el('button', 'recent-item'); b.type = 'button';
    const d = new Date(r.t);
    const main = el('div', 'recent-main');
    const q = el('div', 'recent-q'); q.textContent = r.q;
    const sub = el('div', 'recent-sub');
    sub.innerHTML = '<span class="recent-lang">' + esc(LANG_LABEL[r.lang] || r.lang || '') + '</span><span class="recent-time">' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + '</span>';
    main.append(q, sub);
    b.append(main);
    b.insertAdjacentHTML('beforeend', svg('chevron'));
    b.addEventListener('click', () => { closeModal(); if (r.lang === 'hi' || r.lang === 'gu') setLang(r.lang, { load: false }); ask(r.q); });
    return b;
  }
  function renderRecent() {
    const box = $('recent-list');
    box.innerHTML = '';
    const rec = store.get(KEYS.recent, []);
    if (!rec.length) { box.appendChild(el('div', 'empty', 'Your recent questions will appear here.')); return; }
    rec.slice(0, 5).forEach((r) => box.appendChild(recentRow(r)));
  }

  // ------------------------------------------------------------------ busy state
  function setBusy(b) {
    S.busy = b;
    $('send-btn').hidden = b;
    $('stop-btn').hidden = !b;
    $('mic-btn').classList.toggle('busy', b && !S.recorder);
    $('mic-btn').disabled = b && !S.recorder;
    $('attach-btn').disabled = b;
    updateSend();
  }
  function updateSend() {
    const v = $('composer-input').value.trim();
    $('send-btn').disabled = S.busy || !v;
    $('char-count').textContent = $('composer-input').value.length + '/1000';
  }
  function pushHistory(q, a) {
    S.history.push({ role: 'user', content: q });
    if (a) S.history.push({ role: 'assistant', content: a.slice(0, 1500) });
    S.history = S.history.slice(-6);
  }

  // ------------------------------------------------------------------ SSE
  async function* readSSE(res) {
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    const parse = (raw) => {
      let event = 'message'; const data = [];
      raw.split('\n').forEach((line) => {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
      });
      if (!data.length) return null;
      try { return { event, data: JSON.parse(data.join('\n')) }; } catch (e) { return null; }
    };
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true }).replace(/\r\n?/g, '\n');
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const ev = parse(buf.slice(0, i)); buf = buf.slice(i + 2);
        if (ev) yield ev;
      }
    }
    buf += dec.decode();
    if (buf.trim()) { const ev = parse(buf); if (ev) yield ev; }
  }

  // ------------------------------------------------------------------ ask (text)
  async function ask(question, queryId) {
    question = (question || '').trim();
    if (!question || S.busy) return;
    stopAllAudio();
    closeDrawer();
    const input = $('composer-input');
    input.value = ''; autoGrow(); updateSend();
    addUserMessage(question);
    const msg = addBotMessage();
    msg.query = question;
    setBusy(true);
    const ctrl = new AbortController();
    S.abort = ctrl;
    const timeout = setTimeout(() => ctrl.abort('timeout'), 120000);
    const ttsParts = [];
    let gotMeta = false, ok = false, total = null;
    try {
      const body = {
        query: question,
        language: S.lang,
        auto_detect_language: S.settings.autoDetect,
        top_k: S.settings.topK,
        use_reranker: S.settings.reranker,
        evaluate: S.settings.evaluate,
        voice_reply: S.settings.voiceReply,
        history: S.history.slice(-6),
      };
      if (queryId != null) body.query_id = queryId;
      const res = await api('/api/v1/query/text/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: ctrl.signal });
      for await (const { event, data } of readSSE(res)) {
        if (event === 'token') { msg.answer += data.token || ''; renderStreaming(msg); }
        else if (event === 'voice_limit') { toast(data.message || 'Voice limit reached.', 'error'); refreshQuota(); }
        else if (event === 'tts') {
          if (data.audio_base64) { const bytes = b64ToBytes(data.audio_base64); ttsParts[data.seq || ttsParts.length] = bytes; enqueueChunk(bytes); }
        } else if (event === 'meta') {
          gotMeta = true; ok = !data.no_answer; total = data.latency && data.latency.total_ms;
          finalizeMessage(msg, Object.assign({ query: question }, data));
          if (S.settings.evaluate) renderEvalPending(msg);
        } else if (event === 'tts_done') {
          msg.ttsMs = data.tts_ms;
          const parts = ttsParts.filter(Boolean);
          if (parts.length) {
            buildPlayer(msg, new Blob(parts, { type: 'audio/mpeg' }), false);
          }
          if (msg.latency) renderLatency(msg, msg.latency);
        } else if (event === 'evaluation') {
          renderEval(msg, data.evaluation, data.eval_ms);
        } else if (event === 'error') {
          throw new Error(data.error || 'The server reported an error.');
        } else if (event === 'done') {
          break;
        }
        scrollToEnd();
      }
      if (!gotMeta) {
        if (msg.answer) finalizeMessage(msg, { answer: msg.answer });
        else throw new Error('The answer stream ended unexpectedly.');
      }
      pushHistory(question, msg.answer);
    } catch (e) {
      const aborted = ctrl.signal.aborted;
      if (!gotMeta) {
        msg.els.text.innerHTML = msg.answer ? formatAnswer(msg.answer) : '';
        if (!msg.answer) msg.els.text.innerHTML = '<p class="searched">' + (aborted ? 'Stopped.' : 'No answer.') + '</p>';
      }
      if (aborted) addNote(msg, 'info', svg('stop') + '<span>' + (ctrl.signal.reason === 'timeout' ? 'The request timed out.' : 'Answer stopped.') + '</span>');
      else addNote(msg, 'err', svg('info') + '<span>' + esc(e.message || 'Something went wrong.') + '</span>');
      ok = false;
    } finally {
      clearTimeout(timeout);
      S.abort = null;
      const caret = msg.els.text.querySelector('.caret'); if (caret) caret.classList.remove('caret');
      setBusy(false);
      recordQuery(question, msg.answerLang || S.lang, ok, total);
      if (S.settings.voiceReply) refreshQuota();
      input.focus();
    }
  }

  // ------------------------------------------------------------------ voice
  async function toggleMic() {
    if (S.recorder) { S.recorder.stop(); return; }
    if (S.busy) return;
    if (quotaBlocked()) { toast('You have used all your free voice uses. You can still type your questions.', 'error'); return; }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === 'undefined') {
      toast(window.isSecureContext ? 'Voice recording is not supported in this browser.' : 'The microphone needs HTTPS or localhost.', 'error');
      return;
    }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }); }
    catch (e) {
      const m = { NotAllowedError: 'Microphone permission was denied.', NotFoundError: 'No microphone was found.', NotReadableError: 'The microphone is in use by another app.' }[e.name];
      toast(m || 'Could not access the microphone.', 'error');
      return;
    }
    stopAllAudio();
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'].find((m) => MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) || '';
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks = [];
    const started = Date.now();
    const timerEl = $('rec-timer');
    const tick = setInterval(() => {
      const s = (Date.now() - started) / 1000;
      timerEl.textContent = fmtClock(s);
      if (s >= 60) rec.stop();
    }, 250);
    rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    rec.onstop = () => {
      clearInterval(tick);
      stream.getTracks().forEach((t) => t.stop());
      S.recorder = null;
      setMicUI(false);
      const type = rec.mimeType || mime || 'audio/webm';
      const blob = new Blob(chunks, { type });
      if (blob.size < 2000 || Date.now() - started < 700) { toast('Recording was too short.'); return; }
      const ext = type.includes('ogg') ? 'ogg' : type.includes('mp4') ? 'm4a' : 'webm';
      sendVoice(blob, 'voice.' + ext);
    };
    S.recorder = rec;
    rec.start(250);
    setMicUI(true);
  }
  function setMicUI(on) {
    const b = $('mic-btn');
    b.classList.toggle('recording', on);
    b.setAttribute('aria-pressed', String(on));
    b.setAttribute('aria-label', on ? 'Stop recording and send' : 'Start voice input');
    b.innerHTML = svg(on ? 'stop' : 'mic');
    $('rec-timer').hidden = !on;
    $('rec-timer').textContent = '0:00';
    $('composer-input').placeholder = on ? 'Listening… tap the red button to send' : placeholderText();
  }

  async function sendVoice(blob, filename) {
    if (blob.size > 10 * 1024 * 1024) { toast('The audio file is larger than 10 MB.', 'error'); return; }
    const user = addUserMessage('🎤 Transcribing…', { voice: true });
    const msg = addBotMessage();
    setBusy(true);
    const ctrl = new AbortController(); S.abort = ctrl;
    let ok = false, total = null, question = '(voice)';
    try {
      const fd = new FormData();
      fd.append('file', blob, filename);
      fd.append('language', S.lang);
      fd.append('auto_detect_language', String(S.settings.autoDetect));
      fd.append('top_k', String(S.settings.topK));
      fd.append('use_reranker', String(S.settings.reranker));
      fd.append('evaluate', String(S.settings.evaluate));
      fd.append('voice_reply', 'true');
      fd.append('history_json', JSON.stringify(S.history.slice(-6)));
      if (S.settings.evaluate) renderEvalPending(msg);
      const res = await api('/api/v1/query/voice', { method: 'POST', body: fd, signal: ctrl.signal });
      const d = await res.json();
      question = d.transcription || d.query || question;
      user.bubble.textContent = question;
      finalizeMessage(msg, d);
      ok = !d.no_answer; total = d.latency && d.latency.total_ms;
      if (d.audio_base64) {
        buildPlayer(msg, new Blob([b64ToBytes(d.audio_base64)], { type: 'audio/mpeg' }), true);
      }
      if (S.settings.evaluate) renderEval(msg, d.evaluation, d.latency && d.latency.eval_ms);
      pushHistory(question, msg.answer);
    } catch (e) {
      user.bubble.textContent = '🎤 Voice message';
      msg.els.text.innerHTML = '';
      msg.els.insights.hidden = true;
      if (msg.insightsBtn) msg.insightsBtn.hidden = true;
      addNote(msg, ctrl.signal.aborted ? 'info' : 'err', svg('info') + '<span>' + esc(ctrl.signal.aborted ? 'Stopped.' : (e.message || 'Voice request failed.')) + '</span>');
    } finally {
      S.abort = null;
      setBusy(false);
      recordQuery(question, msg.answerLang || S.lang, ok, total);
      refreshQuota();
    }
  }

  // ------------------------------------------------------------------ modal
  let lastFocus = null;
  function openModal(title, bodyNode, opts) {
    lastFocus = document.activeElement;
    $('modal-title').textContent = title;
    const body = $('modal-body');
    body.innerHTML = '';
    body.appendChild(bodyNode);
    const dlg = document.querySelector('.modal-dialog');
    dlg.classList.toggle('small', !!(opts && opts.small));
    $('modal').hidden = false;
    hydrateIcons(body);
    const first = body.querySelector('input, button, textarea');
    (first || dlg).focus();
  }
  function closeModal() {
    if ($('modal').hidden) return;
    if ($('modal').dataset.locked === '1') return;
    $('modal').hidden = true;
    setNav('chat');
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  function setNav(name) {
    document.querySelectorAll('.nav-item').forEach((b) => b.classList.toggle('active', b.dataset.nav === name));
  }

  function askApiKey(message) {
    const f = el('form');
    f.innerHTML = '<p class="row-sub" style="margin-bottom:14px">' + esc(message) + '</p>' +
      '<div class="field"><label for="key-input">API key</label><input id="key-input" class="input" type="password" autocomplete="off" spellcheck="false" required></div>' +
      '<div class="btn-row"><button type="submit" class="btn primary">Save key</button></div>';
    f.addEventListener('submit', (e) => {
      e.preventDefault();
      const v = f.querySelector('#key-input').value.trim();
      if (!v) return;
      S.apiKey = v; store.setRaw(KEYS.key, v);
      $('modal').dataset.locked = '0';
      closeModal();
      toast('API key saved');
      loadSamples();
    });
    openModal('API key required', f, { small: true });
    $('modal').dataset.locked = S.authRequired ? '1' : '0';
  }

  function openSamplesModal() {
    setNav('samples');
    const wrap = el('div');
    const types = Array.from(new Set(S.samples.map((q) => q.query_type).filter(Boolean))).sort();
    const filters = el('div', 'filter-row');
    const search = el('input', 'input'); search.type = 'search'; search.placeholder = 'Search questions…'; search.style.width = '100%'; search.style.marginBottom = '12px';
    const list = el('div', 'q-list');
    let active = 'ALL';
    const draw = () => {
      list.innerHTML = '';
      const term = search.value.trim().toLowerCase();
      const items = S.samples.filter((q) => (active === 'ALL' || q.query_type === active) && (!term || q.question.toLowerCase().includes(term)));
      if (!items.length) list.appendChild(el('div', 'empty', 'No matching questions.'));
      items.forEach((q) => {
        const b = el('button', 'q-item'); b.type = 'button';
        const t = el('span', 'q-text'); t.textContent = q.question;
        const ty = el('span', 'q-type'); ty.textContent = q.query_type || '';
        b.append(t, ty);
        b.addEventListener('click', () => { closeModal(); ask(q.question, q.query_id); });
        list.appendChild(b);
      });
    };
    ['ALL'].concat(types).forEach((tp) => {
      const b = el('button', 'filter'); b.type = 'button'; b.textContent = tp === 'ALL' ? 'All' : tp.charAt(0) + tp.slice(1).toLowerCase();
      b.setAttribute('aria-pressed', String(tp === active));
      b.addEventListener('click', () => { active = tp; filters.querySelectorAll('.filter').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); draw(); });
      filters.appendChild(b);
    });
    search.addEventListener('input', draw);
    wrap.append(search, filters, list);
    draw();
    openModal('Sample Queries · ' + L[S.lang].name, wrap);
  }

  function openEvalModal() {
    setNav('evaluation');
    const log = store.get(KEYS.evals, []);
    const wrap = el('div');
    const avg = (key) => {
      const vals = log.map((x) => (key === 'overall' ? x.overall : x.s && x.s[key])).filter((v) => typeof v === 'number');
      return vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(2) : '–';
    };
    const tiles = [['Evaluated answers', String(log.length)], ['Overall', avg('overall')], ['Faithfulness', avg('faithfulness')], ['Answer Relevance', avg('answer_relevance')], ['Context Recall', avg('context_recall')], ['Correctness', avg('answer_correctness')]];
    const grid = el('div', 'eval-summary');
    tiles.forEach(([k, v]) => grid.appendChild(el('div', 'eval-tile', '<div class="k">' + k + '</div><div class="v">' + v + '</div>')));
    const row = el('div', 'row');
    row.innerHTML = '<div class="row-text"><span class="row-title">Evaluate every answer</span><span class="row-sub">Runs DeepEval after each answer. Adds a few seconds and judge API calls.</span></div>';
    const sw = el('label', 'switch'); sw.innerHTML = '<input type="checkbox"><span class="track"></span>';
    const cb = sw.querySelector('input'); cb.checked = S.settings.evaluate;
    cb.addEventListener('change', () => { S.settings.evaluate = cb.checked; $('opt-eval').checked = cb.checked; saveSettings(); });
    row.appendChild(sw);
    const actions = el('div', 'btn-row');
    const clear = el('button', 'btn danger', 'Clear evaluation log'); clear.type = 'button';
    clear.addEventListener('click', () => { store.set(KEYS.evals, []); closeModal(); toast('Evaluation log cleared'); });
    actions.appendChild(clear);
    wrap.append(grid, row, actions);
    openModal('Evaluation', wrap);
  }

  function openSettingsModal() {
    setNav('settings');
    const wrap = el('div');
    const toggle = (title, sub, key, mirrorId) => {
      const row = el('div', 'row');
      row.innerHTML = '<div class="row-text"><span class="row-title">' + title + '</span><span class="row-sub">' + sub + '</span></div>';
      const sw = el('label', 'switch'); sw.innerHTML = '<input type="checkbox"><span class="track"></span>';
      const cb = sw.querySelector('input'); cb.checked = !!S.settings[key]; cb.setAttribute('aria-label', title);
      cb.addEventListener('change', () => { S.settings[key] = cb.checked; if (mirrorId) $(mirrorId).checked = cb.checked; saveSettings(); });
      row.appendChild(sw);
      return row;
    };
    wrap.append(
      toggle('Voice reply', 'Speak typed answers aloud (voice questions always get a spoken reply).', 'voiceReply', 'opt-voice'),
      toggle('Evaluate answers', 'Score each answer with DeepEval.', 'evaluate', 'opt-eval'),
      toggle('Auto-detect language', 'Route by the script of your question.', 'autoDetect', 'opt-auto'),
      toggle('Re-ranker', 'Re-rank retrieved passages for better precision (slower).', 'reranker', null),
    );
    const kRow = el('div', 'row');
    kRow.innerHTML = '<div class="row-text"><span class="row-title">Sources per answer</span><span class="row-sub">How many passages the answer is based on.</span></div>';
    const k = el('input', 'input'); k.type = 'number'; k.min = 1; k.max = 20; k.value = S.settings.topK; k.style.width = '80px'; k.setAttribute('aria-label', 'Sources per answer');
    k.addEventListener('change', () => { S.settings.topK = Math.max(1, Math.min(20, parseInt(k.value, 10) || 5)); k.value = S.settings.topK; saveSettings(); });
    kRow.appendChild(k);
    const keyRow = el('div', 'row');
    keyRow.innerHTML = '<div class="row-text"><span class="row-title">API key</span><span class="row-sub">' + (S.apiKey ? 'A key is saved in this browser.' : (S.authRequired ? 'Required by this server.' : 'Not required by this server.')) + '</span></div>';
    const kb = el('button', 'btn secondary', S.apiKey ? 'Change' : 'Set key'); kb.type = 'button';
    kb.addEventListener('click', () => askApiKey('Enter the API key configured on the server (RAG_API_KEY).'));
    keyRow.appendChild(kb);
    const actions = el('div', 'btn-row');
    const clear = el('button', 'btn danger', 'Clear history & stats'); clear.type = 'button';
    clear.addEventListener('click', () => { store.set(KEYS.recent, []); store.set(KEYS.stats, { count: 0, ok: 0, ms: 0, timed: 0 }); renderStats(); renderRecent(); toast('History cleared'); });
    const legacy = el('a', 'btn secondary', 'Open classic UI'); legacy.href = '/static/legacy.html';
    actions.append(legacy, clear);
    wrap.append(kRow, keyRow, actions);
    openModal('Settings', wrap);
  }

  function openRecentModal() {
    const wrap = el('div', 'recent-list');
    const rec = store.get(KEYS.recent, []);
    if (!rec.length) wrap.appendChild(el('div', 'empty', 'No questions yet.'));
    rec.forEach((r) => wrap.appendChild(recentRow(r)));
    openModal('Recent Conversation', wrap);
  }

  // ------------------------------------------------------------------ misc UI
  function autoGrow() {
    const ta = $('composer-input');
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  }
  function newChat() {
    if (S.abort) S.abort.abort();
    if (S.recorder) S.recorder.stop();
    stopAllAudio();
    document.querySelectorAll('.msg').forEach((m) => m.remove());
    S.history = [];
    document.querySelector('.chat-card').classList.remove('has-msgs');
    renderChips();
    $('composer-input').focus();
  }
  function openDrawer() {
    $('sidebar-left').classList.add('open'); $('drawer-backdrop').hidden = false;
    $('menu-btn').setAttribute('aria-expanded', 'true');
  }
  function closeDrawer() {
    $('sidebar-left').classList.remove('open'); $('drawer-backdrop').hidden = true;
    $('menu-btn').setAttribute('aria-expanded', 'false');
  }

  // ------------------------------------------------------------------ wiring
  function init() {
    hydrateIcons();
    syncThemeIcon();
    mqDark.addEventListener && mqDark.addEventListener('change', syncThemeIcon);
    $('theme-btn').addEventListener('click', toggleTheme);
    document.querySelectorAll('.seg-btn').forEach((b) => b.addEventListener('click', () => { if (b.dataset.lang !== S.lang) setLang(b.dataset.lang); }));
    $('profile-btn').addEventListener('click', openSettingsModal);
    $('menu-btn').addEventListener('click', () => ($('sidebar-left').classList.contains('open') ? closeDrawer() : openDrawer()));
    $('drawer-backdrop').addEventListener('click', closeDrawer);
    document.querySelectorAll('.nav-item').forEach((b) => b.addEventListener('click', () => {
      closeDrawer();
      const n = b.dataset.nav;
      if (n === 'chat') { setNav('chat'); $('composer-input').focus(); }
      else if (n === 'samples') openSamplesModal();
      else if (n === 'evaluation') openEvalModal();
      else if (n === 'settings') openSettingsModal();
    }));
    $('refresh-samples').addEventListener('click', () => (S.samples.length ? renderChips() : loadSamples()));
    $('new-chat-btn').addEventListener('click', newChat);
    $('view-all-recent').addEventListener('click', openRecentModal);

    // options
    $('opt-voice').checked = S.settings.voiceReply;
    $('opt-eval').checked = S.settings.evaluate;
    $('opt-auto').checked = S.settings.autoDetect;
    $('opt-voice').addEventListener('change', (e) => { S.settings.voiceReply = e.target.checked; saveSettings(); });
    $('opt-eval').addEventListener('change', (e) => { S.settings.evaluate = e.target.checked; saveSettings(); });
    $('opt-auto').addEventListener('change', (e) => { S.settings.autoDetect = e.target.checked; saveSettings(); });

    // composer
    const input = $('composer-input');
    input.addEventListener('input', () => { autoGrow(); updateSend(); });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) { e.preventDefault(); ask(input.value); }
    });
    $('composer').addEventListener('submit', (e) => { e.preventDefault(); ask(input.value); });
    $('stop-btn').addEventListener('click', () => { if (S.abort) S.abort.abort('user'); stopAllAudio(); });
    $('mic-btn').addEventListener('click', toggleMic);
    $('attach-btn').addEventListener('click', () => { if (quotaBlocked()) { toast('You have used all your free voice uses.', 'error'); return; } $('file-input').click(); });
    $('file-input').addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      e.target.value = '';
      if (f && !S.busy) sendVoice(f, f.name);
    });

    // modal
    $('modal').addEventListener('click', (e) => { if (e.target === $('modal') || e.target.closest('[data-close]')) closeModal(); });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { if (!$('modal').hidden) closeModal(); else closeDrawer(); }
      if (e.key === 'Tab' && !$('modal').hidden) {
        const f = Array.from(document.querySelectorAll('.modal-dialog button, .modal-dialog input, .modal-dialog a, .modal-dialog textarea')).filter((x) => !x.disabled && x.offsetParent !== null);
        if (!f.length) return;
        if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
        else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
      }
    });

    mqNarrow.addEventListener && mqNarrow.addEventListener('change', () => { if (!S.recorder) $('composer-input').placeholder = placeholderText(); });
    setLang(S.lang, { load: false });
    renderStats();
    renderRecent();
    updateSend();
    checkHealth().then(() => { loadSamples(); refreshQuota(); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
