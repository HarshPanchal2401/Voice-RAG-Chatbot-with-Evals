/**
 * Multi-Language Voice & Text RAG — client application (v5.0.0)
 * ===============================================================
 * Plain browser script, no build step. Requires /static/utils.js (window.RagUtils).
 *
 * Security rules followed throughout this file:
 *  - no inline event handlers; every listener is attached with addEventListener;
 *  - untrusted text (answers, sources, transcripts, history, server/language lists,
 *    localStorage values) is written with textContent only;
 *  - answers are rendered with RagUtils.formatAnswerHtml, which escapes first.
 */
(function () {
  'use strict';

  const U = window.RagUtils;
  if (!U) {
    console.error('utils.js did not load; the UI cannot start.');
    return;
  }

  // ============================================================
  // Constants
  // ============================================================
  const ASSET_VERSION = '5.0.0';
  const TOP_K = 5;
  const HISTORY_MESSAGES = 6;             // conversation memory sent to the server
  const MAX_QUERY_CHARS = 1000;
  const MAX_UPLOAD_MB = 10;
  const TEXT_IDLE_TIMEOUT_MS = 60000;      // abort a stream that stays silent this long (before the answer)
  const AFTER_ANSWER_IDLE_MS = 150000;     // TTS + DeepEval can be quiet for a while after `meta`
  const TEXT_TOTAL_TIMEOUT_MS = 300000;
  const UPLOAD_TIMEOUT_MS = 300000;
  const VOICE_MAX_RECORD_MS = 60000;       // matches RAG_WS_MAX_SECONDS default
  const VOICE_ANSWER_IDLE_MS = 120000;     // no server message for this long after "stop" -> give up
  const HEALTH_INTERVAL_MS = 30000;
  const AUDIO_EXTENSIONS = ['wav', 'mp3', 'm4a', 'ogg', 'webm', 'flac', 'mp4'];

  const STORAGE_KEYS = {
    lang: 'voice_rag_lang',
    history: 'voice_rag_history',
    theme: 'voice_rag_theme',
    apiKey: 'voice_rag_api_key',
    autoDetect: 'voice_rag_auto_detect',
  };

  /** UI copy per language (the server list from /api/v1/languages decides what is offered). */
  const LANGUAGE_COPY = {
    gu: {
      code: 'gu',
      locale: 'gu-IN',
      name: 'Gujarati',
      nativeName: 'ગુજરાતી',
      placeholder: 'ગુજરાતી માં પ્રશ્ન લખો... (Type your question in Gujarati or English)',
      speakPrompt: 'Speak in Gujarati',
      welcomeMsg: 'નમસ્તે! હું તમારો ગુજરાતી AI સહાયક છું. તમે મને બોલીને (Voice) અથવા લખીને (Text) કોઈ પણ પ્રશ્ન પૂછી શકો છો.',
      appTitle: 'Gujarati Voice RAG',
      appSubtitle: 'Voice + Text RAG for Gujarati',
      samplePoolFile: '/static/golden_sample_queries.json',
      thinkingText: 'વિચાર કરી રહ્યો છે...',
      errPrefix: 'ક્ષમા કરશો, પ્રશ્નનો ઉત્તર મેળવવામાં સમસ્યા આવી',
    },
    hi: {
      code: 'hi',
      locale: 'hi-IN',
      name: 'Hindi',
      nativeName: 'हिन्दी',
      placeholder: 'हिंदी में प्रश्न पूछें... (Type your question in Hindi or English)',
      speakPrompt: 'Speak in Hindi',
      welcomeMsg: 'नमस्ते! मैं आपका हिंदी AI सहायक हूँ। आप मुझसे बोलकर (Voice) या लिखकर (Text) कोई भी प्रश्न पूछ सकते हैं।',
      appTitle: 'Hindi Voice RAG',
      appSubtitle: 'Voice + Text RAG for Hindi',
      samplePoolFile: '/static/golden_sample_queries_hi.json',
      thinkingText: 'विचार कर रहा हूँ...',
      errPrefix: 'क्षमा करें, उत्तर प्राप्त करने में समस्या आई',
    },
  };

  const ANSWER_LANGUAGE_NAMES = { gu: 'Gujarati', hi: 'Hindi', en: 'English' };

  const store = U.createSafeStorage(() => window.localStorage);
  const $ = (id) => document.getElementById(id);

  // ============================================================
  // State
  // ============================================================
  const STATE = {
    lang: 'gu',
    languages: [],            // [{code, name, nativeName, locale, active}] offered in the menu
    mode: 'voice',            // 'voice' | 'chat'
    busy: null,               // null | 'text' | 'upload' | 'voice' (one global busy state)
    voice: 'idle',            // 'idle' | 'connecting' | 'recording' | 'processing'
    request: null,            // active fetch request handle (text / upload)
    recent: [],               // sidebar query history (persisted)
    session: [],              // conversation memory for this session: [{role, content}]
    sessionEpoch: 0,          // bumped on New Session; stale callbacks compare against it
    pendingSample: null,      // {text, queryId} while a sample sits unedited in the input
    samples: [],
    modalCategory: 'all',
    apiKey: '',
    authRequired: false,
    stickToBottom: true,
    lastMessageId: null,
    lastEvalLanguage: null,
  };

  const messages = new Map(); // id -> bot message object
  let messageCounter = 0;
  let sampleRequestSeq = 0;
  let healthTimer = null;
  let voiceToken = 0;
  let liveVoiceWs = null;
  const voice = { ctx: null, capture: null, maxTimer: null, answerTimer: null };
  const cachedPools = {};

  // ============================================================
  // Tiny DOM helpers
  // ============================================================
  /**
   * h('div', {className, text, attrs, hidden, title, type, on}, ...children)
   * Text is always set via textContent.
   */
  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    const p = props || {};
    if (p.className) el.className = p.className;
    if (p.text !== undefined && p.text !== null) el.textContent = String(p.text);
    if (p.title) el.title = p.title;
    if (p.type) el.type = p.type;
    if (p.hidden) el.hidden = true;
    if (p.attrs) {
      Object.keys(p.attrs).forEach((k) => {
        if (p.attrs[k] !== undefined && p.attrs[k] !== null) el.setAttribute(k, String(p.attrs[k]));
      });
    }
    if (p.on) {
      Object.keys(p.on).forEach((evt) => el.addEventListener(evt, p.on[evt]));
    }
    children.forEach((child) => {
      if (child === null || child === undefined || child === false) return;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    });
    return el;
  }

  function on(id, event, handler) {
    const el = $(id);
    if (el) el.addEventListener(event, handler);
    return el;
  }

  function setText(id, value) {
    const el = $(id);
    if (el) el.textContent = value;
  }

  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function langCopy(code) {
    if (LANGUAGE_COPY[code]) return LANGUAGE_COPY[code];
    const server = STATE.languages.find((l) => l.code === code);
    const name = server ? server.name : String(code || '').toUpperCase();
    return {
      code,
      locale: server ? server.locale : code,
      name,
      nativeName: server ? server.nativeName : name,
      placeholder: 'Type your question…',
      speakPrompt: 'Speak in ' + name,
      welcomeMsg: 'Hello! Ask me a question by voice or text.',
      appTitle: name + ' Voice RAG',
      appSubtitle: 'Voice + Text RAG for ' + name,
      samplePoolFile: null,
      thinkingText: 'Thinking…',
      errPrefix: 'Sorry, something went wrong',
    };
  }

  function langLabel(code) {
    const c = langCopy(code);
    return c.nativeName && c.nativeName !== c.name ? c.nativeName + ' (' + c.name + ')' : c.name;
  }

  // ============================================================
  // Toasts
  // ============================================================
  function showToast(message, kind) {
    const region = $('toast-region');
    if (!region) return;
    const toast = h('div', { className: 'toast' + (kind === 'error' ? ' toast-error' : ''), text: message });
    region.append(toast);
    setTimeout(() => toast.classList.add('toast-hide'), kind === 'error' ? 5200 : 3200);
    setTimeout(() => toast.remove(), kind === 'error' ? 5600 : 3600);
  }

  // ============================================================
  // Networking helpers (API key, errors)
  // ============================================================
  function apiHeaders(extra) {
    const headers = new Headers(extra || {});
    if (STATE.apiKey) headers.set('X-API-Key', STATE.apiKey);
    return headers;
  }

  /** fetch() for /api/* routes: adds X-API-Key and reacts to 401. */
  async function apiFetch(path, options) {
    const opts = Object.assign({}, options || {});
    opts.headers = apiHeaders(opts.headers);
    const res = await fetch(path, opts);
    if (res.status === 401) handleUnauthorized();
    return res;
  }

  async function httpError(res, opts) {
    let detail = '';
    try {
      const type = res.headers.get('content-type') || '';
      if (type.includes('json')) {
        const body = await res.json();
        detail = U.formatErrorDetail(body && (body.detail !== undefined ? body.detail : (body.error || body.message)));
      } else {
        detail = (await res.text()).trim().slice(0, 300);
      }
    } catch (e) {
      /* body unreadable */
    }
    const err = new Error(U.describeHttpError(res.status, detail, res.headers.get('Retry-After'), opts));
    err.status = res.status;
    return err;
  }

  function timeoutSignal(ms) {
    if (window.AbortSignal && typeof AbortSignal.timeout === 'function') return AbortSignal.timeout(ms);
    const controller = new AbortController();
    setTimeout(() => controller.abort(), ms);
    return controller.signal;
  }

  function handleUnauthorized(message) {
    if (STATE.apiKey) {
      STATE.apiKey = '';
      store.remove(STORAGE_KEYS.apiKey);
    }
    openApiKeyModal(message || 'The server rejected the request. Please enter a valid API key.');
  }

  // ============================================================
  // Modal manager (focus trap, Escape, focus restore)
  // ============================================================
  const Modal = (() => {
    const stack = [];
    const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

    function focusables(root) {
      return Array.from(root.querySelectorAll(FOCUSABLE)).filter((el) => !el.closest('[hidden]') && el.getClientRects().length > 0);
    }

    function open(backdrop, options) {
      if (!backdrop || stack.some((e) => e.backdrop === backdrop)) return;
      const opts = options || {};
      stack.push({ backdrop, restore: document.activeElement, onClose: opts.onClose });
      backdrop.hidden = false;
      document.body.classList.add('modal-open');
      const dialog = backdrop.querySelector('[role="dialog"]');
      requestAnimationFrame(() => {
        const target = (opts.initialFocus && backdrop.querySelector(opts.initialFocus)) || focusables(backdrop)[0] || dialog;
        if (target) target.focus();
      });
    }

    function close(backdrop) {
      const idx = stack.findIndex((e) => e.backdrop === backdrop);
      if (idx === -1) return;
      const entry = stack.splice(idx, 1)[0];
      backdrop.hidden = true;
      if (!stack.length) document.body.classList.remove('modal-open');
      if (entry.restore && document.contains(entry.restore) && typeof entry.restore.focus === 'function') {
        entry.restore.focus();
      }
      if (typeof entry.onClose === 'function') entry.onClose();
    }

    function top() {
      return stack.length ? stack[stack.length - 1] : null;
    }

    function handleKeydown(e) {
      const current = top();
      if (!current) return false;
      if (e.key === 'Escape') {
        e.preventDefault();
        close(current.backdrop);
        return true;
      }
      if (e.key === 'Tab') {
        const items = focusables(current.backdrop);
        if (!items.length) {
          e.preventDefault();
          return true;
        }
        const first = items[0];
        const last = items[items.length - 1];
        const active = document.activeElement;
        if (!current.backdrop.contains(active)) {
          e.preventDefault();
          first.focus();
        } else if (e.shiftKey && (active === first || !items.includes(active))) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && active === last) {
          e.preventDefault();
          first.focus();
        }
        return true;
      }
      return false;
    }

    function wire(backdrop) {
      if (!backdrop) return;
      let downOnBackdrop = false;
      backdrop.addEventListener('mousedown', (e) => {
        downOnBackdrop = e.target === backdrop;
      });
      backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop && downOnBackdrop) close(backdrop);
        downOnBackdrop = false;
      });
      backdrop.querySelectorAll('[data-close-modal]').forEach((btn) => {
        btn.addEventListener('click', () => close(backdrop));
      });
    }

    return { open, close, top, handleKeydown, wire, isOpen: (b) => stack.some((e) => e.backdrop === b) };
  })();

  // ============================================================
  // Audio player (one shared player, ordered queue of blob URLs)
  // ============================================================
  const Player = (() => {
    const audio = new Audio();
    audio.preload = 'auto';
    let queue = [];
    let owner = null;
    let playing = false;
    const listeners = new Set();

    function notify() {
      listeners.forEach((fn) => {
        try {
          fn();
        } catch (e) {
          console.error(e);
        }
      });
    }

    function playNext() {
      const next = queue.shift();
      if (!next) {
        playing = false;
        notify();
        return;
      }
      playing = true;
      audio.src = next;
      const promise = audio.play();
      if (promise && typeof promise.catch === 'function') {
        promise.catch((err) => {
          if (!playing || audio.src !== next) return;
          console.warn('Audio playback failed:', err);
          if (err && err.name === 'NotAllowedError') {
            queue = [];
            playing = false;
            notify();
            showToast('Autoplay was blocked by the browser — press “Play answer” to listen.');
          } else {
            playNext();
          }
        });
      }
      notify();
    }

    audio.addEventListener('ended', () => {
      if (playing) playNext();
    });
    audio.addEventListener('error', () => {
      if (playing) playNext();
    });

    return {
      /** Append to the queue of `ownerId`; switching owner stops the previous audio. */
      enqueue(url, ownerId) {
        if (owner !== ownerId) {
          this.stop();
          owner = ownerId;
        }
        queue.push(url);
        if (!playing) playNext();
      },
      playAll(urls, ownerId) {
        this.stop();
        if (!urls.length) return;
        owner = ownerId;
        queue = urls.slice();
        playNext();
      },
      stop() {
        const wasActive = playing || queue.length > 0;
        queue = [];
        playing = false;
        owner = null;
        try {
          audio.pause();
        } catch (e) {
          /* ignore */
        }
        audio.removeAttribute('src');
        try {
          audio.load();
        } catch (e) {
          /* ignore */
        }
        if (wasActive) notify();
      },
      isPlaying(ownerId) {
        return playing && owner === ownerId;
      },
      onChange(fn) {
        listeners.add(fn);
      },
    };
  })();

  /** Stop all speech and stop streaming chunks from auto-playing. */
  function stopAllAudio() {
    messages.forEach((m) => {
      m.autoplay = false;
    });
    Player.stop();
  }

  Player.onChange(() => messages.forEach(updateAudioButton));

  // ============================================================
  // Busy state & controls
  // ============================================================
  function setBusy(kind) {
    STATE.busy = kind;
    refreshControls();
  }

  function clearBusy(kind) {
    if (STATE.busy === kind) {
      STATE.busy = null;
      refreshControls();
    }
  }

  function refreshControls() {
    const busy = !!STATE.busy;
    const send = $('send-query-btn');
    const upload = $('upload-audio-btn');
    const mic = $('main-mic-btn');
    const stop = $('stop-generation-btn');
    if (send) send.disabled = busy;
    if (upload) upload.disabled = busy;
    if (mic) {
      mic.disabled = (busy && STATE.busy !== 'voice') || STATE.voice === 'connecting' || STATE.voice === 'processing';
    }
    if (stop) {
      stop.hidden = !(STATE.busy === 'text' || STATE.busy === 'upload' || STATE.voice === 'processing' || STATE.voice === 'connecting');
    }
    document.querySelectorAll('.btn-ask-query').forEach((b) => {
      b.disabled = busy;
    });
  }

  /** Request handle with AbortController + idle/total timeouts. */
  function beginRequest(kind, msg, timeouts) {
    const t = Object.assign({}, timeouts || {});
    const controller = new AbortController();
    let idleTimer = null;
    const req = {
      kind,
      msg,
      signal: controller.signal,
      cancelled: false,
      reason: null,
      cancel(reason) {
        if (req.cancelled) return;
        req.cancelled = true;
        req.reason = reason || 'user';
        try {
          controller.abort();
        } catch (e) {
          /* ignore */
        }
      },
      touch() {
        if (!t.idleMs) return;
        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => req.cancel('timeout'), t.idleMs);
      },
      setIdle(ms) {
        t.idleMs = ms;
        req.touch();
      },
      dispose() {
        clearTimeout(idleTimer);
        clearTimeout(totalTimer);
      },
    };
    const totalTimer = t.totalMs ? setTimeout(() => req.cancel('timeout'), t.totalMs) : null;
    req.touch();
    STATE.request = req;
    setBusy(kind);
    return req;
  }

  function endRequest(req) {
    req.dispose();
    if (STATE.request === req) {
      STATE.request = null;
      clearBusy(req.kind);
    }
  }

  /** Cancel whatever is running (fetch request and/or live voice). */
  function cancelActive(reason) {
    if (STATE.request) STATE.request.cancel(reason);
    if (STATE.voice !== 'idle') cancelVoice(reason);
  }

  function cancelReasonText(reason) {
    switch (reason) {
      case 'timeout':
        return 'The request timed out. Please try again.';
      case 'language':
        return 'Cancelled because the language was changed.';
      case 'mode':
        return 'Cancelled.';
      default:
        return 'Stopped.';
    }
  }

  // ============================================================
  // Scrolling (the scroll container is .canvas-container)
  // ============================================================
  function scroller() {
    return $('canvas-container');
  }

  function scrollToBottom(force) {
    const sc = scroller();
    if (!sc) return;
    if (!force && !STATE.stickToBottom) return;
    requestAnimationFrame(() => {
      sc.scrollTop = sc.scrollHeight;
      STATE.stickToBottom = true;
    });
  }

  // ============================================================
  // Chat rendering
  // ============================================================
  function appendUserMessage(text, time, isVoice) {
    const feed = $('chat-feed');
    if (!feed) return null;
    const textEl = h('div', { className: 'message-text', text });
    const row = h('div', { className: 'message-row user-row' },
      h('div', { className: 'message-avatar', attrs: { 'aria-hidden': 'true' }, text: '👤' }),
      h('div', { className: 'message-bubble' },
        h('div', { className: 'message-header' },
          h('span', { className: 'message-sender', text: isVoice ? '🎙️ You' : 'You' }),
          h('span', { className: 'message-time', text: time })),
        textEl));
    feed.append(row);
    return { row, textEl };
  }

  /** Creates a bot bubble. Parts are updated individually later (never rebuilt). */
  function createBotMessage(opts) {
    const o = opts || {};
    const feed = $('chat-feed');
    const id = 'm' + (++messageCounter);

    const els = {};
    els.langTag = h('span', { className: 'message-lang-tag', hidden: true });
    els.evalChip = h('span', { className: 'eval-chip-inline', hidden: true });
    els.header = h('div', { className: 'message-header' },
      h('span', { className: 'message-sender', text: 'Answer' }),
      els.langTag,
      h('span', { className: 'message-time', text: o.time || U.formatClock(new Date()) }),
      els.evalChip);
    els.retrievalNote = h('div', { className: 'retrieval-note', hidden: true });
    els.noAnswer = h('div', { className: 'no-answer-badge', hidden: true, text: 'No answer found in the indexed sources' });
    els.text = h('div', { className: 'message-text bot-body-text' });
    if (o.thinking) {
      els.text.append(h('span', { className: 'loading-dots', text: o.thinking }));
    }
    els.warning = h('div', { className: 'message-warning', hidden: true, attrs: { role: 'alert' } });

    els.audioBtn = h('button', { className: 'audio-play-pill', type: 'button', text: '🔊 Listen', hidden: true });
    els.sourcesBtn = h('button', {
      className: 'sources-toggle-btn',
      type: 'button',
      hidden: true,
      attrs: { 'aria-expanded': 'false', 'aria-controls': id + '-sources' },
    });
    els.drawer = h('div', { className: 'sources-drawer', hidden: true, attrs: { id: id + '-sources' } });

    els.likeBtn = h('button', { className: 'msg-action-btn', type: 'button', text: '👍', title: 'Helpful', hidden: true, attrs: { 'aria-label': 'Mark answer as helpful', 'aria-pressed': 'false' } });
    els.dislikeBtn = h('button', { className: 'msg-action-btn', type: 'button', text: '👎', title: 'Not helpful', hidden: true, attrs: { 'aria-label': 'Mark answer as not helpful', 'aria-pressed': 'false' } });
    els.copyBtn = h('button', { className: 'msg-action-btn', type: 'button', text: '📋', title: 'Copy answer', hidden: true, attrs: { 'aria-label': 'Copy answer' } });
    els.actions = h('div', { className: 'message-actions', hidden: true }, els.likeBtn, els.dislikeBtn, els.copyBtn);

    els.bubble = h('div', { className: 'message-bubble', attrs: { 'aria-busy': o.thinking ? 'true' : 'false' } },
      els.header, els.retrievalNote, els.noAnswer, els.text, els.warning, els.audioBtn, els.sourcesBtn, els.drawer, els.actions);
    const row = h('div', { className: 'message-row bot-row' },
      h('div', { className: 'message-avatar', attrs: { 'aria-hidden': 'true' }, text: '✦' }),
      els.bubble);

    const msg = {
      id,
      row,
      els,
      answer: '',
      streamNode: null,
      pendingTokens: '',
      rafId: 0,
      language: o.language || STATE.lang,
      answerLanguage: null,
      traceId: null,
      evaluation: null,
      evalRequested: false,
      evaluationReceived: false,
      latency: null,
      audio: new Map(),               // seq -> blob URL (kept for replay)
      seqBuffer: U.createSeqBuffer(0),
      autoplay: false,
      finished: false,
      query: o.query || '',
    };

    els.audioBtn.addEventListener('click', () => onAudioButton(msg));
    els.sourcesBtn.addEventListener('click', () => toggleSources(msg));
    els.copyBtn.addEventListener('click', () => copyText(msg.answer));
    els.likeBtn.addEventListener('click', () => sendFeedback(msg, 1, els.likeBtn));
    els.dislikeBtn.addEventListener('click', () => sendFeedback(msg, 0, els.dislikeBtn));

    messages.set(id, msg);
    STATE.lastMessageId = id;
    if (feed) feed.append(row);
    return msg;
  }

  function appendWelcomeMessage() {
    const copy = langCopy(STATE.lang);
    const msg = createBotMessage({ time: U.formatClock(new Date()) });
    msg.answer = copy.welcomeMsg;
    msg.language = STATE.lang;
    msg.answerLanguage = STATE.lang;
    msg.isWelcome = true;
    msg.finished = true;
    msg.els.text.innerHTML = U.formatAnswerHtml(copy.welcomeMsg);
    msg.els.actions.hidden = false;
    msg.els.copyBtn.hidden = false;
    msg.els.audioBtn.hidden = false;
    updateAudioButton(msg);
    return msg;
  }

  /** Streaming: tokens go into ONE text node, flushed once per animation frame. */
  function msgAppendToken(msg, token) {
    if (!msg || !token || msg.finished) return;
    if (!msg.streamNode) {
      msg.els.text.textContent = '';
      msg.els.text.classList.add('streaming');
      msg.streamNode = document.createTextNode('');
      msg.els.text.append(msg.streamNode);
    }
    msg.pendingTokens += token;
    if (!msg.rafId) {
      msg.rafId = requestAnimationFrame(() => {
        msg.rafId = 0;
        if (msg.streamNode && msg.pendingTokens) {
          msg.streamNode.appendData(msg.pendingTokens);
          msg.pendingTokens = '';
          scrollToBottom(false);
        }
      });
    }
  }

  function stopStreaming(msg) {
    if (msg.rafId) {
      cancelAnimationFrame(msg.rafId);
      msg.rafId = 0;
    }
    msg.pendingTokens = '';
    msg.streamNode = null;
    msg.els.text.classList.remove('streaming');
  }

  function streamedText(msg) {
    const current = msg.streamNode ? msg.streamNode.data : '';
    return current + (msg.pendingTokens || '');
  }

  /** Final answer + metadata (SSE `meta`, WS `answer`, upload response). */
  function msgSetAnswer(msg, data) {
    const answer = typeof data.answer === 'string' && data.answer ? data.answer : streamedText(msg);
    stopStreaming(msg);
    msg.answer = answer;
    msg.finished = true;
    msg.language = U.isLangCode(data.language) ? data.language : msg.language;
    msg.answerLanguage = U.isLangCode(data.answer_language) ? data.answer_language : msg.language;
    msg.traceId = typeof data.trace_id === 'string' && data.trace_id ? data.trace_id : null;
    msg.els.bubble.setAttribute('aria-busy', 'false');
    msg.els.text.innerHTML = U.formatAnswerHtml(answer);

    // Answer language label (server-reported, not the dropdown value)
    const tag = msg.els.langTag;
    tag.textContent = msg.answerLanguage.toUpperCase();
    tag.title = 'Answer language: ' + (ANSWER_LANGUAGE_NAMES[msg.answerLanguage] || msg.answerLanguage) +
      (msg.language && msg.language !== msg.answerLanguage ? ' · index: ' + msg.language.toUpperCase() : '');
    tag.hidden = false;

    // "Searched for: …" when the condensed retrieval query differs
    const query = data.query || msg.query;
    if (U.queriesDiffer(query, data.retrieval_query)) {
      msg.els.retrievalNote.textContent = '';
      msg.els.retrievalNote.append(h('span', { className: 'retrieval-note-label', text: 'Searched for: ' }), String(data.retrieval_query));
      msg.els.retrievalNote.hidden = false;
    }

    // No answer found in the sources
    const noAnswer = data.no_answer === true;
    msg.els.noAnswer.hidden = !noAnswer;
    msg.row.classList.toggle('no-answer', noAnswer);

    renderSources(msg, Array.isArray(data.sources) ? data.sources : []);

    msg.els.actions.hidden = false;
    msg.els.copyBtn.hidden = !answer;
    msg.els.likeBtn.hidden = !msg.traceId;
    msg.els.dislikeBtn.hidden = !msg.traceId;
    msg.els.audioBtn.hidden = !answer;
    updateAudioButton(msg);
    scrollToBottom(false);
  }

  function msgSetError(msg, text) {
    if (!msg) return;
    stopStreaming(msg);
    msg.finished = true;
    msg.row.classList.add('message-error');
    msg.els.bubble.setAttribute('aria-busy', 'false');
    msg.els.text.textContent = '⚠️ ' + text;
    msg.els.audioBtn.hidden = true;
    msg.els.actions.hidden = true;
    scrollToBottom(false);
  }

  /** Non-destructive warning under an answer that is already shown. */
  function msgAddWarning(msg, text) {
    if (!msg) return;
    msg.els.warning.textContent = '⚠️ ' + text;
    msg.els.warning.hidden = false;
  }

  function msgSetCancelled(msg, reason, hadAnswer) {
    if (!msg) return;
    if (hadAnswer) {
      msgAddWarning(msg, cancelReasonText(reason));
      return;
    }
    const partial = streamedText(msg);
    stopStreaming(msg);
    msg.finished = true;
    msg.els.bubble.setAttribute('aria-busy', 'false');
    msg.row.classList.add('message-cancelled');
    if (partial.trim()) {
      msg.answer = partial;
      msg.els.text.innerHTML = U.formatAnswerHtml(partial);
      msgAddWarning(msg, cancelReasonText(reason));
      msg.els.actions.hidden = false;
      msg.els.copyBtn.hidden = false;
    } else {
      msg.els.text.textContent = cancelReasonText(reason);
    }
  }

  function renderSources(msg, sources) {
    const btn = msg.els.sourcesBtn;
    const drawer = msg.els.drawer;
    if (msg.sourcesRendered) return; // built once: never collapses an open drawer
    msg.sourcesRendered = true;
    if (!sources.length) return;

    sources.forEach((s, idx) => {
      if (!s || typeof s !== 'object') return;
      const head = h('div', { className: 'source-head' }, h('strong', { text: '[' + (idx + 1) + ']' }));
      if (s.title) head.append(h('span', { className: 'source-title', text: String(s.title) }));
      const href = U.safeHttpUrl(s.url);
      if (href) {
        let host = href;
        try {
          host = new URL(href).hostname;
        } catch (e) {
          /* keep href */
        }
        head.append(h('a', {
          className: 'source-link',
          text: host,
          attrs: { href, target: '_blank', rel: 'noopener noreferrer' },
        }));
      }
      const metaParts = [];
      const rrf = U.toNumber(s.rrf_score);
      if (rrf !== null) metaParts.push('RRF ' + rrf.toFixed(4));
      const rerank = U.toNumber(s.rerank_score);
      if (rerank !== null) metaParts.push('Rerank ' + rerank.toFixed(3));
      if (s.doc_key) metaParts.push(String(s.doc_key));
      drawer.append(h('div', { className: 'source-item' },
        head,
        h('p', { className: 'source-text', text: U.truncateText(s.text || '', 220) }),
        metaParts.length ? h('div', { className: 'source-meta', text: metaParts.join(' · ') }) : null));
    });
    btn.textContent = '📑 References (' + sources.length + ') ▾';
    btn.hidden = false;
  }

  function toggleSources(msg) {
    const drawer = msg.els.drawer;
    const open = drawer.hidden;
    drawer.hidden = !open;
    msg.els.sourcesBtn.setAttribute('aria-expanded', String(open));
    msg.els.sourcesBtn.textContent = msg.els.sourcesBtn.textContent.replace(open ? '▾' : '▴', open ? '▴' : '▾');
  }

  function setEvalChip(msg) {
    const chip = msg.els.evalChip;
    const evaluation = msg.evaluation;
    chip.classList.remove('score-medium', 'score-low');
    if (!msg.evalRequested && !evaluation) {
      chip.hidden = true;
      return;
    }
    if (!msg.evaluationReceived) {
      chip.textContent = '🎯 Evaluating…';
      chip.title = 'DeepEval is scoring this answer';
      chip.hidden = false;
      return;
    }
    const overall = evaluation && evaluation.scores ? U.toNumber(evaluation.scores.overall_score) : null;
    if (overall === null) {
      chip.textContent = '🎯 Eval failed';
      chip.classList.add('score-low');
      const failed = evaluation && Array.isArray(evaluation.failed_metrics) ? evaluation.failed_metrics.join(', ') : '';
      chip.title = failed ? 'Failed metrics: ' + failed : 'Evaluation failed';
    } else {
      chip.textContent = '🎯 DeepEval: ' + Math.round(overall * 100) + '%';
      if (overall < 0.5) chip.classList.add('score-low');
      else if (overall < 0.75) chip.classList.add('score-medium');
      const scores = evaluation.scores || {};
      chip.title = String(scores.reason || scores.explanation || 'DeepEval score');
    }
    chip.hidden = false;
  }

  // ---------- Audio per message ----------
  /** Chunks already released in order (seq < next expected); later ones are still waiting for a gap. */
  function audioUrls(msg) {
    const limit = msg.seqBuffer.nextSeq;
    return Array.from(msg.audio.keys())
      .filter((k) => k < limit)
      .sort((a, b) => a - b)
      .map((k) => msg.audio.get(k));
  }

  function msgAddAudioChunk(msg, seq, b64) {
    if (!msg || !b64) return;
    let url;
    try {
      url = URL.createObjectURL(new Blob([U.base64ToBytes(b64)], { type: 'audio/mpeg' }));
    } catch (e) {
      console.warn('Invalid TTS audio chunk:', e);
      return;
    }
    let s = Number(seq);
    if (seq === null || seq === undefined || !Number.isInteger(s)) s = msg.audio.size ? Math.max(...msg.audio.keys()) + 1 : 0;
    if (msg.audio.has(s)) {
      URL.revokeObjectURL(url);
      return;
    }
    msg.audio.set(s, url);
    const ready = msg.seqBuffer.add(s, url);
    if (msg.autoplay) ready.forEach((u) => Player.enqueue(u, msg.id));
    msg.els.audioBtn.hidden = false;
    updateAudioButton(msg);
  }

  /** End of TTS for this message (tts_done / stream end): play anything held back by a gap. */
  function msgDrainAudio(msg) {
    if (!msg) return;
    msg.ttsDone = true;
    const rest = msg.seqBuffer.drain();
    if (msg.autoplay) rest.forEach((u) => Player.enqueue(u, msg.id));
  }

  function revokeMessageAudio(msg) {
    msg.audio.forEach((url) => URL.revokeObjectURL(url));
    msg.audio.clear();
  }

  function updateAudioButton(msg) {
    const btn = msg.els.audioBtn;
    if (!btn || btn.hidden) return;
    if (msg.synthesizing) {
      btn.textContent = '⏳ Preparing audio…';
      btn.disabled = true;
      return;
    }
    btn.disabled = false;
    if (Player.isPlaying(msg.id)) {
      btn.textContent = '⏹ Stop audio';
      btn.setAttribute('aria-label', 'Stop audio');
    } else if (msg.audio.size) {
      btn.textContent = '🔊 Play answer';
      btn.setAttribute('aria-label', 'Play the spoken answer');
    } else {
      btn.textContent = '🔊 Listen';
      btn.setAttribute('aria-label', 'Listen to the answer (Sarvam TTS)');
    }
  }

  async function onAudioButton(msg) {
    if (Player.isPlaying(msg.id)) {
      msg.autoplay = false;
      Player.stop();
      return;
    }
    if (msg.audio.size) {
      // Replay everything received so far, in order; chunks still streaming are appended.
      msg.autoplay = !msg.ttsDone;
      Player.playAll(audioUrls(msg), msg.id);
      return;
    }
    if (!msg.answer) return;
    msg.synthesizing = true;
    updateAudioButton(msg);
    try {
      const b64 = await synthesize(msg.answer, msg.answerLanguage || msg.language || STATE.lang);
      if (!messages.has(msg.id)) return; // session was reset meanwhile
      msg.synthesizing = false;
      msgAddAudioChunk(msg, 0, b64);
      Player.playAll(audioUrls(msg), msg.id);
    } catch (err) {
      msg.synthesizing = false;
      updateAudioButton(msg);
      showToast('Voice playback failed: ' + err.message, 'error');
    }
  }

  /** POST /api/v1/voice/tts -> base64 MP3 */
  async function synthesize(text, language) {
    const lang = ['gu', 'hi', 'en'].includes(language) ? language : STATE.lang;
    const res = await apiFetch('/api/v1/voice/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language: lang }),
      signal: timeoutSignal(60000),
    });
    if (!res.ok) throw await httpError(res);
    const data = await res.json();
    if (!data || !data.audio_base64) throw new Error('The server returned no audio.');
    return data.audio_base64;
  }

  // ---------- Copy & feedback ----------
  function legacyCopy(text) {
    const ta = h('textarea', { attrs: { readonly: '', 'aria-hidden': 'true' } });
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    ta.style.pointerEvents = 'none';
    document.body.append(ta);
    ta.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } finally {
      ta.remove();
    }
    if (!ok) throw new Error('copy command failed');
  }

  async function copyText(text) {
    if (!text) return;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text); // newlines preserved
      } else {
        legacyCopy(text);
      }
      showToast('Copied to clipboard');
    } catch (err) {
      try {
        legacyCopy(text);
        showToast('Copied to clipboard');
      } catch (e) {
        showToast('Could not copy the text.', 'error');
      }
    }
  }

  async function sendFeedback(msg, score, button) {
    if (!msg.traceId) return;
    const { likeBtn, dislikeBtn } = msg.els;
    likeBtn.disabled = true;
    dislikeBtn.disabled = true;
    button.classList.add('selected');
    button.setAttribute('aria-pressed', 'true');
    try {
      const res = await apiFetch('/api/v1/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trace_id: msg.traceId, score }),
        signal: timeoutSignal(15000),
      });
      if (!res.ok) throw await httpError(res);
      showToast('Thanks for the feedback!');
    } catch (err) {
      likeBtn.disabled = false;
      dislikeBtn.disabled = false;
      button.classList.remove('selected');
      button.setAttribute('aria-pressed', 'false');
      showToast('Feedback was not sent: ' + err.message, 'error');
    }
  }

  // ============================================================
  // Shared answer handling (text stream, upload, live voice)
  // ============================================================
  function handleAnswerMeta(msg, data, context) {
    const ctx = context || {};
    msgSetAnswer(msg, data);

    // Recent-history label uses the server-resolved language
    if (ctx.recentItem && U.isLangCode(data.language) && ctx.recentItem.lang !== data.language) {
      ctx.recentItem.lang = data.language;
      persistRecent();
      renderRecent();
    }

    // Conversation memory for follow-up questions
    if (ctx.epoch === STATE.sessionEpoch && ctx.query && msg.answer) {
      STATE.session.push({ role: 'user', content: ctx.query }, { role: 'assistant', content: msg.answer });
      if (STATE.session.length > 40) STATE.session = STATE.session.slice(-40);
    }

    // Keep timings that arrived before the answer (e.g. tts_done) unless the server sends them.
    const previous = msg.latency || {};
    msg.latency = Object.assign({}, data.latency || {});
    ['tts_ms', 'eval_ms', 'stt_ms'].forEach((key) => {
      if (U.toNumber(msg.latency[key]) === null && U.toNumber(previous[key]) !== null) msg.latency[key] = previous[key];
    });
    if (ctx.sttMs !== undefined && ctx.sttMs !== null && U.toNumber(msg.latency.stt_ms) === null) {
      msg.latency.stt_ms = ctx.sttMs;
    }
    if (STATE.lastMessageId === msg.id) updateLatencySidebar(msg.latency);

    // Fresh scorecard for every answer
    if (msg.evalRequested) {
      setScorecardPlaceholder('pending');
    } else {
      setScorecardPlaceholder('off');
    }
    setEvalChip(msg);
  }

  function handleEvaluation(msg, evaluation, evalMs) {
    if (!msg) return;
    msg.evaluation = evaluation || null;
    msg.evaluationReceived = true;
    setEvalChip(msg);
    const ms = U.toNumber(evalMs);
    if (ms !== null) {
      msg.latency = Object.assign({}, msg.latency || {}, { eval_ms: ms });
    }
    if (STATE.lastMessageId === msg.id) {
      if (msg.latency) updateLatencySidebar(msg.latency);
      STATE.lastEvalLanguage = msg.answerLanguage || msg.language;
      renderScorecard(msg.evaluation);
    }
  }

  function patchLatency(msg, field, value) {
    const n = U.toNumber(value);
    if (!msg || n === null) return;
    msg.latency = Object.assign({}, msg.latency || {}, { [field]: n });
    if (STATE.lastMessageId === msg.id && msg.finished) updateLatencySidebar(msg.latency);
  }

  // ============================================================
  // Text query (SSE stream)
  // ============================================================
  function readToggle(id) {
    const el = $(id);
    return !!(el && el.checked);
  }

  function takeSampleQueryId(query) {
    const pending = STATE.pendingSample;
    STATE.pendingSample = null;
    if (pending && pending.text.trim() === query.trim() && Number.isInteger(pending.queryId)) return pending.queryId;
    return null;
  }

  function submitCurrentQuery() {
    const input = $('query-input');
    if (!input) return;
    const query = input.value.trim();
    if (!query) return;
    if (STATE.busy) {
      showToast('Please wait for the current answer (or press Stop).');
      return;
    }
    const queryId = takeSampleQueryId(query);
    input.value = '';
    handleInputResize();
    submitTextQuery(query, queryId);
  }

  async function submitTextQuery(rawQuery, queryId) {
    const query = String(rawQuery || '').trim().slice(0, MAX_QUERY_CHARS);
    if (!query || STATE.busy) return;

    const lang = STATE.lang;
    const copy = langCopy(lang);
    const evaluate = readToggle('eval-toggle');
    const voiceReply = readToggle('tts-toggle');
    const autoDetect = readToggle('autodetect-toggle');
    const history = U.buildHistory(STATE.session, HISTORY_MESSAGES);
    const epoch = STATE.sessionEpoch;
    const time = U.formatClock(new Date());

    stopAllAudio();
    appendUserMessage(query, time, false);
    const recentItem = saveRecent(query, 'text', lang);
    const msg = createBotMessage({ time, thinking: copy.thinkingText, query, language: lang });
    msg.evalRequested = evaluate;
    msg.autoplay = voiceReply;
    scrollToBottom(true);

    const req = beginRequest('text', msg, { idleMs: TEXT_IDLE_TIMEOUT_MS, totalMs: TEXT_TOTAL_TIMEOUT_MS });
    const body = {
      query,
      language: lang,
      top_k: TOP_K,
      auto_detect_language: autoDetect,
      evaluate,
      voice_reply: voiceReply,
      history,
    };
    if (Number.isInteger(queryId)) body.query_id = queryId;

    let gotMeta = false;
    let gotDone = false;
    let streamError = null;

    const onEvent = (evt) => {
      req.touch();
      if (req.cancelled || epoch !== STATE.sessionEpoch) return;
      let data = {};
      if (evt.data) {
        try {
          data = JSON.parse(evt.data);
        } catch (e) {
          console.warn('Ignoring malformed SSE data:', evt.data);
          return;
        }
      }
      // `error` is handled outside any parse try/catch so it can never be swallowed.
      switch (evt.event) {
        case 'token': {
          const token = typeof data.token === 'string' ? data.token : (typeof data.content === 'string' ? data.content : '');
          if (token) msgAppendToken(msg, token);
          break;
        }
        case 'tts':
          msgAddAudioChunk(msg, data.seq, data.audio_base64);
          break;
        case 'meta':
          gotMeta = true;
          req.setIdle(AFTER_ANSWER_IDLE_MS);
          handleAnswerMeta(msg, data, { query, recentItem, epoch });
          break;
        case 'tts_done':
          msgDrainAudio(msg);
          patchLatency(msg, 'tts_ms', data.tts_ms);
          break;
        case 'evaluation':
          handleEvaluation(msg, data.evaluation, data.eval_ms);
          break;
        case 'error':
          streamError = U.formatErrorDetail(data.error || data.message || data.detail) || 'Streaming error';
          if (gotMeta) {
            msgAddWarning(msg, streamError);
          } else {
            msgSetError(msg, copy.errPrefix + ': ' + streamError);
          }
          break;
        case 'done':
          gotDone = true;
          break;
        default:
          break;
      }
    };

    try {
      const res = await apiFetch('/api/v1/query/text/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify(body),
        signal: req.signal,
      });
      if (!res.ok) throw await httpError(res);
      if (!res.body || typeof res.body.getReader !== 'function') throw new Error('This browser cannot read streaming responses.');

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      const parser = U.createSSEParser(onEvent);
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        parser.push(decoder.decode(value, { stream: true }));
      }
      parser.push(decoder.decode());
      parser.flush();

      if (epoch !== STATE.sessionEpoch) return;
      if (!gotMeta && !streamError) {
        msgSetError(msg, copy.errPrefix + ': the connection closed before the answer was complete.');
      } else if (gotMeta && !gotDone && !streamError) {
        msgAddWarning(msg, 'The connection closed early; audio or evaluation may be incomplete.');
      }
      if (gotMeta && evaluate && !msg.evaluationReceived) {
        handleEvaluation(msg, null, null);
      }
      msgDrainAudio(msg);
    } catch (err) {
      if (epoch !== STATE.sessionEpoch) return;
      if (req.cancelled) {
        msgSetCancelled(msg, req.reason, gotMeta);
        if (gotMeta && evaluate && !msg.evaluationReceived && STATE.lastMessageId === msg.id) setScorecardPlaceholder('cancelled');
      } else {
        console.error('Query failed:', err);
        if (gotMeta) msgAddWarning(msg, err.message);
        else msgSetError(msg, copy.errPrefix + ': ' + err.message);
      }
    } finally {
      endRequest(req);
    }
  }

  // ============================================================
  // Audio file upload
  // ============================================================
  function fileExtension(name) {
    const idx = String(name || '').lastIndexOf('.');
    return idx === -1 ? '' : name.slice(idx + 1).toLowerCase();
  }

  async function handleAudioFile(file) {
    if (!file) return;
    if (STATE.busy) {
      showToast('Please wait for the current answer (or press Stop).');
      return;
    }
    const ext = fileExtension(file.name);
    if (!AUDIO_EXTENSIONS.includes(ext)) {
      showToast('Unsupported file type' + (ext ? ' ".' + ext + '"' : '') + '. Use: ' + AUDIO_EXTENSIONS.join(', ') + '.', 'error');
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      showToast('File is too large (' + (file.size / 1048576).toFixed(1) + ' MB). Maximum is ' + MAX_UPLOAD_MB + ' MB.', 'error');
      return;
    }

    const lang = STATE.lang;
    const copy = langCopy(lang);
    const evaluate = readToggle('eval-toggle');
    const autoDetect = readToggle('autodetect-toggle');
    const history = U.buildHistory(STATE.session, HISTORY_MESSAGES);
    const epoch = STATE.sessionEpoch;
    const time = U.formatClock(new Date());

    stopAllAudio();
    // User bubble first, bot bubble after it
    const user = appendUserMessage('🎧 ' + file.name + ' — Transcribing…', time, true);
    const msg = createBotMessage({ time, thinking: copy.thinkingText, language: lang });
    msg.evalRequested = evaluate;
    msg.autoplay = true; // voice in -> voice out
    showTranscript('Transcribing audio… ⏳', 'info');
    scrollToBottom(true);

    const req = beginRequest('upload', msg, { totalMs: UPLOAD_TIMEOUT_MS });
    const form = new FormData();
    form.append('file', file);
    form.append('language', lang);
    form.append('top_k', String(TOP_K));
    form.append('auto_detect_language', autoDetect ? 'true' : 'false');
    form.append('evaluate', evaluate ? 'true' : 'false');
    form.append('voice_reply', 'true');
    if (history.length) form.append('history_json', JSON.stringify(history));

    try {
      const res = await apiFetch('/api/v1/query/voice', { method: 'POST', body: form, signal: req.signal });
      if (!res.ok) throw await httpError(res, { maxMb: MAX_UPLOAD_MB });
      const data = await res.json();
      if (epoch !== STATE.sessionEpoch) return;

      const transcript = String(data.transcription || data.query || '').trim();
      if (!transcript) {
        if (user) user.textEl.textContent = '🎧 ' + file.name;
        showTranscript('No speech was recognised in this file.', 'error');
        msgSetError(msg, 'No speech could be recognised in the uploaded audio.');
        return;
      }
      if (user) user.textEl.textContent = transcript;
      showTranscript(transcript, 'final');
      msg.query = transcript;
      const recentItem = saveRecent(transcript, 'voice', lang);
      if (data.audio_base64) msgAddAudioChunk(msg, 0, data.audio_base64);
      handleAnswerMeta(msg, data, { query: transcript, recentItem, epoch });
      msgDrainAudio(msg);
      if (evaluate) handleEvaluation(msg, data.evaluation || null, data.latency && data.latency.eval_ms);
    } catch (err) {
      if (epoch !== STATE.sessionEpoch) return;
      if (user && user.textEl.textContent.endsWith('Transcribing…')) user.textEl.textContent = '🎧 ' + file.name;
      if (req.cancelled) {
        msgSetCancelled(msg, req.reason, false);
        showTranscript(cancelReasonText(req.reason), 'info');
      } else {
        console.error('Audio upload failed:', err);
        showTranscript('❌ ' + err.message, 'error');
        msgSetError(msg, copy.errPrefix + ': ' + err.message);
      }
    } finally {
      endRequest(req);
    }
  }

  // ============================================================
  // Live voice (WebSocket + AudioWorklet)
  // ============================================================
  function showTranscript(text, kind) {
    const card = $('voice-transcript-card');
    if (!card) return;
    card.hidden = false;
    card.dataset.kind = kind || 'info';
    setText('transcript-timestamp', U.formatClock(new Date()));
    setText('transcript-text', text);
  }

  function setVoiceState(state) {
    STATE.voice = state;
    const mic = $('main-mic-btn');
    const listening = $('listening-indicator');
    const prompt = $('speak-prompt-pill');
    const copy = langCopy(STATE.lang);
    if (mic) {
      mic.classList.toggle('recording', state === 'recording');
      mic.classList.toggle('connecting', state === 'connecting');
      mic.classList.toggle('processing', state === 'processing');
      mic.setAttribute('aria-pressed', String(state === 'recording' || state === 'connecting'));
      const labels = {
        idle: ['Start voice input', 'Tap to speak'],
        connecting: ['Connecting to the microphone…', 'Connecting…'],
        recording: ['Stop recording', 'Tap to stop'],
        processing: ['Processing your question…', 'Processing…'],
      };
      mic.setAttribute('aria-label', labels[state][0]);
      mic.title = labels[state][1];
    }
    if (listening) listening.hidden = state !== 'recording';
    if (prompt) prompt.hidden = state === 'recording';
    const promptText = {
      idle: copy.speakPrompt,
      connecting: 'Connecting…',
      recording: '',
      processing: 'Processing your question…',
    };
    setText('speak-prompt-text', promptText[state]);
    refreshControls();
    if (state === 'recording') drawLiveWaveform();
    else if (state !== 'recording') drawIdleWaveform();
  }

  function toggleVoiceRecording() {
    if (STATE.voice === 'recording') {
      stopVoiceRecording();
    } else if (STATE.voice === 'idle') {
      startVoiceRecording();
    }
    // 'connecting' / 'processing': the button is disabled; Stop cancels.
  }

  async function createCapture(stream) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) throw new Error('Web Audio is not supported in this browser.');
    const ctx = new AC();
    try {
      if (ctx.state === 'suspended') await ctx.resume();
    } catch (e) {
      /* resume() can reject without a gesture; capture still starts on most browsers */
    }
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 128;
    source.connect(analyser);

    let onPcm = null;
    let released = false;
    let workletNode = null;
    let processor = null;

    const handleFloat = (samples) => {
      if (released || !onPcm) return;
      const pcm = U.downsampleTo16kPCM(samples, ctx.sampleRate);
      if (pcm.byteLength) onPcm(pcm);
    };

    if (ctx.audioWorklet && typeof window.AudioWorkletNode === 'function') {
      try {
        await ctx.audioWorklet.addModule('/static/pcm-worklet.js?v=' + ASSET_VERSION);
        workletNode = new AudioWorkletNode(ctx, 'pcm-capture-processor', {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 1,
          processorOptions: { chunkSize: 4096 },
        });
        workletNode.port.onmessage = (e) => handleFloat(e.data);
      } catch (err) {
        console.warn('AudioWorklet unavailable, falling back to ScriptProcessorNode:', err);
        workletNode = null;
      }
    }

    return {
      ctx,
      analyser,
      levels: new Uint8Array(analyser.frequencyBinCount), // reused every frame
      start(callback) {
        onPcm = callback;
        if (workletNode) {
          source.connect(workletNode);
          workletNode.connect(ctx.destination); // outputs silence; keeps the graph pulling
        } else {
          processor = ctx.createScriptProcessor(4096, 1, 1);
          processor.onaudioprocess = (e) => handleFloat(e.inputBuffer.getChannelData(0));
          source.connect(processor);
          processor.connect(ctx.destination);
        }
      },
      release() {
        if (released) return;
        released = true;
        onPcm = null;
        if (workletNode) {
          try {
            workletNode.port.postMessage('stop');
          } catch (e) {
            /* ignore */
          }
          workletNode.port.onmessage = null;
        }
        if (processor) processor.onaudioprocess = null;
        [workletNode, processor, analyser, source].forEach((node) => {
          try {
            if (node) node.disconnect();
          } catch (e) {
            /* ignore */
          }
        });
        stream.getTracks().forEach((track) => track.stop());
        if (ctx.state !== 'closed') ctx.close().catch(() => {});
      },
    };
  }

  function releaseCapture() {
    if (voice.capture) {
      voice.capture.release();
      voice.capture = null;
    }
    clearTimeout(voice.maxTimer);
    voice.maxTimer = null;
  }

  async function startVoiceRecording() {
    if (STATE.voice !== 'idle') return;
    if (STATE.busy) {
      showToast('Please wait for the current answer (or press Stop).');
      return;
    }
    const problem = U.micSupportProblem({
      isSecureContext: window.isSecureContext,
      hasGetUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
      host: location.host,
    });
    if (problem) {
      showTranscript('❌ ' + problem, 'error');
      return;
    }

    const token = ++voiceToken;
    const lang = STATE.lang;
    const copy = langCopy(lang);
    const evaluate = readToggle('eval-toggle');
    const autoDetect = readToggle('autodetect-toggle');
    const history = U.buildHistory(STATE.session, HISTORY_MESSAGES);
    const epoch = STATE.sessionEpoch;

    setBusy('voice');
    setVoiceState('connecting'); // disables the mic button immediately
    stopAllAudio();              // never record over our own TTS
    releaseCapture();            // never leak a previous stream
    showTranscript('🎙️ ' + copy.speakPrompt + '…', 'info');

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (err) {
      if (token !== voiceToken) return;
      console.error('Microphone capture error:', err);
      showTranscript('❌ ' + U.describeMicError(err), 'error');
      finishVoiceIdle();
      return;
    }
    if (token !== voiceToken || STATE.voice !== 'connecting') {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    let capture;
    try {
      capture = await createCapture(stream);
    } catch (err) {
      stream.getTracks().forEach((t) => t.stop());
      if (token !== voiceToken) return;
      showTranscript('❌ Could not start audio processing: ' + err.message, 'error');
      finishVoiceIdle();
      return;
    }
    if (token !== voiceToken || STATE.voice !== 'connecting') {
      capture.release();
      return;
    }
    voice.capture = capture;

    const params = { lang, evaluate: String(evaluate), auto_detect: String(autoDetect) };
    if (STATE.apiKey) params.api_key = STATE.apiKey;
    let ws;
    try {
      ws = new WebSocket(U.buildWsUrl(location, '/api/v1/voice/live', params));
    } catch (err) {
      releaseCapture();
      showTranscript('❌ Could not open the voice connection: ' + err.message, 'error');
      finishVoiceIdle();
      return;
    }
    ws.binaryType = 'arraybuffer';
    liveVoiceWs = ws;

    const ctx = {
      ws,
      token,
      epoch,
      lang,
      copy,
      evaluate,
      history,
      time: U.formatClock(new Date()),
      opened: false,
      stopSent: false,
      gotFinal: false,
      finalText: '',
      sttMs: null,
      gotAnswer: false,
      gotDone: false,
      error: null,
      msg: null,
      user: null,
      recentItem: null,
    };
    voice.ctx = ctx;

    ws.onopen = () => {
      if (ws !== liveVoiceWs) return;
      ctx.opened = true;
      if (history.length) {
        try {
          ws.send(JSON.stringify({ type: 'history', history }));
        } catch (e) {
          /* ignore */
        }
      }
      if (!voice.capture) return;
      voice.capture.start((pcm) => {
        if (ws.readyState === WebSocket.OPEN && !ctx.stopSent) ws.send(pcm);
      });
      setVoiceState('recording');
      showTranscript('🎙️ Listening… ' + copy.speakPrompt, 'info');
      voice.maxTimer = setTimeout(() => {
        if (voice.ctx === ctx && STATE.voice === 'recording') stopVoiceRecording();
      }, VOICE_MAX_RECORD_MS);
    };

    ws.onmessage = (event) => {
      if (ws !== liveVoiceWs) return;
      if (STATE.voice === 'processing') armVoiceIdleTimer(ctx);
      handleVoiceMessage(ctx, event);
    };

    ws.onerror = () => {
      if (ws !== liveVoiceWs) return;
      if (!ctx.error) ctx.error = ctx.opened ? 'The voice connection failed.' : 'Could not connect to the voice server.';
    };

    ws.onclose = (event) => {
      if (ws !== liveVoiceWs) return; // an old socket must never affect a new recording
      liveVoiceWs = null;
      onVoiceClosed(ctx, event);
    };
  }

  /** User tapped the mic again, the first `final` arrived, or the time limit hit. */
  function stopVoiceRecording() {
    const ctx = voice.ctx;
    if (!ctx || STATE.voice !== 'recording') return;
    if (!ctx.stopSent && ctx.ws.readyState === WebSocket.OPEN) {
      ctx.stopSent = true;
      try {
        ctx.ws.send(JSON.stringify({ type: 'stop' }));
      } catch (e) {
        /* ignore */
      }
    }
    releaseCapture();
    setVoiceState('processing');
    if (!ctx.gotFinal) showTranscript('⏳ Processing…', 'info');
    armVoiceIdleTimer(ctx);
  }

  /** Give up if the server stays silent for VOICE_ANSWER_IDLE_MS while we wait for the answer. */
  function armVoiceIdleTimer(ctx) {
    clearTimeout(voice.answerTimer);
    voice.answerTimer = setTimeout(() => {
      if (voice.ctx === ctx) cancelVoice('timeout');
    }, VOICE_ANSWER_IDLE_MS);
  }

  function ensureVoiceBubble(ctx) {
    if (!ctx.msg) {
      ctx.msg = createBotMessage({ time: ctx.time, thinking: ctx.copy.thinkingText, language: ctx.lang });
      ctx.msg.evalRequested = ctx.evaluate;
      ctx.msg.autoplay = true;
      ctx.msg.query = ctx.finalText;
    }
    return ctx.msg;
  }

  function handleVoiceMessage(ctx, event) {
    if (typeof event.data !== 'string') return;
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      console.warn('Ignoring malformed voice message:', event.data);
      return;
    }
    if (!data || typeof data !== 'object' || ctx.epoch !== STATE.sessionEpoch) return;

    switch (data.type) {
      case 'partial': {
        const text = typeof data.text === 'string' ? data.text.trim() : '';
        if (text) showTranscript(text, 'partial');
        break;
      }
      case 'final': {
        const text = typeof data.text === 'string' ? data.text.trim() : '';
        if (!text) break;
        if (U.toNumber(data.stt_ms) !== null) ctx.sttMs = U.toNumber(data.stt_ms);
        if (!ctx.gotFinal) {
          ctx.gotFinal = true;
          ctx.finalText = text;
          ctx.user = appendUserMessage(text, ctx.time, true);
          ctx.recentItem = saveRecent(text, 'voice', ctx.lang);
          ensureVoiceBubble(ctx);
          scrollToBottom(true);
          // Contract: the client sends {"type":"stop"} on the first final transcript.
          stopVoiceRecording();
        } else {
          ctx.finalText += ' ' + text;
          if (ctx.user) ctx.user.textEl.textContent = ctx.finalText;
          if (ctx.recentItem) {
            ctx.recentItem.query = ctx.finalText.slice(0, 1000);
            persistRecent();
            renderRecent();
          }
        }
        if (ctx.msg) ctx.msg.query = ctx.finalText;
        showTranscript('✅ “' + ctx.finalText + '”', 'final');
        break;
      }
      case 'token': {
        const token = typeof data.content === 'string' ? data.content : (typeof data.token === 'string' ? data.token : '');
        if (token) msgAppendToken(ensureVoiceBubble(ctx), token);
        break;
      }
      case 'tts':
        msgAddAudioChunk(ensureVoiceBubble(ctx), data.seq, data.audio_base64);
        break;
      case 'answer': {
        const msg = ensureVoiceBubble(ctx);
        ctx.gotAnswer = true;
        handleAnswerMeta(msg, data, {
          query: ctx.finalText || data.query || '',
          recentItem: ctx.recentItem,
          epoch: ctx.epoch,
          sttMs: ctx.sttMs,
        });
        break;
      }
      case 'tts_done':
        if (ctx.msg) {
          msgDrainAudio(ctx.msg);
          patchLatency(ctx.msg, 'tts_ms', data.tts_ms);
        }
        break;
      case 'evaluation':
        if (ctx.msg) handleEvaluation(ctx.msg, data.evaluation || null, data.eval_ms);
        break;
      case 'error': {
        ctx.error = U.formatErrorDetail(data.message || data.error || data.detail) || 'Voice error occurred.';
        showTranscript('❌ ' + ctx.error, 'error');
        if (ctx.msg) {
          if (ctx.gotAnswer) msgAddWarning(ctx.msg, ctx.error);
          else msgSetError(ctx.msg, ctx.copy.errPrefix + ': ' + ctx.error);
        }
        break;
      }
      case 'done':
        ctx.gotDone = true;
        try {
          ctx.ws.close(1000, 'done');
        } catch (e) {
          /* ignore */
        }
        break;
      default:
        break;
    }
  }

  function onVoiceClosed(ctx, event) {
    releaseCapture();
    if (event && event.code === 4401) {
      ctx.error = 'API key missing or invalid.';
      handleUnauthorized('The voice server rejected the API key. Please enter a valid key.');
    } else if (!ctx.opened && !ctx.error) {
      ctx.error = 'Could not connect to the voice server' + (event && event.code ? ' (code ' + event.code + ')' : '') + '.';
    } else if (!ctx.gotAnswer && !ctx.error) {
      ctx.error = ctx.gotFinal
        ? 'The voice connection closed before the answer arrived.'
        : 'No speech was recognised. Please try again.';
    }
    finalizeVoice(ctx, null);
  }

  function finalizeVoice(ctx, cancelReason) {
    clearTimeout(voice.answerTimer);
    voice.answerTimer = null;
    if (voice.ctx === ctx) voice.ctx = null;

    if (ctx.epoch === STATE.sessionEpoch) {
      if (cancelReason) {
        showTranscript(cancelReasonText(cancelReason), 'info');
        if (ctx.msg) msgSetCancelled(ctx.msg, cancelReason, ctx.gotAnswer);
        if (ctx.gotAnswer && ctx.evaluate && ctx.msg && !ctx.msg.evaluationReceived && STATE.lastMessageId === ctx.msg.id) {
          setScorecardPlaceholder('cancelled');
        }
      } else if (ctx.error) {
        showTranscript('❌ ' + ctx.error, 'error');
        if (ctx.msg && !ctx.msg.finished) msgSetError(ctx.msg, ctx.copy.errPrefix + ': ' + ctx.error);
        else if (ctx.msg && ctx.gotAnswer && !ctx.msg.els.warning.textContent) msgAddWarning(ctx.msg, ctx.error);
      }
      if (ctx.msg) msgDrainAudio(ctx.msg);
      if (!cancelReason && ctx.gotAnswer && ctx.evaluate && ctx.msg && !ctx.msg.evaluationReceived) {
        handleEvaluation(ctx.msg, null, null);
      }
    }
    finishVoiceIdle();
  }

  function finishVoiceIdle() {
    releaseCapture();
    setVoiceState('idle');
    clearBusy('voice');
  }

  /** Cancel recording/processing; ignores any later events from that socket. */
  function cancelVoice(reason) {
    voiceToken += 1; // invalidates a pending getUserMedia()
    const ctx = voice.ctx;
    releaseCapture();
    if (ctx) {
      const ws = ctx.ws;
      if (liveVoiceWs === ws) liveVoiceWs = null;
      try {
        if (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN) ws.close(1000, 'cancelled');
      } catch (e) {
        /* ignore */
      }
      finalizeVoice(ctx, reason || 'user');
    } else {
      finishVoiceIdle();
    }
  }

  // ============================================================
  // Waveform canvas (DPR-aware, reused buffers/gradients, theme colours)
  // ============================================================
  const Wave = { canvas: null, ctx: null, w: 0, h: 0, idleGrad: null, liveGrad: null, raf: 0 };

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function setupWaveCanvas() {
    const canvas = $('waveform-canvas');
    if (!canvas) return false;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return false; // hidden (chat mode)
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const w = Math.round(rect.width);
    const h2 = Math.round(rect.height);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h2 * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h2 * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    Wave.canvas = canvas;
    Wave.ctx = ctx;
    Wave.w = w;
    Wave.h = h2;

    const c1 = cssVar('--wave-1', '#06b6d4');
    const c2 = cssVar('--wave-2', '#6366f1');
    const c3 = cssVar('--wave-3', '#ec4899');
    const idle = ctx.createLinearGradient(0, 0, w, 0);
    idle.addColorStop(0, c2);
    idle.addColorStop(0.5, c1);
    idle.addColorStop(1, c2);
    const live = ctx.createLinearGradient(0, 0, 0, h2);
    live.addColorStop(0, c1);
    live.addColorStop(0.5, c2);
    live.addColorStop(1, c3);
    Wave.idleGrad = idle;
    Wave.liveGrad = live;
    return true;
  }

  function drawIdleWaveform() {
    if (Wave.raf) {
      cancelAnimationFrame(Wave.raf);
      Wave.raf = 0;
    }
    if (!setupWaveCanvas()) return;
    const { ctx, w, h: hh } = Wave;
    ctx.clearRect(0, 0, w, hh);
    ctx.globalAlpha = 0.55;
    ctx.strokeStyle = Wave.idleGrad;
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.beginPath();
    const bars = 45;
    const bw = w / bars;
    for (let i = 0; i < bars; i += 1) {
      const x = i * bw + bw / 2;
      const barH = 18 * (Math.sin((i / bars) * Math.PI) * 0.7 + 0.15);
      ctx.moveTo(x, hh / 2 - barH / 2);
      ctx.lineTo(x, hh / 2 + barH / 2);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawLiveWaveform() {
    if (Wave.raf) cancelAnimationFrame(Wave.raf);
    Wave.raf = 0;
    if (!setupWaveCanvas()) return;
    const frame = () => {
      const capture = voice.capture;
      if (STATE.voice !== 'recording' || !capture) {
        Wave.raf = 0;
        return;
      }
      const { ctx, w, h: hh } = Wave;
      const levels = capture.levels;
      capture.analyser.getByteFrequencyData(levels);
      ctx.clearRect(0, 0, w, hh);
      ctx.strokeStyle = Wave.liveGrad;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.beginPath();
      const bars = 45;
      const bw = w / bars;
      for (let i = 0; i < bars; i += 1) {
        const value = levels[Math.floor((i / bars) * levels.length)] || 0;
        const barH = Math.max(4, (value / 255) * hh * 0.9);
        const x = i * bw + bw / 2;
        ctx.moveTo(x, hh / 2 - barH / 2);
        ctx.lineTo(x, hh / 2 + barH / 2);
      }
      ctx.stroke();
      Wave.raf = requestAnimationFrame(frame);
    };
    Wave.raf = requestAnimationFrame(frame);
  }

  // ============================================================
  // Telemetry sidebar
  // ============================================================
  function setOptionalRow(rowId, valueId, value) {
    const row = $(rowId);
    const n = U.toNumber(value);
    if (row) row.hidden = n === null;
    setText(valueId, U.formatMs(n));
  }

  function updateLatencySidebar(latency) {
    if (!latency) return;
    const total = U.toNumber(latency.total_ms);
    setText('latency-total-val', total === null ? '--' : (total / 1000).toFixed(2) + ' s');

    const fill = $('gauge-needle');
    if (fill) fill.style.width = (total === null ? 0 : Math.min(100, Math.max(0, (total / 4000) * 100))) + '%';

    const rating = U.latencyRating(total);
    const pill = $('latency-rating-tag');
    if (pill) {
      pill.classList.remove('good', 'medium', 'slow');
      if (rating.cls) pill.classList.add(rating.cls);
    }
    setText('latency-rating-text', rating.label);

    const ret = latency.retrieval || {};
    setText('time-encoding', U.formatMs(ret.query_encoding_ms));
    setText('time-dense', U.formatMs(ret.dense_search_ms));
    setText('time-sparse', U.formatMs(ret.sparse_search_ms));
    setText('time-metadata', U.formatMs(ret.metadata_ms));
    setText('time-ttft', U.formatMs(latency.ttft_ms));
    setText('time-llm', U.formatMs(latency.llm_ms));
    setOptionalRow('chip-stt', 'time-stt', latency.stt_ms);
    setOptionalRow('chip-condense', 'time-condense', ret.condense_ms);
    setOptionalRow('chip-rerank', 'time-rerank', ret.rerank_ms);
    setOptionalRow('chip-tts', 'time-tts', latency.tts_ms);
    setOptionalRow('chip-eval', 'time-eval', latency.eval_ms);
  }

  function resetLatencySidebar() {
    ['latency-total-val', 'time-encoding', 'time-dense', 'time-sparse', 'time-metadata', 'time-ttft', 'time-llm'].forEach((id) => setText(id, '--'));
    ['chip-stt', 'chip-condense', 'chip-rerank', 'chip-tts', 'chip-eval'].forEach((id) => {
      const el = $(id);
      if (el) el.hidden = true;
    });
    const fill = $('gauge-needle');
    if (fill) fill.style.width = '0%';
    const pill = $('latency-rating-tag');
    if (pill) pill.classList.remove('good', 'medium', 'slow');
    setText('latency-rating-text', 'Idle');
  }

  /** kind: 'off' | 'pending' | 'failed' | 'cancelled' */
  function setScorecardPlaceholder(kind, details) {
    const empty = $('eval-empty-state');
    const content = $('eval-score-content');
    const text = $('eval-empty-text');
    if (!empty || !text) return;
    empty.hidden = false;
    empty.dataset.state = kind;
    if (content) content.hidden = true;
    const golden = $('badge-golden-status');
    const open = $('badge-open-status');
    if (golden) golden.hidden = true;
    if (open) open.hidden = true;

    text.textContent = '';
    const d = details || {};
    switch (kind) {
      case 'pending':
        text.append('⏳ Evaluating the answer with DeepEval…');
        break;
      case 'failed':
        text.append(h('strong', { text: 'Evaluation failed.' }), ' The judge could not score this answer.');
        if (d.failed && d.failed.length) {
          text.append(h('br'), 'Failed metrics: ' + d.failed.join(', '));
        }
        if (d.warning) text.append(h('br'), String(d.warning));
        break;
      case 'cancelled':
        text.append('Evaluation was cancelled.');
        break;
      default:
        text.append('Evaluation is ', h('strong', { text: 'OFF' }), '. Turn on ', h('strong', { text: 'Eval' }), ' in the bottom toolbar to run DeepEval metrics.');
    }
  }

  function setMetric(valueId, barId, value) {
    const n = U.toNumber(value);
    const valEl = $(valueId);
    const bar = $(barId);
    if (valEl) {
      valEl.textContent = U.formatScore(n);
      valEl.classList.toggle('is-na', n === null);
    }
    if (bar) bar.style.width = (n === null ? 0 : Math.min(100, Math.max(0, n * 100))) + '%';
  }

  function renderScorecard(evalData) {
    if (!evalData) {
      setScorecardPlaceholder('failed');
      return;
    }
    const failed = Array.isArray(evalData.failed_metrics) ? evalData.failed_metrics.filter((m) => typeof m === 'string') : [];
    const scores = evalData.scores;
    if (!scores || typeof scores !== 'object') {
      setScorecardPlaceholder('failed', { failed, warning: evalData.warning });
      return;
    }

    $('eval-empty-state').hidden = true;
    $('eval-score-content').hidden = false;
    $('badge-golden-status').hidden = !evalData.is_golden;
    $('badge-open-status').hidden = !!evalData.is_golden;

    const overall = U.toNumber(scores.overall_score);
    setText('eval-overall-score', U.formatScore(overall));
    const verdict = U.verdictFor(overall);
    const ring = $('score-ring-bar');
    if (ring) {
      const circumference = 2 * Math.PI * 26;
      ring.style.strokeDashoffset = String(circumference - Math.min(1, Math.max(0, overall || 0)) * circumference);
      ring.classList.remove('excellent', 'high', 'fair', 'low', 'failed');
      ring.classList.add(verdict.cls);
    }
    const verdictTag = $('eval-verdict-tag');
    if (verdictTag) {
      verdictTag.classList.remove('excellent', 'high', 'fair', 'low', 'failed');
      verdictTag.classList.add(verdict.cls);
    }
    setText('eval-verdict-text', verdict.label);
    setText('eval-query-type', evalData.query_type ? String(evalData.query_type) : (evalData.is_golden ? 'GOLDEN' : 'OPEN'));
    setText('eval-query-id', evalData.query_id !== null && evalData.query_id !== undefined ? '#' + evalData.query_id : 'N/A');

    setMetric('m-correctness', 'bar-correctness', scores.answer_correctness);
    setMetric('m-faithfulness', 'bar-faithfulness', scores.faithfulness);
    setMetric('m-relevance', 'bar-relevance', scores.answer_relevance);
    setMetric('m-similarity', 'bar-similarity', scores.answer_similarity);
    setMetric('m-recall', 'bar-recall', scores.context_recall);

    const failedEl = $('eval-failed-metrics');
    if (failedEl) {
      failedEl.hidden = !failed.length;
      failedEl.textContent = failed.length ? 'Failed metrics (shown as n/a): ' + failed.join(', ') : '';
    }

    // Ground truth comes only from the evaluation result (never from the static sample files)
    const gtBox = $('gt-box');
    const gtText = $('gt-text');
    const hasGt = typeof evalData.ground_truth_answer === 'string' && evalData.ground_truth_answer.trim();
    if (gtBox) gtBox.classList.toggle('open-mode', !hasGt);
    setText('gt-badge', hasGt ? 'Verified Target' : 'Open Query');
    if (gtText) {
      gtText.textContent = hasGt
        ? evalData.ground_truth_answer
        : (evalData.warning
          ? String(evalData.warning)
          : 'ℹ️ Open query with no reference answer: it was scored for faithfulness and relevance against the retrieved passages.');
      gtText.classList.toggle('gt-note', !hasGt);
    }
    const gtCopy = $('gt-copy-btn');
    const gtSpeak = $('gt-speak-btn');
    if (gtCopy) gtCopy.hidden = !hasGt;
    if (gtSpeak) gtSpeak.hidden = !hasGt;

    const reason = scores.reason || scores.explanation;
    const box = $('eval-explanation-box');
    if (box) box.hidden = !reason;
    setText('eval-explanation-text', reason ? String(reason) : '--');
  }

  // ============================================================
  // Health, languages, API key
  // ============================================================
  function setStatusPill(state, data) {
    const pill = $('system-status-pill');
    if (!pill) return;
    let label;
    if (state === 'online' || state === 'degraded') {
      const langs = Array.isArray(data && data.loaded_languages)
        ? data.loaded_languages.map((l) => String(l).toUpperCase()).join(' & ')
        : '';
      label = (state === 'online' ? 'Online' : 'Degraded') + (langs ? ' (' + langs + ')' : '');
    } else if (state === 'offline') {
      label = 'Offline';
    } else {
      label = 'Checking…';
    }
    if (pill.dataset.label === state + ':' + label) return; // unchanged: don't re-announce (aria-live)
    pill.dataset.label = state + ':' + label;
    pill.classList.remove('online', 'offline', 'checking', 'degraded');
    pill.classList.add(state);
    pill.textContent = '';
    pill.append(
      h('span', { className: 'status-dot', attrs: { 'aria-hidden': 'true' } }),
      h('span', { className: 'status-text' }, 'Engine ', h('strong', { text: label })));
    pill.title = state === 'offline' ? 'The server is not reachable. Retrying every 30 s.' : '';
  }

  function updateModelBadge(data) {
    const badge = $('nav-system-name');
    if (!badge || !data) return;
    const parts = ['BGE-M3'];
    if (data.llm_model) parts.push(String(data.llm_model));
    if (data.reranker_backend) parts.push('rerank: ' + String(data.reranker_backend));
    parts.push('Sarvam AI');
    badge.textContent = parts.join(' • ');
  }

  async function checkHealth() {
    try {
      const res = await fetch('/health', { cache: 'no-store', signal: timeoutSignal(8000) });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      const wasAuth = STATE.authRequired;
      STATE.authRequired = !!data.auth_required;
      setStatusPill(data.status && data.status !== 'healthy' ? 'degraded' : 'online', data);
      updateModelBadge(data);
      if (STATE.authRequired && !STATE.apiKey && !wasAuth) {
        openApiKeyModal('This server requires an API key. It is stored in this browser only and sent with every request.');
      }
      return true;
    } catch (err) {
      console.warn('Health check failed:', err);
      setStatusPill('offline');
      return false;
    }
  }

  function scheduleHealthChecks() {
    clearInterval(healthTimer);
    healthTimer = setInterval(() => {
      if (!document.hidden) checkHealth();
    }, HEALTH_INTERVAL_MS);
  }

  async function loadLanguages() {
    try {
      const res = await apiFetch('/api/v1/languages', { cache: 'no-store', signal: timeoutSignal(10000) });
      if (!res.ok) return;
      const data = await res.json();
      const list = Array.isArray(data && data.languages) ? data.languages : [];
      const parsed = list
        .filter((l) => l && U.isLangCode(l.code))
        .map((l) => ({
          code: l.code,
          name: typeof l.name === 'string' ? l.name : l.code.toUpperCase(),
          nativeName: typeof l.native_name === 'string' ? l.native_name : (typeof l.name === 'string' ? l.name : l.code),
          locale: typeof l.locale === 'string' ? l.locale : l.code,
          active: l.is_active !== false,
        }));
      if (!parsed.length) return;
      STATE.languages = parsed;
      renderLanguageMenu();
      const current = parsed.find((l) => l.code === STATE.lang && l.active);
      if (!current) {
        const def = parsed.find((l) => l.code === data.default_language && l.active) || parsed.find((l) => l.active);
        if (def) selectLanguage(def.code);
      }
    } catch (err) {
      console.warn('Could not load the language list:', err);
    }
  }

  function openApiKeyModal(message) {
    const modal = $('api-key-modal');
    if (!modal) return;
    setText('api-key-desc', message || 'This server requires an API key.');
    const errEl = $('api-key-error');
    if (errEl) errEl.hidden = true;
    const input = $('api-key-input');
    if (input && !Modal.isOpen(modal)) input.value = '';
    Modal.open(modal, { initialFocus: '#api-key-input' });
  }

  async function saveApiKey(event) {
    event.preventDefault();
    const input = $('api-key-input');
    const errEl = $('api-key-error');
    const saveBtn = $('api-key-save-btn');
    const key = input ? input.value.trim() : '';
    const showErr = (text) => {
      if (!errEl) return;
      errEl.textContent = text;
      errEl.hidden = false;
    };
    if (!key) {
      showErr('Please enter the API key.');
      return;
    }
    if (saveBtn) saveBtn.disabled = true;
    try {
      // Validate against a cheap protected route
      const res = await fetch('/api/v1/languages', { headers: { 'X-API-Key': key }, cache: 'no-store', signal: timeoutSignal(10000) });
      if (res.status === 401) {
        showErr('That key was rejected by the server.');
        return;
      }
      if (res.status === 429) {
        showErr(U.describeHttpError(429, '', res.headers.get('Retry-After')));
        return;
      }
    } catch (err) {
      console.warn('Could not verify the API key now; saving it anyway.', err);
    } finally {
      if (saveBtn) saveBtn.disabled = false;
    }
    STATE.apiKey = key;
    store.set(STORAGE_KEYS.apiKey, key);
    Modal.close($('api-key-modal'));
    showToast('API key saved.');
    loadLanguages();
    fetchSampleQueries();
  }

  // ============================================================
  // Language selection
  // ============================================================
  function renderLanguageMenu() {
    const menu = $('lang-dropdown-menu');
    if (!menu) return;
    menu.textContent = '';
    const list = STATE.languages.length
      ? STATE.languages
      : Object.keys(LANGUAGE_COPY).map((code) => ({ code, name: LANGUAGE_COPY[code].name, nativeName: LANGUAGE_COPY[code].nativeName, locale: LANGUAGE_COPY[code].locale, active: true }));
    list.forEach((l) => {
      const selected = l.code === STATE.lang;
      const item = h('button', {
        className: 'lang-menu-item' + (selected ? ' active' : ''),
        type: 'button',
        attrs: { role: 'menuitemradio', 'aria-checked': String(selected), 'data-lang': l.code },
        on: { click: (e) => { e.stopPropagation(); selectLanguage(l.code); closeLanguageMenu(true); } },
      },
      h('span', { className: 'lang-flag', text: '🇮🇳', attrs: { 'aria-hidden': 'true' } }),
      h('span', { className: 'lang-info' },
        h('span', { className: 'lang-name', text: l.nativeName || l.name }),
        h('span', { className: 'lang-meta', text: l.name + ' (' + (l.locale || l.code) + ')' + (l.active ? '' : ' — unavailable') })),
      h('span', { className: 'lang-check', text: selected ? '✓' : '', attrs: { 'aria-hidden': 'true' } }));
      if (!l.active) item.disabled = true;
      menu.append(item);
    });
  }

  function languageMenuOpen() {
    const wrapper = $('top-lang-dropdown');
    return !!(wrapper && wrapper.classList.contains('open'));
  }

  function openLanguageMenu(trigger) {
    const wrapper = $('top-lang-dropdown');
    if (!wrapper) return;
    wrapper.classList.add('open');
    STATE.langMenuTrigger = trigger || $('current-lang-btn');
    ['current-lang-btn', 'voice-lang-pill'].forEach((id) => {
      const el = $(id);
      if (el) el.setAttribute('aria-expanded', String(el === STATE.langMenuTrigger));
    });
    const active = document.querySelector('#lang-dropdown-menu .lang-menu-item.active') || document.querySelector('#lang-dropdown-menu .lang-menu-item');
    if (active) active.focus();
  }

  function closeLanguageMenu(restoreFocus) {
    const wrapper = $('top-lang-dropdown');
    if (!wrapper || !wrapper.classList.contains('open')) return;
    wrapper.classList.remove('open');
    ['current-lang-btn', 'voice-lang-pill'].forEach((id) => {
      const el = $(id);
      if (el) el.setAttribute('aria-expanded', 'false');
    });
    if (restoreFocus && STATE.langMenuTrigger) STATE.langMenuTrigger.focus();
  }

  function toggleLanguageMenu(event) {
    event.preventDefault();
    event.stopPropagation();
    if (languageMenuOpen()) closeLanguageMenu(false);
    else openLanguageMenu(event.currentTarget);
  }

  function selectLanguage(code) {
    if (!U.isLangCode(code)) return;
    if (STATE.languages.length && !STATE.languages.some((l) => l.code === code && l.active)) return;
    const previous = STATE.lang;
    if (previous === code) return;
    // A request in flight was asked in the old language: cancel it.
    cancelActive('language');
    STATE.lang = code;
    store.set(STORAGE_KEYS.lang, code);
    applyLanguageUI(true);
    renderLanguageMenu();
    fetchSampleQueries();
  }

  function applyLanguageUI(replaceWelcome) {
    const code = STATE.lang;
    const copy = langCopy(code);
    document.documentElement.lang = code;
    Array.from(document.body.classList)
      .filter((c) => c.startsWith('lang-'))
      .forEach((c) => document.body.classList.remove(c));
    document.body.classList.add('lang-' + code);

    setText('current-lang-label', langLabel(code));
    setText('voice-stage-lang-name', langLabel(code));
    if (STATE.voice === 'idle') setText('speak-prompt-text', copy.speakPrompt);
    const input = $('query-input');
    if (input) input.placeholder = copy.placeholder;
    setText('brand-app-title', copy.appTitle);
    setText('brand-app-subtitle', copy.appSubtitle);

    if (replaceWelcome) {
      const feed = $('chat-feed');
      const only = feed && feed.children.length === 1 ? messages.get(STATE.lastMessageId) : null;
      if (only && only.isWelcome) {
        resetFeed();
      }
    }
  }

  // ============================================================
  // Sample queries
  // ============================================================
  function setSampleSpinning(spinning) {
    ['refresh-queries-btn', 'modal-refresh-queries-btn'].forEach((id) => {
      const el = $(id);
      if (el) el.classList.toggle('spinning', spinning);
    });
  }

  async function loadStaticSamples(lang) {
    const copy = langCopy(lang);
    if (!copy.samplePoolFile) return [];
    if (!cachedPools[lang]) {
      const res = await fetch(copy.samplePoolFile + '?v=' + ASSET_VERSION, { signal: timeoutSignal(10000) });
      if (!res.ok) return [];
      cachedPools[lang] = U.normalizeSampleQueries(await res.json()); // drops ground_truth_answer
    }
    return U.shuffled(cachedPools[lang]).slice(0, 20);
  }

  async function fetchSampleQueries() {
    const seq = ++sampleRequestSeq;
    const lang = STATE.lang;
    setSampleSpinning(true);
    try {
      let list = [];
      try {
        const res = await apiFetch('/api/v1/sample-queries?lang=' + encodeURIComponent(lang) + '&count=20&_t=' + Date.now(), {
          cache: 'no-store',
          signal: timeoutSignal(10000),
        });
        if (res.ok) {
          const data = await res.json();
          list = U.normalizeSampleQueries(data && data.queries);
        }
      } catch (err) {
        console.warn('Sample queries endpoint unavailable, using the static pool:', err);
      }
      if (seq !== sampleRequestSeq) return; // a newer request (language switch) won
      if (!list.length) {
        try {
          list = await loadStaticSamples(lang);
        } catch (err) {
          console.warn('Static sample pool unavailable:', err);
          list = [];
        }
        if (seq !== sampleRequestSeq) return;
      }
      STATE.samples = list;
      setText('sample-count', String(list.length));
      renderSampleQueries();
      if (Modal.isOpen($('all-queries-modal'))) applyModalFilters();
    } finally {
      if (seq === sampleRequestSeq) setSampleSpinning(false);
    }
  }

  function renderSampleQueries() {
    const container = $('sample-queries-list');
    if (!container) return;
    container.textContent = '';
    if (!STATE.samples.length) {
      container.append(h('p', { className: 'empty-state-text', text: 'No samples available' }));
      return;
    }
    STATE.samples.slice(0, 3).forEach((s) => {
      container.append(h('button', {
        className: 'sample-query-item',
        type: 'button',
        title: 'Load into the input box',
        on: { click: () => { useQueryInInput(s.question, { queryId: s.query_id }); closeDrawer(); } },
      },
      h('span', { className: 'sample-text', text: s.question }),
      s.query_type ? h('span', { className: 'sample-tag', text: s.query_type }) : null));
    });
  }

  function useQueryInInput(text, opts) {
    const o = opts || {};
    const input = $('query-input');
    if (!input) return;
    input.value = text;
    STATE.pendingSample = Number.isInteger(o.queryId) ? { text, queryId: o.queryId } : null;
    handleInputResize();
    const dock = document.querySelector('.dock-container');
    if (dock && !prefersReducedMotion()) {
      dock.classList.remove('dock-loaded-pulse');
      void dock.offsetWidth; // restart the animation
      dock.classList.add('dock-loaded-pulse');
      setTimeout(() => dock.classList.remove('dock-loaded-pulse'), 800);
    }
    if (o.autoSubmit) {
      submitCurrentQuery();
    } else {
      input.focus();
    }
  }

  function fillRandomSampleQuery() {
    if (!STATE.samples.length) {
      showToast('No sample queries are loaded yet.');
      return;
    }
    const s = STATE.samples[Math.floor(Math.random() * STATE.samples.length)];
    useQueryInInput(s.question, { queryId: s.query_id });
  }

  function openAllQueriesModal() {
    const modal = $('all-queries-modal');
    if (!modal) return;
    // Fresh filter state every time
    STATE.modalCategory = 'all';
    syncCategoryButtons();
    const search = $('query-modal-search');
    if (search) search.value = '';
    applyModalFilters();
    closeDrawer();
    Modal.open(modal, { initialFocus: '#query-modal-search' });
  }

  function syncCategoryButtons() {
    document.querySelectorAll('#modal-category-filters .cat-filter-btn').forEach((btn) => {
      const active = btn.dataset.cat === STATE.modalCategory;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', String(active));
    });
  }

  function applyModalFilters() {
    const search = $('query-modal-search');
    const term = search ? search.value.toLowerCase().trim() : '';
    const filtered = STATE.samples.filter((s) => {
      if (!U.matchesCategory(s.query_type, STATE.modalCategory)) return false;
      if (!term) return true;
      return s.question.toLowerCase().includes(term) || String(s.query_type || '').toLowerCase().includes(term);
    });
    renderModalQueries(filtered);
  }

  function renderModalQueries(list) {
    const container = $('modal-all-queries-list');
    if (!container) return;
    setText('modal-queries-count', String(list.length));
    container.textContent = '';
    if (!list.length) {
      container.append(h('div', { className: 'empty-state-text modal-empty', text: 'No questions match your filter.' }));
      return;
    }
    const modal = $('all-queries-modal');
    list.forEach((s, idx) => {
      const type = s.query_type || 'QUESTION';
      const fill = () => {
        Modal.close(modal);
        useQueryInInput(s.question, { queryId: s.query_id });
      };
      const ask = (e) => {
        e.stopPropagation();
        if (STATE.busy) {
          showToast('Please wait for the current answer (or press Stop).');
          return;
        }
        Modal.close(modal);
        useQueryInInput(s.question, { queryId: s.query_id, autoSubmit: true });
      };
      const card = h('div', { className: 'modal-query-card' },
        h('div', { className: 'query-card-header' },
          h('span', { className: 'query-type-tag ' + U.queryTypeClass(type), text: type }),
          h('span', { className: 'query-index', text: '#' + (idx + 1) })),
        h('p', { className: 'query-card-text', text: s.question }),
        h('div', { className: 'query-card-footer' },
          h('button', { className: 'btn-use-query', type: 'button', title: 'Fill into the input box', on: { click: (e) => { e.stopPropagation(); fill(); } } },
            h('span', { text: 'Fill Input' })),
          h('button', { className: 'btn-ask-query', type: 'button', title: 'Ask this question now', on: { click: ask } },
            h('span', { text: 'Ask' }))));
      // Mouse convenience: clicking the card body fills the input (keyboard users have the buttons).
      card.addEventListener('click', fill);
      if (STATE.busy) card.querySelector('.btn-ask-query').disabled = true;
      container.append(card);
    });
  }

  // ============================================================
  // Recent history (sidebar + modal)
  // ============================================================
  function persistRecent() {
    store.setJSON(STORAGE_KEYS.history, STATE.recent);
  }

  function saveRecent(query, mode, lang) {
    const item = { query: String(query).slice(0, 1000), mode, lang: U.isLangCode(lang) ? lang : null, time: U.formatClock(new Date()) };
    STATE.recent.push(item);
    if (STATE.recent.length > 50) STATE.recent = STATE.recent.slice(-50);
    persistRecent();
    renderRecent();
    return item;
  }

  function historyButton(item, onPick, extraClass) {
    return h('button', {
      className: 'history-item' + (extraClass ? ' ' + extraClass : ''),
      type: 'button',
      title: 'Load into the input box',
      on: { click: () => onPick(item) },
    },
    h('span', { className: 'history-item-top' },
      h('span', null,
        item.mode === 'voice' ? '🎙️ Voice' : '💬 Text',
        item.lang ? h('span', { className: 'history-lang-badge', text: item.lang.toUpperCase() }) : null),
      h('span', { text: item.time })),
    h('span', { className: 'history-item-text', text: item.query }));
  }

  function pickHistoryItem(item) {
    if (item.lang && item.lang !== STATE.lang && !STATE.busy) selectLanguage(item.lang);
    useQueryInInput(item.query);
    closeDrawer();
  }

  function renderRecent() {
    const container = $('history-list');
    if (!container) return;
    container.textContent = '';
    if (!STATE.recent.length) {
      container.append(h('p', { className: 'empty-state-text', text: 'No queries asked yet.' }));
      return;
    }
    STATE.recent.slice(-6).reverse().forEach((item) => container.append(historyButton(item, pickHistoryItem)));
  }

  function openHistoryModal() {
    const modal = $('history-modal');
    const list = $('modal-history-list');
    if (!modal || !list) return;
    list.textContent = '';
    if (!STATE.recent.length) {
      list.append(h('p', { className: 'empty-state-text', text: 'No history records yet.' }));
    } else {
      STATE.recent.slice().reverse().forEach((item) => {
        list.append(historyButton(item, (picked) => {
          Modal.close(modal);
          pickHistoryItem(picked);
        }, 'history-modal-item'));
      });
    }
    closeDrawer();
    Modal.open(modal);
  }

  function clearHistory() {
    if (!STATE.recent.length) return;
    if (!window.confirm('Clear the query history stored in this browser?')) return;
    STATE.recent = [];
    store.remove(STORAGE_KEYS.history);
    renderRecent();
  }

  // ============================================================
  // Input dock
  // ============================================================
  function handleInputResize() {
    const input = $('query-input');
    if (!input) return;
    input.style.height = 'auto';
    input.style.height = Math.max(42, Math.min(input.scrollHeight, 150)) + 'px';
    setText('char-counter', input.value.length + ' / ' + MAX_QUERY_CHARS);
  }

  function handleInputKeyDown(e) {
    // Never submit while an IME (Gujarati/Hindi keyboards) is composing.
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitCurrentQuery();
    }
  }

  function syncToggle(inputId, textId, labelId, onText, offText) {
    const input = $(inputId);
    if (!input) return;
    setText(textId, input.checked ? onText : offText);
    const label = $(labelId);
    if (label) label.classList.toggle('active', input.checked);
  }

  function syncAllToggles() {
    syncToggle('eval-toggle', 'eval-toggle-text', 'eval-toggle-label', 'Eval: ON', 'Eval: OFF');
    syncToggle('tts-toggle', 'tts-toggle-text', 'tts-toggle-label', 'Voice Reply: ON', 'Voice Reply: OFF');
    syncToggle('autodetect-toggle', 'autodetect-toggle-text', 'autodetect-toggle-label', 'Auto-detect: ON', 'Auto-detect: OFF');
  }

  // ============================================================
  // Mode, drawer, telemetry panel, theme
  // ============================================================
  function switchMode(mode) {
    if (mode !== 'voice' && mode !== 'chat') return;
    STATE.mode = mode;
    const voiceBtn = $('btn-mode-voice');
    const chatBtn = $('btn-mode-chat');
    if (voiceBtn) {
      voiceBtn.classList.toggle('active', mode === 'voice');
      voiceBtn.setAttribute('aria-pressed', String(mode === 'voice'));
    }
    if (chatBtn) {
      chatBtn.classList.toggle('active', mode === 'chat');
      chatBtn.setAttribute('aria-pressed', String(mode === 'chat'));
    }
    const stage = $('voice-stage');
    if (stage) stage.hidden = mode !== 'voice';
    if (mode === 'chat') {
      if (STATE.voice === 'connecting' || STATE.voice === 'recording') cancelVoice('mode');
      const input = $('query-input');
      if (input) input.focus();
    } else {
      requestAnimationFrame(() => (STATE.voice === 'recording' ? drawLiveWaveform() : drawIdleWaveform()));
    }
    closeDrawer();
  }

  const mqMobile = window.matchMedia('(max-width: 900px)');

  function drawerOpen() {
    const sb = $('sidebar-left');
    return !!(sb && sb.classList.contains('open'));
  }

  function setDrawer(open) {
    const sb = $('sidebar-left');
    const backdrop = $('sidebar-backdrop');
    const toggle = $('sidebar-toggle-btn');
    if (!sb) return;
    const shouldOpen = open && mqMobile.matches;
    sb.classList.toggle('open', shouldOpen);
    if (backdrop) backdrop.hidden = !shouldOpen;
    if (toggle) {
      toggle.setAttribute('aria-expanded', String(shouldOpen));
      toggle.setAttribute('aria-label', shouldOpen ? 'Close menu' : 'Open menu');
    }
    if (shouldOpen) {
      const first = sb.querySelector('button:not([disabled])');
      if (first) requestAnimationFrame(() => first.focus());
    } else if (sb.contains(document.activeElement) && mqMobile.matches && toggle) {
      toggle.focus();
    }
  }

  function closeDrawer() {
    if (drawerOpen()) setDrawer(false);
  }

  function telemetryVisible() {
    const layout = document.querySelector('.app-layout');
    const panel = $('telemetry-panel');
    if (mqMobile.matches) return !!(panel && panel.classList.contains('open'));
    return !(layout && layout.classList.contains('inspector-collapsed'));
  }

  function setTelemetry(visible) {
    const layout = document.querySelector('.app-layout');
    const panel = $('telemetry-panel');
    if (mqMobile.matches) {
      if (panel) panel.classList.toggle('open', visible);
    } else if (layout) {
      layout.classList.toggle('inspector-collapsed', !visible);
    }
    const toggle = $('toggle-inspector-btn');
    if (toggle) toggle.setAttribute('aria-expanded', String(telemetryVisible()));
  }

  function syncResponsiveState() {
    if (!mqMobile.matches) setDrawer(false);
    const toggle = $('toggle-inspector-btn');
    if (toggle) toggle.setAttribute('aria-expanded', String(telemetryVisible()));
  }

  const mqDark = window.matchMedia('(prefers-color-scheme: dark)');

  function effectiveTheme() {
    const explicit = document.documentElement.getAttribute('data-theme');
    if (explicit === 'light' || explicit === 'dark') return explicit;
    return mqDark.matches ? 'dark' : 'light';
  }

  function syncThemeButton() {
    const dark = effectiveTheme() === 'dark';
    setText('theme-icon', dark ? '🌙' : '☀️');
    const btn = $('theme-toggle-btn');
    if (btn) btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  }

  function toggleTheme() {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    store.set(STORAGE_KEYS.theme, next);
    syncThemeButton();
    if (STATE.voice !== 'recording') drawIdleWaveform(); // re-read canvas colours
  }

  // ============================================================
  // New session
  // ============================================================
  function resetFeed() {
    messages.forEach(revokeMessageAudio);
    messages.clear();
    STATE.lastMessageId = null;
    const feed = $('chat-feed');
    if (feed) feed.textContent = '';
    appendWelcomeMessage();
  }

  function startNewSession() {
    STATE.sessionEpoch += 1;
    cancelActive('session');
    stopAllAudio();
    STATE.session = [];
    STATE.pendingSample = null;
    resetFeed();
    resetLatencySidebar();
    setScorecardPlaceholder('off');
    const card = $('voice-transcript-card');
    if (card) card.hidden = true;
    closeDrawer();
    showToast('Started a new conversation.');
  }

  // ============================================================
  // Event wiring
  // ============================================================
  function wireEvents() {
    on('new-session-btn', 'click', startNewSession);
    on('btn-mode-voice', 'click', () => switchMode('voice'));
    on('btn-mode-chat', 'click', () => switchMode('chat'));
    on('refresh-queries-btn', 'click', fetchSampleQueries);
    on('modal-refresh-queries-btn', 'click', fetchSampleQueries);
    on('browse-queries-btn', 'click', openAllQueriesModal);
    on('dock-browse-btn', 'click', openAllQueriesModal);
    on('open-history-btn', 'click', openHistoryModal);
    on('clear-history-btn', 'click', clearHistory);
    on('use-sample-btn', 'click', fillRandomSampleQuery);

    on('current-lang-btn', 'click', toggleLanguageMenu);
    on('voice-lang-pill', 'click', toggleLanguageMenu);
    on('toggle-inspector-btn', 'click', () => setTelemetry(!telemetryVisible()));
    on('close-inspector-btn', 'click', () => {
      setTelemetry(false);
      const toggle = $('toggle-inspector-btn');
      if (toggle) toggle.focus();
    });
    on('theme-toggle-btn', 'click', toggleTheme);
    on('sidebar-toggle-btn', 'click', () => setDrawer(!drawerOpen()));
    on('sidebar-backdrop', 'click', () => setDrawer(false));

    on('main-mic-btn', 'click', toggleVoiceRecording);
    on('stop-generation-btn', 'click', () => {
      cancelActive('user');
      stopAllAudio();
    });

    const input = $('query-input');
    if (input) {
      input.addEventListener('input', () => {
        handleInputResize();
        if (STATE.pendingSample && input.value.trim() !== STATE.pendingSample.text.trim()) {
          STATE.pendingSample = null; // edited: no longer the benchmark question
        }
      });
      input.addEventListener('keydown', handleInputKeyDown);
    }
    on('send-query-btn', 'click', submitCurrentQuery);

    on('upload-audio-btn', 'click', () => {
      const fileInput = $('audio-file-input');
      if (fileInput && !STATE.busy) fileInput.click();
    });
    on('audio-file-input', 'change', (e) => {
      const file = e.target.files && e.target.files[0];
      e.target.value = '';
      handleAudioFile(file);
    });

    on('eval-toggle', 'change', syncAllToggles);
    on('tts-toggle', 'change', () => {
      syncAllToggles();
      if (!readToggle('tts-toggle')) stopAllAudio();
    });
    on('autodetect-toggle', 'change', () => {
      syncAllToggles();
      store.set(STORAGE_KEYS.autoDetect, readToggle('autodetect-toggle') ? '1' : '0');
    });

    on('query-modal-search', 'input', applyModalFilters);
    document.querySelectorAll('#modal-category-filters .cat-filter-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        STATE.modalCategory = btn.dataset.cat || 'all';
        syncCategoryButtons();
        applyModalFilters();
      });
    });

    on('gt-copy-btn', 'click', () => {
      const gt = $('gt-text');
      if (gt && gt.textContent && gt.textContent !== '--') copyText(gt.textContent);
    });
    on('gt-speak-btn', 'click', async () => {
      const gt = $('gt-text');
      const btn = $('gt-speak-btn');
      if (!gt || !gt.textContent || gt.textContent === '--') return;
      if (Player.isPlaying('ground-truth')) {
        Player.stop();
        return;
      }
      if (btn) btn.disabled = true;
      try {
        const b64 = await synthesize(gt.textContent, STATE.lastEvalLanguage || STATE.lang);
        const url = URL.createObjectURL(new Blob([U.base64ToBytes(b64)], { type: 'audio/mpeg' }));
        if (STATE.gtAudioUrl) URL.revokeObjectURL(STATE.gtAudioUrl);
        STATE.gtAudioUrl = url;
        stopAllAudio();
        Player.playAll([url], 'ground-truth');
      } catch (err) {
        showToast('Voice playback failed: ' + err.message, 'error');
      } finally {
        if (btn) btn.disabled = false;
      }
    });

    const apiForm = $('api-key-form');
    if (apiForm) apiForm.addEventListener('submit', saveApiKey);

    ['all-queries-modal', 'history-modal', 'api-key-modal'].forEach((id) => Modal.wire($(id)));

    // One document-level click handler: close the language menu on outside clicks.
    document.addEventListener('click', (e) => {
      const wrapper = $('top-lang-dropdown');
      if (!languageMenuOpen()) return;
      if (wrapper && wrapper.contains(e.target)) return;
      if (e.target.closest && e.target.closest('#voice-lang-pill')) return;
      closeLanguageMenu(false);
    });

    // One document-level keydown handler: modals > language menu > drawer > telemetry.
    document.addEventListener('keydown', (e) => {
      if (Modal.handleKeydown(e)) return;
      if (e.key !== 'Escape') return;
      if (languageMenuOpen()) {
        e.preventDefault();
        closeLanguageMenu(true);
      } else if (drawerOpen()) {
        e.preventDefault();
        setDrawer(false);
      } else if (mqMobile.matches && telemetryVisible()) {
        e.preventDefault();
        setTelemetry(false);
      }
    });

    const sc = scroller();
    if (sc) {
      sc.addEventListener('scroll', () => {
        STATE.stickToBottom = U.isNearBottom(sc.scrollTop, sc.scrollHeight, sc.clientHeight, 80);
      }, { passive: true });
    }

    const onMq = (mq, fn) => {
      if (typeof mq.addEventListener === 'function') mq.addEventListener('change', fn);
      else if (typeof mq.addListener === 'function') mq.addListener(fn);
    };
    onMq(mqMobile, syncResponsiveState);
    onMq(mqDark, () => {
      syncThemeButton();
      if (STATE.voice !== 'recording') drawIdleWaveform();
    });

    let resizeRaf = 0;
    window.addEventListener('resize', () => {
      if (resizeRaf) return;
      resizeRaf = requestAnimationFrame(() => {
        resizeRaf = 0;
        if (STATE.voice !== 'recording') drawIdleWaveform();
      });
    });

    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) checkHealth();
    });

    window.addEventListener('beforeunload', () => {
      cancelActive('session');
      messages.forEach(revokeMessageAudio);
    });
  }

  // ============================================================
  // Init
  // ============================================================
  function initApp() {
    const savedLang = store.get(STORAGE_KEYS.lang, 'gu');
    STATE.lang = U.isLangCode(savedLang) && LANGUAGE_COPY[savedLang] ? savedLang : 'gu';
    STATE.apiKey = store.get(STORAGE_KEYS.apiKey, '') || '';
    STATE.recent = U.sanitizeHistoryItems(store.getJSON(STORAGE_KEYS.history, []));

    const autoDetect = $('autodetect-toggle');
    if (autoDetect) autoDetect.checked = store.get(STORAGE_KEYS.autoDetect, '1') !== '0';
    const evalToggle = $('eval-toggle');
    if (evalToggle) evalToggle.checked = false;
    const ttsToggle = $('tts-toggle');
    if (ttsToggle) ttsToggle.checked = false;

    wireEvents();
    syncAllToggles();
    syncThemeButton();
    renderLanguageMenu();
    applyLanguageUI(false);
    renderRecent();
    resetFeed();
    resetLatencySidebar();
    setScorecardPlaceholder('off');
    setVoiceState('idle');
    syncResponsiveState();
    handleInputResize();
    refreshControls();

    checkHealth().then(() => {
      loadLanguages();
      fetchSampleQueries();
    });
    scheduleHealthChecks();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }
})();
