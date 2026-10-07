/**
 * Voice RAG — pure helper functions (v5.0.0)
 * ===========================================
 * No DOM access here, so every function can be unit-tested in Node:
 *   const U = require('./utils.js');
 * In the browser this file is loaded as a classic script and exposes `window.RagUtils`.
 */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.RagUtils = api;
  }
})(typeof self !== 'undefined' ? self : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  'use strict';

  // ------------------------------------------------------------------
  // Escaping & safe answer formatting
  // ------------------------------------------------------------------
  const HTML_ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

  /** Escape a value for use as HTML text or inside a quoted attribute. */
  function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value).replace(/[&<>"']/g, (ch) => HTML_ESCAPES[ch]);
  }

  /** Inline formatting on ALREADY-ESCAPED text: **bold** and `code` only. */
  function formatInline(escaped) {
    return escaped
      .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`\n]+?)`/g, '<code>$1</code>');
  }

  /**
   * Turn model output into a small, safe HTML fragment.
   * The text is HTML-escaped FIRST; only a fixed set of tags generated here
   * (<p>, <br>, <ul>, <ol>, <li>, <strong>, <code>) can appear in the result.
   */
  function formatAnswerHtml(text) {
    if (text === null || text === undefined) return '';
    const lines = String(text).replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let paragraph = [];
    let listType = null; // 'ul' | 'ol'

    const flushParagraph = () => {
      if (paragraph.length) {
        out.push('<p>' + paragraph.join('<br>') + '</p>');
        paragraph = [];
      }
    };
    const closeList = () => {
      if (listType) {
        out.push('</' + listType + '>');
        listType = null;
      }
    };

    for (const rawLine of lines) {
      const line = rawLine.replace(/\s+$/, '');
      const bullet = /^\s*[*\-•]\s+(.+)$/.exec(line);
      const numbered = /^\s*(\d{1,3})[.)]\s+(.+)$/.exec(line);
      const heading = /^\s*#{1,6}\s+(.+)$/.exec(line);

      if (bullet || numbered) {
        flushParagraph();
        const type = bullet ? 'ul' : 'ol';
        if (listType !== type) {
          closeList();
          if (type === 'ol' && numbered[1] !== '1') {
            out.push('<ol start="' + parseInt(numbered[1], 10) + '">');
          } else {
            out.push('<' + type + '>');
          }
          listType = type;
        }
        const content = bullet ? bullet[1] : numbered[2];
        out.push('<li>' + formatInline(escapeHtml(content)) + '</li>');
        continue;
      }

      closeList();
      if (!line.trim()) {
        flushParagraph();
        continue;
      }
      if (heading) {
        flushParagraph();
        out.push('<p><strong>' + formatInline(escapeHtml(heading[1])) + '</strong></p>');
        continue;
      }
      paragraph.push(formatInline(escapeHtml(line)));
    }
    flushParagraph();
    closeList();
    return out.join('');
  }

  // ------------------------------------------------------------------
  // Server-Sent Events parser (WHATWG event-stream rules)
  // ------------------------------------------------------------------
  /**
   * createSSEParser(onEvent) -> { push(textChunk), flush() }
   * - accepts \n, \r\n and \r line endings (also when split across chunks)
   * - joins multi-line `data:` fields with "\n"
   * - ignores comments (":...") and unknown fields
   * - flush() processes a trailing line without a newline and dispatches the
   *   last event even if the stream did not end with a blank line.
   * onEvent receives { event, data } where data is the raw string.
   */
  function createSSEParser(onEvent) {
    let buffer = '';
    let eventType = '';
    let dataLines = [];
    let hasData = false;

    function dispatch() {
      if (hasData) {
        onEvent({ event: eventType || 'message', data: dataLines.join('\n') });
      }
      eventType = '';
      dataLines = [];
      hasData = false;
    }

    function processLine(line) {
      if (line === '') {
        dispatch();
        return;
      }
      if (line.charCodeAt(0) === 58 /* ':' */) return; // comment
      const idx = line.indexOf(':');
      let field;
      let value;
      if (idx === -1) {
        field = line;
        value = '';
      } else {
        field = line.slice(0, idx);
        value = line.slice(idx + 1);
        if (value.charCodeAt(0) === 32 /* ' ' */) value = value.slice(1);
      }
      if (field === 'event') {
        eventType = value;
      } else if (field === 'data') {
        dataLines.push(value);
        hasData = true;
      }
      // "id" and "retry" are not needed by this client.
    }

    return {
      push(chunk) {
        if (!chunk) return;
        buffer += chunk;
        let start = 0;
        let i = 0;
        while (i < buffer.length) {
          const ch = buffer.charCodeAt(i);
          if (ch === 10 /* \n */) {
            processLine(buffer.slice(start, i));
            start = i + 1;
          } else if (ch === 13 /* \r */) {
            if (i + 1 >= buffer.length) break; // might be the first half of \r\n
            processLine(buffer.slice(start, i));
            if (buffer.charCodeAt(i + 1) === 10) i += 1;
            start = i + 1;
          }
          i += 1;
        }
        buffer = buffer.slice(start);
      },
      flush() {
        if (buffer.length) {
          let rest = buffer;
          buffer = '';
          if (rest.endsWith('\r')) rest = rest.slice(0, -1);
          rest.split(/\r\n|\r|\n/).forEach(processLine);
        }
        dispatch();
      },
    };
  }

  // ------------------------------------------------------------------
  // Error formatting
  // ------------------------------------------------------------------
  /** FastAPI `detail` (string | [{loc, msg, type}] | object) -> readable text. */
  function formatErrorDetail(detail) {
    if (detail === null || detail === undefined) return '';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (item === null || item === undefined) return '';
          if (typeof item === 'string') return item;
          if (typeof item !== 'object') return String(item);
          let loc = '';
          if (Array.isArray(item.loc)) {
            loc = item.loc
              .filter((p) => p !== 'body' && p !== 'query' && p !== 'form' && p !== 'path')
              .join('.');
          } else if (item.loc) {
            loc = String(item.loc);
          }
          const msg = item.msg || item.message || safeJson(item);
          return loc ? loc + ': ' + msg : String(msg);
        })
        .filter(Boolean)
        .join('; ');
    }
    if (typeof detail === 'object') {
      if (detail.msg || detail.message || detail.detail) {
        return formatErrorDetail(detail.msg || detail.message || detail.detail);
      }
      return safeJson(detail);
    }
    return String(detail);
  }

  function safeJson(value) {
    try {
      return JSON.stringify(value);
    } catch (e) {
      return String(value);
    }
  }

  /** Retry-After header (delta-seconds or HTTP date) -> whole seconds, or null. */
  function parseRetryAfter(value, nowMs) {
    if (value === null || value === undefined || value === '') return null;
    const str = String(value).trim();
    if (/^\d+(\.\d+)?$/.test(str)) return Math.max(0, Math.ceil(parseFloat(str)));
    const when = Date.parse(str);
    if (Number.isNaN(when)) return null;
    const now = typeof nowMs === 'number' ? nowMs : Date.now();
    return Math.max(0, Math.ceil((when - now) / 1000));
  }

  /** Human-readable message for an HTTP error status. */
  function describeHttpError(status, detailText, retryAfter, opts) {
    const options = opts || {};
    const detail = detailText ? String(detailText) : '';
    switch (status) {
      case 401:
        return 'API key missing or invalid.' + (detail ? ' (' + detail + ')' : '');
      case 403:
        return 'Access denied.' + (detail ? ' ' + detail : '');
      case 413:
        return 'File is too large' + (options.maxMb ? ' (maximum ' + options.maxMb + ' MB)' : '') + '.' +
          (detail ? ' ' + detail : '');
      case 415:
        return 'Unsupported audio format. Use WAV, MP3, M4A, OGG, WEBM, FLAC or MP4.' + (detail ? ' (' + detail + ')' : '');
      case 422:
        return 'Invalid request: ' + (detail || 'validation failed') + '.';
      case 429: {
        const secs = parseRetryAfter(retryAfter, options.nowMs);
        return 'Too many requests.' + (secs !== null ? ' Please retry in ' + secs + ' s.' : ' Please wait a moment and retry.');
      }
      default:
        if (status >= 500) return 'Server error (' + status + ')' + (detail ? ': ' + detail : '.');
        return detail || 'Request failed (HTTP ' + status + ').';
    }
  }

  // ------------------------------------------------------------------
  // Sentence-level TTS ordering
  // ------------------------------------------------------------------
  /**
   * Re-orders chunks that carry a 0-based `seq`.
   * add(seq, item) returns the items that are now playable, in order.
   * drain() returns everything still buffered (sorted), skipping gaps.
   */
  function createSeqBuffer(startSeq) {
    let next = typeof startSeq === 'number' ? startSeq : 0;
    let maxSeen = next - 1;
    const pending = new Map();
    return {
      add(seq, item) {
        let s = Number(seq);
        if (seq === null || seq === undefined || !Number.isInteger(s)) s = Math.max(next, maxSeen + 1);
        if (s < next || pending.has(s)) return []; // duplicate / already played
        pending.set(s, item);
        if (s > maxSeen) maxSeen = s;
        const ready = [];
        while (pending.has(next)) {
          ready.push(pending.get(next));
          pending.delete(next);
          next += 1;
        }
        return ready;
      },
      drain() {
        const keys = Array.from(pending.keys()).sort((a, b) => a - b);
        const items = keys.map((k) => pending.get(k));
        pending.clear();
        if (keys.length) next = keys[keys.length - 1] + 1;
        return items;
      },
      reset(start) {
        pending.clear();
        next = typeof start === 'number' ? start : 0;
        maxSeen = next - 1;
      },
      get nextSeq() {
        return next;
      },
      get pendingCount() {
        return pending.size;
      },
    };
  }

  // ------------------------------------------------------------------
  // Conversation memory
  // ------------------------------------------------------------------
  /**
   * Last `maxMessages` {role, content} messages (oldest first) for the API.
   * Drops empty/invalid entries, trims and caps content, and never starts with
   * an assistant message.
   */
  function buildHistory(messages, maxMessages, maxChars) {
    const limit = typeof maxMessages === 'number' ? maxMessages : 6;
    const cap = typeof maxChars === 'number' ? maxChars : 2000;
    if (!Array.isArray(messages) || limit <= 0) return [];
    const clean = messages
      .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim())
      .map((m) => ({ role: m.role, content: m.content.trim().slice(0, cap) }));
    let recent = clean.slice(-limit);
    while (recent.length && recent[0].role !== 'user') recent = recent.slice(1);
    return recent;
  }

  // ------------------------------------------------------------------
  // Small formatting helpers
  // ------------------------------------------------------------------
  function toNumber(value) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function formatMs(value, digits) {
    const n = toNumber(value);
    if (n === null) return '--';
    return n.toFixed(typeof digits === 'number' ? digits : 1) + ' ms';
  }

  /** Metric score -> "0.83", or "n/a" when the judge did not produce it. */
  function formatScore(value) {
    const n = toNumber(value);
    return n === null ? 'n/a' : n.toFixed(2);
  }

  function formatClock(date) {
    const d = date instanceof Date ? date : new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  }

  /** Cut long text at `max` characters and add an ellipsis only when something was cut. */
  function truncateText(text, max) {
    const str = text === null || text === undefined ? '' : String(text);
    const chars = Array.from(str);
    if (chars.length <= max) return str;
    return chars.slice(0, max).join('').replace(/\s+$/, '') + '…';
  }

  /** Only absolute http(s) URLs are allowed as links. */
  function safeHttpUrl(url) {
    if (!url || typeof url !== 'string') return null;
    try {
      const parsed = new URL(url.trim());
      if (parsed.protocol === 'http:' || parsed.protocol === 'https:') return parsed.href;
    } catch (e) {
      /* not a URL */
    }
    return null;
  }

  function normalizeForCompare(text) {
    return String(text || '').replace(/\s+/g, ' ').trim().toLowerCase();
  }

  /** True when the server searched for something other than what the user typed. */
  function queriesDiffer(query, retrievalQuery) {
    if (!retrievalQuery) return false;
    return normalizeForCompare(query) !== normalizeForCompare(retrievalQuery);
  }

  function isNearBottom(scrollTop, scrollHeight, clientHeight, threshold) {
    const t = typeof threshold === 'number' ? threshold : 80;
    return scrollHeight - (scrollTop + clientHeight) <= t;
  }

  function latencyRating(totalMs) {
    const n = toNumber(totalMs);
    if (n === null) return { cls: '', label: 'Idle' };
    if (n < 1200) return { cls: 'good', label: 'Fast' };
    if (n < 2500) return { cls: 'medium', label: 'Moderate' };
    return { cls: 'slow', label: 'Slow' };
  }

  function verdictFor(score) {
    const n = toNumber(score);
    if (n === null) return { cls: 'failed', label: 'Score unavailable' };
    if (n >= 0.85) return { cls: 'excellent', label: 'Excellent quality' };
    if (n >= 0.7) return { cls: 'high', label: 'High quality' };
    if (n >= 0.5) return { cls: 'fair', label: 'Fair alignment' };
    return { cls: 'low', label: 'Low alignment' };
  }

  const QUERY_TYPES = ['description', 'entity', 'location', 'numeric', 'person'];

  function queryTypeClass(queryType) {
    const t = String(queryType || '').toUpperCase();
    if (t.startsWith('DESC')) return 'tag-desc';
    if (t.startsWith('ENT')) return 'tag-entity';
    if (t.startsWith('LOC')) return 'tag-location';
    if (t.startsWith('NUM')) return 'tag-numeric';
    if (t.startsWith('PERS')) return 'tag-person';
    return 'tag-default';
  }

  function matchesCategory(queryType, category) {
    if (!category || category === 'all') return true;
    return String(queryType || '').trim().toLowerCase() === String(category).toLowerCase();
  }

  /** Keep only {question, query_type, query_id}: never carry ground-truth answers into the UI. */
  function normalizeSampleQueries(list) {
    if (!Array.isArray(list)) return [];
    const out = [];
    for (const item of list) {
      if (typeof item === 'string') {
        if (item.trim()) out.push({ question: item.trim(), query_type: null, query_id: null });
        continue;
      }
      if (!item || typeof item !== 'object') continue;
      const question = typeof item.question === 'string' ? item.question.trim() : '';
      if (!question) continue;
      const qid = item.query_id === null || item.query_id === undefined || item.query_id === '' ? NaN : Number(item.query_id);
      out.push({
        question,
        query_type: typeof item.query_type === 'string' ? item.query_type : null,
        query_id: Number.isInteger(qid) ? qid : null,
      });
    }
    return out;
  }

  /** Validate history items read back from localStorage. */
  function sanitizeHistoryItems(list, maxItems) {
    const limit = typeof maxItems === 'number' ? maxItems : 50;
    if (!Array.isArray(list)) return [];
    const out = [];
    for (const item of list) {
      if (!item || typeof item !== 'object' || typeof item.query !== 'string' || !item.query.trim()) continue;
      out.push({
        query: item.query.slice(0, 1000),
        mode: item.mode === 'voice' ? 'voice' : 'text',
        lang: isLangCode(item.lang) ? item.lang : null,
        time: typeof item.time === 'string' ? item.time.slice(0, 40) : '',
      });
    }
    return out.slice(-limit);
  }

  function isLangCode(value) {
    return typeof value === 'string' && /^[a-z]{2,3}$/.test(value);
  }

  function shuffled(list, random) {
    const rnd = typeof random === 'function' ? random : Math.random;
    const arr = list.slice();
    for (let i = arr.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rnd() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  // ------------------------------------------------------------------
  // URLs, storage, audio
  // ------------------------------------------------------------------
  /** Build ws(s)://host/path?params from a location-like object. */
  function buildWsUrl(loc, path, params) {
    const proto = loc && loc.protocol === 'https:' ? 'wss:' : 'ws:';
    const p = params || {};
    const query = Object.keys(p)
      .filter((k) => p[k] !== undefined && p[k] !== null && p[k] !== '')
      .map((k) => encodeURIComponent(k) + '=' + encodeURIComponent(String(p[k])))
      .join('&');
    return proto + '//' + loc.host + path + (query ? '?' + query : '');
  }

  /** localStorage wrapper that never throws (private mode, blocked storage, bad JSON). */
  function createSafeStorage(getStorage) {
    function storage() {
      try {
        return typeof getStorage === 'function' ? getStorage() : null;
      } catch (e) {
        return null;
      }
    }
    return {
      get(key, fallback) {
        const def = fallback === undefined ? null : fallback;
        try {
          const s = storage();
          const v = s ? s.getItem(key) : null;
          return v === null || v === undefined ? def : v;
        } catch (e) {
          return def;
        }
      },
      set(key, value) {
        try {
          const s = storage();
          if (!s) return false;
          s.setItem(key, String(value));
          return true;
        } catch (e) {
          return false;
        }
      },
      remove(key) {
        try {
          const s = storage();
          if (s) s.removeItem(key);
        } catch (e) {
          /* ignore */
        }
      },
      getJSON(key, fallback) {
        const raw = this.get(key, null);
        if (raw === null) return fallback;
        try {
          return JSON.parse(raw);
        } catch (e) {
          return fallback;
        }
      },
      setJSON(key, value) {
        let raw;
        try {
          raw = JSON.stringify(value);
        } catch (e) {
          return false;
        }
        return this.set(key, raw);
      },
    };
  }

  function base64ToBytes(b64) {
    const clean = String(b64 || '').replace(/^data:[^,]*,/, '').replace(/\s+/g, '');
    if (typeof atob === 'function') {
      const bin = atob(clean);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
      return bytes;
    }
    // Node fallback (tests)
    return new Uint8Array(Buffer.from(clean, 'base64'));
  }

  /** Float32 mono samples at `inputRate` -> 16 kHz PCM16 little-endian ArrayBuffer. */
  function downsampleTo16kPCM(input, inputRate, targetRate) {
    const target = targetRate || 16000;
    const toInt16 = (s) => {
      const v = Math.max(-1, Math.min(1, s));
      return v < 0 ? v * 0x8000 : v * 0x7fff;
    };
    if (!input || !input.length) return new ArrayBuffer(0);
    if (inputRate === target) {
      const pcm = new Int16Array(input.length);
      for (let i = 0; i < input.length; i += 1) pcm[i] = toInt16(input[i]);
      return pcm.buffer;
    }
    const ratio = inputRate / target;
    const outLength = Math.round(input.length / ratio);
    const pcm = new Int16Array(outLength);
    let offsetIn = 0;
    for (let o = 0; o < outLength; o += 1) {
      const nextIn = Math.round((o + 1) * ratio);
      let sum = 0;
      let count = 0;
      for (let i = offsetIn; i < nextIn && i < input.length; i += 1) {
        sum += input[i];
        count += 1;
      }
      pcm[o] = toInt16(count ? sum / count : 0);
      offsetIn = nextIn;
    }
    return pcm.buffer;
  }

  // ------------------------------------------------------------------
  // Microphone diagnostics
  // ------------------------------------------------------------------
  /** Returns an explanation string when the mic cannot be used here, else null. */
  function micSupportProblem(env) {
    const e = env || {};
    if (!e.isSecureContext) {
      return 'Microphone access needs a secure page. Open the app over https:// or via http://localhost — ' +
        'browsers block the microphone on plain http:// addresses such as a LAN IP (http://' + (e.host || '192.168.x.x:8000') + ').';
    }
    if (!e.hasGetUserMedia) {
      return 'This browser does not support microphone capture (navigator.mediaDevices.getUserMedia is unavailable).';
    }
    return null;
  }

  function describeMicError(err) {
    const name = err && err.name ? err.name : '';
    switch (name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
      case 'SecurityError':
        return 'Microphone permission was denied. Allow microphone access for this site in the browser settings and try again.';
      case 'NotFoundError':
      case 'DevicesNotFoundError':
      case 'OverconstrainedError':
        return 'No microphone was found. Connect a microphone and try again.';
      case 'NotReadableError':
      case 'TrackStartError':
      case 'AbortError':
        return 'The microphone is busy or could not be started (another app may be using it).';
      default:
        return 'Could not access the microphone' + (err && err.message ? ': ' + err.message : '.');
    }
  }

  return {
    escapeHtml,
    formatAnswerHtml,
    createSSEParser,
    formatErrorDetail,
    parseRetryAfter,
    describeHttpError,
    createSeqBuffer,
    buildHistory,
    toNumber,
    formatMs,
    formatScore,
    formatClock,
    truncateText,
    safeHttpUrl,
    queriesDiffer,
    isNearBottom,
    latencyRating,
    verdictFor,
    QUERY_TYPES,
    queryTypeClass,
    matchesCategory,
    normalizeSampleQueries,
    sanitizeHistoryItems,
    isLangCode,
    shuffled,
    buildWsUrl,
    createSafeStorage,
    base64ToBytes,
    downsampleTo16kPCM,
    micSupportProblem,
    describeMicError,
  };
});
