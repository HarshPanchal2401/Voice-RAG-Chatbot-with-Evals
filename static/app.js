/**
 * Multi-Language Voice & Text RAG — Client Application Logic (v3.2.0)
 * ====================================================================
 * Supports dynamic language routing for Gujarati (gu-IN) and Hindi (hi-IN).
 */

// ============================================================
// Supported Languages Configuration
// ============================================================
const LANGUAGES = {
  gu: {
    code: 'gu',
    locale: 'gu-IN',
    name: 'Gujarati',
    nativeName: 'ગુજરાતી',
    label: 'ગુજરાતી (Gujarati)',
    placeholder: 'ગુજરાતી માં પ્રશ્ન લખો... (Type your question in Gujarati or English)',
    speakPrompt: 'Speak in Gujarati',
    welcomeMsg: 'નમસ્તે! હું તમારો ગુજરાતી AI સહાયક છું. તમે મને બોલીને (Voice) અથવા લખીને (Text) કોઈ પણ પ્રશ્ન પૂછી શકો છો.',
    appTitle: 'Gujarati Voice RAG',
    appSubtitle: 'Voice + Text RAG for Gujarati',
    samplePoolFile: '/static/golden_sample_queries.json',
    thinkingText: 'વિચાર કરી રહ્યો છે...',
    errPrefix: 'ક્ષમા કરશો, પ્રશ્નનો ઉત્તર મેળવવામાં સમસ્યા આવી'
  },
  hi: {
    code: 'hi',
    locale: 'hi-IN',
    name: 'Hindi',
    nativeName: 'हिन्दी',
    label: 'हिन्दी (Hindi)',
    placeholder: 'हिंदी में प्रश्न पूछें... (Type your question in Hindi or English)',
    speakPrompt: 'Speak in Hindi',
    welcomeMsg: 'नमस्ते! मैं आपका हिंदी AI सहायक हूँ। आप मुझसे बोलकर (Voice) या लिखकर (Text) कोई भी प्रश्न पूछ सकते हैं।',
    appTitle: 'Hindi Voice RAG',
    appSubtitle: 'Voice + Text RAG for Hindi',
    samplePoolFile: '/static/golden_sample_queries_hi.json',
    thinkingText: 'विचार कर रहा हूँ...',
    errPrefix: 'क्षमा करें, उत्तर प्राप्त करने में समस्या आई'
  }
};

// ============================================================
// State Management
// ============================================================
const STATE = {
  currentLanguage: localStorage.getItem('voice_rag_lang') || 'gu',
  currentMode: 'voice', // 'voice' | 'chat'
  isRecording: false,
  audioStream: null,
  audioContext: null,
  analyser: null,
  animationFrameId: null,
  isGenerating: false,
  chatHistory: JSON.parse(localStorage.getItem('voice_rag_history') || '[]'),
};

let liveVoiceWs = null;
let scriptProcessorNode = null;
let currentGoldenQueries = [];
let cachedGoldenPools = { gu: [], hi: [] };

// ============================================================
// Initialization Logic & Health Check
// ============================================================
function initApp() {
  // Validate language code
  if (!LANGUAGES[STATE.currentLanguage]) {
    STATE.currentLanguage = 'gu';
  }

  applyLanguageUI(STATE.currentLanguage, false);
  fetchSampleQueries();
  renderHistoryList();
  initWaveformCanvas();
  checkHealthStatus();

  // Global click listener to close language dropdown if clicked outside
  document.addEventListener('click', (e) => {
    const dropdown = document.getElementById('top-lang-dropdown');
    if (dropdown && !dropdown.contains(e.target) && !e.target.closest('#voice-lang-pill')) {
      dropdown.classList.remove('open');
    }
  });

  // Add initial assistant welcome message if feed is empty
  const feed = document.getElementById('chat-feed');
  if (feed && feed.children.length === 0) {
    const langInfo = LANGUAGES[STATE.currentLanguage];
    appendBotMessage({
      answer: langInfo.welcomeMsg,
      timestamp: formatTime(new Date()),
      sources: []
    });
  }

  // Ensure evaluation toggle is off by default
  const evalToggle = document.getElementById('eval-toggle');
  if (evalToggle) {
    evalToggle.checked = false;
    handleEvalToggleChange(evalToggle);
  }

  // Ensure TTS voice reply toggle is off by default
  const ttsToggle = document.getElementById('tts-toggle');
  if (ttsToggle) {
    ttsToggle.checked = false;
    handleTtsToggleChange(ttsToggle);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

async function checkHealthStatus() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      const statusPill = document.getElementById('system-status-pill');
      if (statusPill) {
        const langsStr = (data.loaded_languages || ['GU', 'HI']).map(l => l.toUpperCase()).join(' & ');
        statusPill.innerHTML = `<span class="status-dot"></span><span class="status-text">Model Status: <strong>Online (${langsStr})</strong></span>`;
      }
    }
  } catch (err) {
    console.warn("Could not reach /health endpoint:", err);
  }
}

// ============================================================
// Language Routing & UI Localization
// ============================================================
function toggleLanguageMenu(event) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }
  const wrapper = document.getElementById('top-lang-dropdown');
  const menu = document.getElementById('lang-dropdown-menu');
  if (wrapper) wrapper.classList.toggle('open');
  if (menu) menu.classList.toggle('show');
}

function selectLanguage(langCode, event) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }
  if (!LANGUAGES[langCode]) return;

  const prevLang = STATE.currentLanguage;
  STATE.currentLanguage = langCode;
  localStorage.setItem('voice_rag_lang', langCode);

  const wrapper = document.getElementById('top-lang-dropdown');
  const menu = document.getElementById('lang-dropdown-menu');
  if (wrapper) wrapper.classList.remove('open');
  if (menu) menu.classList.remove('show');

  applyLanguageUI(langCode, true);

  if (prevLang !== langCode) {
    fetchSampleQueries();
  }
}

// Close language dropdown if user clicks anywhere outside
document.addEventListener('click', (event) => {
  const wrapper = document.getElementById('top-lang-dropdown');
  const menu = document.getElementById('lang-dropdown-menu');
  if (wrapper && !wrapper.contains(event.target)) {
    wrapper.classList.remove('open');
    if (menu) menu.classList.remove('show');
  }
});

function applyLanguageUI(langCode, updateWelcomeIfSingle = false) {
  const info = LANGUAGES[langCode] || LANGUAGES.gu;

  // 1. Update HTML lang and body classes
  document.documentElement.lang = langCode;
  document.body.classList.remove('lang-gu', 'lang-hi');
  document.body.classList.add(`lang-${langCode}`);

  // 2. Update Top Navbar Dropdown Label & Checkmarks
  const currentLangLabel = document.getElementById('current-lang-label');
  if (currentLangLabel) currentLangLabel.textContent = info.label;

  const checkGu = document.getElementById('check-gu');
  const checkHi = document.getElementById('check-hi');
  if (checkGu) checkGu.style.display = langCode === 'gu' ? 'inline' : 'none';
  if (checkHi) checkHi.style.display = langCode === 'hi' ? 'inline' : 'none';

  // Update active state on menu items
  document.querySelectorAll('.lang-menu-item').forEach(item => {
    if (item.dataset.lang === langCode) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  // 3. Update Voice Stage Pill & Speak Prompt
  const voiceStageLangName = document.getElementById('voice-stage-lang-name');
  if (voiceStageLangName) voiceStageLangName.textContent = info.label;

  const speakPromptText = document.getElementById('speak-prompt-text');
  if (speakPromptText) speakPromptText.textContent = info.speakPrompt;

  // 4. Update Textarea Placeholder
  const queryInput = document.getElementById('query-input');
  if (queryInput) {
    queryInput.placeholder = info.placeholder;
  }

  // 5. Update App Brand Title & Subtitle
  const brandAppTitle = document.getElementById('brand-app-title');
  const brandAppSubtitle = document.getElementById('brand-app-subtitle');
  if (brandAppTitle) brandAppTitle.textContent = info.appTitle;
  if (brandAppSubtitle) brandAppSubtitle.textContent = info.appSubtitle;

  // 6. Update Welcome message if only initial message is present
  if (updateWelcomeIfSingle) {
    const feed = document.getElementById('chat-feed');
    if (feed && feed.children.length === 1 && feed.querySelector('.bot-row')) {
      feed.innerHTML = '';
      appendBotMessage({
        answer: info.welcomeMsg,
        timestamp: formatTime(new Date()),
        sources: []
      });
    }
  }
}

// ============================================================
// UI Rendering: Sample Golden Queries & History
// ============================================================
async function fetchSampleQueries() {
  const btn = document.getElementById('refresh-queries-btn');
  const modalBtn = document.getElementById('modal-refresh-queries-btn');
  const countBadge = document.getElementById('sample-count');
  if (btn) btn.classList.add('spinning');
  if (modalBtn) modalBtn.classList.add('spinning');

  const lang = STATE.currentLanguage || 'gu';

  // 1. Try Backend Dynamic Endpoint for target language
  try {
    const res = await fetch(`/api/v1/sample-queries?lang=${lang}&count=20&_t=${Date.now()}`);
    if (res.ok) {
      const data = await res.json();
      if (data.queries && data.queries.length > 0) {
        currentGoldenQueries = data.queries;
        if (countBadge) countBadge.textContent = data.queries.length;
        renderSampleQueries(currentGoldenQueries);
        applyModalFilters();
        if (btn) btn.classList.remove('spinning');
        if (modalBtn) modalBtn.classList.remove('spinning');
        return;
      }
    }
  } catch (err) {
    console.warn(`Backend sample queries for ${lang} not available, reading static pool:`, err);
  }

  // 2. Fallback to loaded Golden Dataset static pool
  try {
    const poolFile = LANGUAGES[lang]?.samplePoolFile || '/static/golden_sample_queries.json';
    if (!cachedGoldenPools[lang] || cachedGoldenPools[lang].length === 0) {
      const resPool = await fetch(`${poolFile}?_t=${Date.now()}`);
      if (resPool.ok) {
        cachedGoldenPools[lang] = await resPool.json();
      }
    }

    const pool = cachedGoldenPools[lang];
    if (pool && pool.length > 0) {
      const shuffled = [...pool].sort(() => 0.5 - Math.random());
      currentGoldenQueries = shuffled.slice(0, 20);
      if (countBadge) countBadge.textContent = currentGoldenQueries.length;
      renderSampleQueries(currentGoldenQueries);
      applyModalFilters();
    }
  } catch (err2) {
    console.error("Could not load golden dataset queries fallback:", err2);
  } finally {
    if (btn) btn.classList.remove('spinning');
    if (modalBtn) modalBtn.classList.remove('spinning');
  }
}

async function refreshModalSampleQueries() {
  const modalBtn = document.getElementById('modal-refresh-queries-btn');
  const sidebarBtn = document.getElementById('refresh-queries-btn');
  if (modalBtn) modalBtn.classList.add('spinning');
  if (sidebarBtn) sidebarBtn.classList.add('spinning');

  try {
    await fetchSampleQueries();
  } finally {
    if (modalBtn) modalBtn.classList.remove('spinning');
    if (sidebarBtn) sidebarBtn.classList.remove('spinning');
  }
}

function renderSampleQueries(queries = null) {
  const container = document.getElementById('sample-queries-list');
  if (!container) return;
  
  const list = queries || currentGoldenQueries;
  if (!list || list.length === 0) return;

  container.innerHTML = '';
  // Show top 3 preview items in the sidebar to keep it compact and clean
  list.slice(0, 3).forEach((itemObj, idx) => {
    const qText = typeof itemObj === 'string' ? itemObj : itemObj.question;
    const qType = itemObj.query_type ? `<span class="sample-tag">${escapeHtml(itemObj.query_type)}</span>` : '';
    
    const item = document.createElement('div');
    item.className = 'sample-query-item';
    item.title = "Click to load into input box";
    item.innerHTML = `
      <span class="sample-text">${escapeHtml(qText)}</span>
      ${qType}
    `;
    item.onclick = () => {
      useQueryInInput(qText);
    };
    container.appendChild(item);
  });
}

function fillRandomSampleQuery() {
  const lang = STATE.currentLanguage || 'gu';
  let pool = currentGoldenQueries;
  if (!pool || pool.length === 0) {
    pool = cachedGoldenPools[lang] || [];
  }
  if (!pool || pool.length === 0) return;

  const randItem = pool[Math.floor(Math.random() * pool.length)];
  const qText = typeof randItem === 'string' ? randItem : randItem.question;
  useQueryInInput(qText);
}

function useQueryInInput(qText, autoSubmit = false) {
  const input = document.getElementById('query-input');
  if (!input) return;
  input.value = qText;
  handleInputResize(input);
  input.focus();

  // Subtle visual feedback pulse on dock container
  const dockContainer = document.querySelector('.dock-container');
  if (dockContainer) {
    dockContainer.classList.remove('dock-loaded-pulse');
    void dockContainer.offsetWidth; // Trigger reflow
    dockContainer.classList.add('dock-loaded-pulse');
    setTimeout(() => dockContainer.classList.remove('dock-loaded-pulse'), 800);
  }

  // Auto-scroll chat feed down if in chat mode
  const feed = document.getElementById('chat-feed');
  if (feed) {
    feed.scrollTop = feed.scrollHeight;
  }

  if (autoSubmit) {
    submitCurrentQuery();
  }
}

function openAllQueriesModal() {
  const modal = document.getElementById('all-queries-modal');
  if (!modal) return;
  modal.style.display = 'flex';
  renderModalAllQueries(currentGoldenQueries);

  const searchInput = document.getElementById('query-modal-search');
  if (searchInput) {
    searchInput.value = '';
    setTimeout(() => searchInput.focus(), 50);
  }
}

function closeAllQueriesModal(event) {
  if (event && event.target && event.target.id !== 'all-queries-modal' && !event.target.classList.contains('modal-close-btn')) {
    return;
  }
  const modal = document.getElementById('all-queries-modal');
  if (modal) modal.style.display = 'none';
}

let currentModalFilterCat = 'all';

function getTypeBadgeClass(qType) {
  const t = (qType || '').toUpperCase();
  if (t.includes('DESC')) return 'tag-desc';
  if (t.includes('ENT')) return 'tag-entity';
  if (t.includes('FACT')) return 'tag-fact';
  if (t.includes('NUM')) return 'tag-numeric';
  if (t.includes('LOC')) return 'tag-location';
  return 'tag-default';
}

function handleModalSearchInput() {
  applyModalFilters();
}

function filterModalByCategory(cat) {
  currentModalFilterCat = cat;
  document.querySelectorAll('.cat-filter-btn').forEach(btn => {
    if (btn.dataset.cat === cat) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  applyModalFilters();
}

function applyModalFilters() {
  const searchInput = document.getElementById('query-modal-search');
  const term = searchInput ? searchInput.value.toLowerCase().trim() : '';

  const filtered = currentGoldenQueries.filter(itemObj => {
    const qText = (typeof itemObj === 'string' ? itemObj : itemObj.question) || '';
    const qType = ((typeof itemObj === 'object' ? itemObj.query_type : '') || '').toLowerCase();

    // 1. Check category filter
    if (currentModalFilterCat !== 'all') {
      if (!qType.includes(currentModalFilterCat)) return false;
    }

    // 2. Check search text
    if (term) {
      return qText.toLowerCase().includes(term) || qType.includes(term);
    }
    return true;
  });

  renderModalAllQueries(filtered);
}

function renderModalAllQueries(queriesToRender) {
  const container = document.getElementById('modal-all-queries-list');
  const countBadge = document.getElementById('modal-queries-count');
  if (!container) return;

  const list = queriesToRender || currentGoldenQueries;
  if (countBadge) countBadge.textContent = list.length;

  if (!list || list.length === 0) {
    container.innerHTML = '<div class="empty-state-text" style="padding: 32px; text-align: center; grid-column: 1 / -1;">No questions match your filter.</div>';
    return;
  }

  container.innerHTML = '';
  list.forEach((itemObj, idx) => {
    const qText = typeof itemObj === 'string' ? itemObj : itemObj.question;
    const qType = itemObj.query_type || 'QUESTION';
    const badgeClass = getTypeBadgeClass(qType);

    const card = document.createElement('div');
    card.className = 'modal-query-card';
    card.title = "Click anywhere to fill this prompt into input box";
    card.innerHTML = `
      <div class="query-card-header">
        <span class="query-type-tag ${badgeClass}">${escapeHtml(qType)}</span>
        <span class="query-index">#${idx + 1}</span>
      </div>
      <div class="query-card-text">${escapeHtml(qText)}</div>
      <div class="query-card-footer">
        <button class="btn-use-query" title="Fill into input box">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
          </svg>
          <span>Fill Input</span>
        </button>
        <button class="btn-ask-query" title="Submit question immediately">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
          <span>Ask</span>
        </button>
      </div>
    `;

    // Click on card body -> fill query in input box & close modal
    card.onclick = (e) => {
      if (e.target.closest('.btn-ask-query')) return;
      useQueryInInput(qText, false);
      const modal = document.getElementById('all-queries-modal');
      if (modal) modal.style.display = 'none';
    };

    // "Ask" button -> submit immediately
    const askBtn = card.querySelector('.btn-ask-query');
    if (askBtn) {
      askBtn.onclick = (e) => {
        e.stopPropagation();
        const modal = document.getElementById('all-queries-modal');
        if (modal) modal.style.display = 'none';
        useQueryInInput(qText, true);
      };
    }

    container.appendChild(card);
  });
}

function filterModalQueries(keyword) {
  handleModalSearchInput();
}

function renderHistoryList() {
  const container = document.getElementById('history-list');
  if (!container) return;

  if (STATE.chatHistory.length === 0) {
    container.innerHTML = '<p class="empty-state-text">No queries asked yet.</p>';
    return;
  }

  container.innerHTML = '';
  STATE.chatHistory.slice(-6).reverse().forEach(item => {
    const el = document.createElement('div');
    el.className = 'history-item';
    const langBadge = item.lang ? `<span style="font-size:0.65rem; background:rgba(37,99,235,0.1); color:#2563eb; padding:1px 5px; border-radius:4px; margin-left:4px;">${item.lang.toUpperCase()}</span>` : '';
    el.innerHTML = `
      <div class="history-item-top">
        <span>${item.mode === 'voice' ? '🎙️ Voice' : '💬 Text'}${langBadge}</span>
        <span>${item.time}</span>
      </div>
      <div class="history-item-text">${escapeHtml(item.query)}</div>
    `;
    el.onclick = () => {
      const input = document.getElementById('query-input');
      if (input) {
        input.value = item.query;
        handleInputResize(input);
      }
      if (item.lang && LANGUAGES[item.lang] && item.lang !== STATE.currentLanguage) {
        selectLanguage(item.lang);
      }
    };
    container.appendChild(el);
  });
}

function saveHistoryItem(query, mode) {
  STATE.chatHistory.push({
    query,
    mode,
    lang: STATE.currentLanguage,
    time: formatTime(new Date())
  });
  if (STATE.chatHistory.length > 50) {
    STATE.chatHistory.shift();
  }
  localStorage.setItem('voice_rag_history', JSON.stringify(STATE.chatHistory));
  renderHistoryList();
}

function clearHistory() {
  if (confirm("Are you sure you want to clear chat history?")) {
    STATE.chatHistory = [];
    localStorage.removeItem('voice_rag_history');
    renderHistoryList();
  }
}

function openHistoryModal() {
  if (STATE.chatHistory.length === 0) {
    alert("No history records yet.");
    return;
  }
  const allTexts = STATE.chatHistory.map((h, i) => `${i + 1}. [${h.time} | ${(h.lang || 'gu').toUpperCase()}] ${h.query}`).join('\n');
  alert(`Asked Queries History:\n\n${allTexts}`);
}

// ============================================================
// Mode Switcher (Voice vs Chat)
// ============================================================
function switchMode(mode) {
  STATE.currentMode = mode;
  const btnVoice = document.getElementById('btn-mode-voice');
  const btnChat = document.getElementById('btn-mode-chat');
  const voiceStage = document.getElementById('voice-stage');

  if (mode === 'voice') {
    if (btnVoice) btnVoice.classList.add('active');
    if (btnChat) btnChat.classList.remove('active');
    if (voiceStage) voiceStage.style.display = 'flex';
  } else {
    if (btnChat) btnChat.classList.add('active');
    if (btnVoice) btnVoice.classList.remove('active');
    if (voiceStage) voiceStage.style.display = 'none';
  }
}

// ============================================================
// Text Query Submission Flow
// ============================================================
function handleInputResize(textarea) {
  if (!textarea) return;
  textarea.style.height = 'auto';
  textarea.style.height = Math.max(42, Math.min(textarea.scrollHeight, 150)) + 'px';
  const counter = document.getElementById('char-counter');
  if (counter) {
    counter.textContent = `${textarea.value.length} / 1000`;
  }
}

function handleInputKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitCurrentQuery();
  }
}

function handleEvalToggleChange(checkbox) {
  if (!checkbox) return;
  const textEl = document.getElementById('eval-toggle-text');
  const labelEl = document.getElementById('eval-toggle-label');
  if (textEl) {
    textEl.textContent = checkbox.checked ? 'Eval: ON' : 'Eval: OFF';
  }
  if (labelEl) {
    if (checkbox.checked) {
      labelEl.classList.add('active');
      labelEl.title = "Evaluation is ENABLED — DeepEval quality metrics will run on each query";
    } else {
      labelEl.classList.remove('active');
      labelEl.title = "Evaluation is DISABLED — Responses will generate at maximum speed";
    }
  }
}

function handleTtsToggleChange(checkbox) {
  if (!checkbox) return;
  const textEl = document.getElementById('tts-toggle-text');
  const labelEl = document.getElementById('tts-toggle-label');
  if (textEl) {
    textEl.textContent = checkbox.checked ? 'Voice Reply: ON' : 'Voice Reply: OFF';
  }
  if (labelEl) {
    if (checkbox.checked) {
      labelEl.classList.add('active');
      labelEl.title = "Voice Speech Reply is ENABLED — Sarvam Bulbul v3 will speak the response";
    } else {
      labelEl.classList.remove('active');
      labelEl.title = "Voice Speech Reply is DISABLED — Text response only";
    }
  }
}

let currentPlayingAudio = null;

function playAudioBase64(b64Data) {
  if (!b64Data) return;
  try {
    if (currentPlayingAudio) {
      currentPlayingAudio.pause();
      currentPlayingAudio = null;
    }
    const audio = new Audio("data:audio/mp3;base64," + b64Data);
    currentPlayingAudio = audio;
    audio.play().catch(err => {
      console.warn("Audio autoplay blocked by browser policy:", err);
    });
  } catch (err) {
    console.error("Audio playback error:", err);
  }
}

async function synthesizeAndPlayMessage(btn) {
  const text = btn.getAttribute('data-text');
  if (!text) return;
  const originalHtml = btn.innerHTML;
  btn.innerHTML = '⏳ Speaking...';
  btn.disabled = true;
  try {
    const res = await fetch('/api/v1/voice/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        language: STATE.currentLanguage || 'gu',
        speaker: 'shubh'
      })
    });
    if (!res.ok) throw new Error("TTS failed");
    const data = await res.json();
    if (data.audio_base64) {
      playAudioBase64(data.audio_base64);
      btn.innerHTML = '🔊 Replay';
      btn.onclick = () => playAudioBase64(data.audio_base64);
    }
  } catch (err) {
    console.error("On-demand TTS error:", err);
    btn.innerHTML = '⚠️ Voice Error';
  } finally {
    btn.disabled = false;
  }
}

function submitCurrentQuery() {
  const input = document.getElementById('query-input');
  if (!input) return;
  const query = input.value.trim();
  if (!query || STATE.isGenerating) return;
  input.value = '';
  handleInputResize(input);
  submitTextQuery(query, 'text');
}

async function submitTextQuery(query, mode = 'text', sttLatencyMs = null) {
  if (STATE.isGenerating) return;
  STATE.isGenerating = true;
  setControlsDisabled(true);

  const evalToggle = document.getElementById('eval-toggle');
  const evaluate = evalToggle ? evalToggle.checked : false;
  const ttsToggle = document.getElementById('tts-toggle');
  const voiceReply = mode === 'voice' ? true : (ttsToggle ? ttsToggle.checked : false);
  const timeStr = formatTime(new Date());
  const activeLang = STATE.currentLanguage || 'gu';
  const langInfo = LANGUAGES[activeLang] || LANGUAGES.gu;

  // 1. Append User Bubble
  appendUserMessage(query, timeStr, mode === 'voice');
  saveHistoryItem(query, mode);

  // 2. Append Bot Bubble with Typing Loader
  const botBubble = appendBotLoadingMessage(timeStr, langInfo.thinkingText);
  const feed = document.getElementById('chat-feed');

  let streamedAnswer = '';
  let lastAnswerMsg = null;
  let receivedAudioB64 = null;

  try {
    const response = await fetch('/api/v1/query/text/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        language: activeLang,
        top_k: 5,
        evaluate: evaluate,
        voice_reply: voiceReply
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const block of parts) {
        if (!block.trim()) continue;
        const lines = block.split('\n');
        let eventType = 'message';
        let dataStr = '';

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            dataStr += (dataStr ? '\n' : '') + line.slice(5).trim();
          }
        }

        if (!dataStr) continue;

        try {
          const data = JSON.parse(dataStr);

          // ⚡ Real-Time Streamed Token
          if (eventType === 'token') {
            const token = data.token || '';
            streamedAnswer += token;
            const body = botBubble ? botBubble.querySelector('.bot-body-text') : null;
            if (body) {
              body.textContent = streamedAnswer;
              if (feed) feed.scrollTop = feed.scrollHeight;
            }
          }
          // 🤖 Full Answer Meta & References
          else if (eventType === 'meta') {
            lastAnswerMsg = data;
            if (sttLatencyMs && lastAnswerMsg.latency) {
              lastAnswerMsg.latency.stt_ms = sttLatencyMs;
            }
            updateBotMessage(
              botBubble,
              data.answer || streamedAnswer,
              data.sources || [],
              timeStr,
              data.evaluation || null,
              receivedAudioB64
            );
            updateLatencySidebar(lastAnswerMsg.latency);
          }
          // 🔊 TTS Voice Audio (Sarvam Bulbul v3)
          else if (eventType === 'tts') {
            if (data.audio_base64) {
              receivedAudioB64 = data.audio_base64;
              playAudioBase64(data.audio_base64);
              if (botBubble && lastAnswerMsg) {
                updateBotMessage(
                  botBubble,
                  lastAnswerMsg.answer || streamedAnswer,
                  lastAnswerMsg.sources || [],
                  timeStr,
                  lastAnswerMsg.evaluation || null,
                  receivedAudioB64
                );
              }
            }
          }
          // 🎯 DeepEval Evaluation Scorecard
          else if (eventType === 'evaluation') {
            if (lastAnswerMsg) {
              lastAnswerMsg.evaluation = data.evaluation;
              if (lastAnswerMsg.latency && data.eval_ms) {
                lastAnswerMsg.latency.eval_ms = data.eval_ms;
                updateLatencySidebar(lastAnswerMsg.latency);
              }
            }
            updateBotMessage(
              botBubble,
              lastAnswerMsg ? lastAnswerMsg.answer : streamedAnswer,
              lastAnswerMsg ? lastAnswerMsg.sources : [],
              timeStr,
              data.evaluation,
              receivedAudioB64
            );
            updateEvaluationSidebar(data.evaluation);
          }
          // ⚠️ Server Error Event
          else if (eventType === 'error') {
            throw new Error(data.error || 'Streaming error');
          }
        } catch (jsonErr) {
          console.warn("SSE JSON chunk parse error:", jsonErr, dataStr);
        }
      }
    }

  } catch (err) {
    console.error("Query failed:", err);
    updateBotMessage(botBubble, `⚠️ ${langInfo.errPrefix}: ${err.message}`, [], timeStr);
  } finally {
    STATE.isGenerating = false;
    setControlsDisabled(false);
  }
}

// ============================================================
// Sarvam AI Live Streaming Speech-to-Text via WebSocket
// ============================================================

async function toggleVoiceRecording() {
  if (STATE.isRecording) {
    stopVoiceRecording();
  } else {
    startVoiceRecording();
  }
}

async function startVoiceRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    STATE.audioStream = stream;

    const activeLang = STATE.currentLanguage || 'gu';
    const langInfo = LANGUAGES[activeLang] || LANGUAGES.gu;

    // 1. AudioContext setup
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    STATE.audioContext = new AudioContext();
    const source = STATE.audioContext.createMediaStreamSource(stream);
    
    // Setup Analyser for visualizer
    STATE.analyser = STATE.audioContext.createAnalyser();
    STATE.analyser.fftSize = 64;
    source.connect(STATE.analyser);

    // 2. Connect to FastAPI WebSocket with selected language
    const wsProto = location.protocol === 'https:' ? 'wss://' : 'ws://';
    const evalToggleLive = document.getElementById('eval-toggle');
    const evaluateLive = evalToggleLive ? evalToggleLive.checked : false;
    const wsUrl = `${wsProto}${location.host}/api/v1/voice/live?lang=${activeLang}&evaluate=${evaluateLive}`;
    liveVoiceWs = new WebSocket(wsUrl);
    liveVoiceWs.binaryType = 'arraybuffer';

    // Show transcript card
    const transcriptCard = document.getElementById('voice-transcript-card');
    const transcriptBody = document.getElementById('transcript-text');
    const transcriptTime = document.getElementById('transcript-timestamp');
    if (transcriptCard) transcriptCard.style.display = 'block';
    if (transcriptTime) transcriptTime.textContent = formatTime(new Date());
    if (transcriptBody) transcriptBody.textContent = `🎙️ ${langInfo.speakPrompt}...`;

    let currentBotBubble = null;
    let streamedAnswer = '';
    let lastAnswerMsg = null;
    const timeStr = formatTime(new Date());

    liveVoiceWs.onopen = () => {
      STATE.isRecording = true;
      const micBtn = document.getElementById('main-mic-btn');
      const listInd = document.getElementById('listening-indicator');
      const spkPill = document.getElementById('speak-prompt-pill');

      if (micBtn) micBtn.classList.add('recording');
      if (listInd) listInd.style.display = 'flex';
      if (spkPill) spkPill.style.display = 'none';

      // Setup ScriptProcessor to extract 16kHz PCM audio
      const bufferSize = 4096;
      scriptProcessorNode = STATE.audioContext.createScriptProcessor(bufferSize, 1, 1);
      
      scriptProcessorNode.onaudioprocess = (e) => {
        if (!STATE.isRecording || !liveVoiceWs || liveVoiceWs.readyState !== WebSocket.OPEN) return;
        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = downsampleTo16kPCM(inputData, STATE.audioContext.sampleRate);
        if (pcm16 && pcm16.byteLength > 0) {
          liveVoiceWs.send(pcm16);
        }
      };

      source.connect(scriptProcessorNode);
      scriptProcessorNode.connect(STATE.audioContext.destination);

      drawLiveWaveform();
    };

    liveVoiceWs.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        
        // 🎙️ LIVE PARTIAL TRANSCRIPTION
        if (msg.type === 'partial') {
          if (msg.text && msg.text.trim()) {
            if (transcriptBody) transcriptBody.textContent = msg.text.trim();
            const input = document.getElementById('query-input');
            if (input) {
              input.value = msg.text.trim();
              handleInputResize(input);
            }
          }
        } 
        // ✅ FINAL TRANSCRIPTION
        else if (msg.type === 'final') {
          const finalQ = msg.text.trim();
          if (transcriptBody) transcriptBody.textContent = `✅ "${finalQ}"`;
          const input = document.getElementById('query-input');
          if (input) {
            input.value = '';
            handleInputResize(input);
          }
          
          appendUserMessage(finalQ, timeStr, true);
          saveHistoryItem(finalQ, 'voice');
          currentBotBubble = appendBotLoadingMessage(timeStr, langInfo.thinkingText);
        }
        // ⚡ STREAMED ANSWER TOKENS
        else if (msg.type === 'token') {
          streamedAnswer += msg.content || '';
          const body = currentBotBubble ? currentBotBubble.querySelector('.bot-body-text') : null;
          if (body) {
            body.textContent = streamedAnswer;
            const feed = document.getElementById('chat-feed');
            if (feed) feed.scrollTop = feed.scrollHeight;
          }
        }
        // 🎯 DEEPEVAL SCORECARD (arrives after the answer, never delays it)
        else if (msg.type === 'evaluation') {
          if (lastAnswerMsg && currentBotBubble) {
            updateBotMessage(currentBotBubble, lastAnswerMsg.answer, lastAnswerMsg.sources, timeStr, msg.evaluation);
          }
          if (lastAnswerMsg && lastAnswerMsg.latency) {
            lastAnswerMsg.latency.eval_ms = msg.eval_ms;
            updateLatencySidebar(lastAnswerMsg.latency);
          }
          updateEvaluationSidebar(msg.evaluation);
        }
        // 🔊 TTS VOICE AUDIO (Sarvam Bulbul v3)
        else if (msg.type === 'tts' && msg.audio_base64) {
          playAudioBase64(msg.audio_base64);
          if (currentBotBubble) {
            updateBotMessage(currentBotBubble, lastAnswerMsg ? lastAnswerMsg.answer : streamedAnswer, lastAnswerMsg ? lastAnswerMsg.sources : [], timeStr, lastAnswerMsg ? lastAnswerMsg.evaluation : null, msg.audio_base64);
          }
        }
        // 🤖 GENERATED RAG ANSWER & SOURCES
        else if (msg.type === 'answer') {
          lastAnswerMsg = msg;
          if (currentBotBubble) {
            updateBotMessage(currentBotBubble, msg.answer, msg.sources, timeStr, msg.evaluation);
          } else {
            appendBotMessage({
              answer: msg.answer,
              sources: msg.sources,
              timestamp: timeStr,
              evaluation: msg.evaluation
            });
          }
          updateLatencySidebar(msg.latency);
          updateEvaluationSidebar(msg.evaluation);
        }
        else if (msg.type === 'error') {
          console.error("Sarvam AI live error:", msg.message);
          if (transcriptBody) transcriptBody.textContent = `❌ Error: ${msg.message}`;
          if (currentBotBubble) {
            updateBotMessage(currentBotBubble, `⚠️ ${msg.message || 'Voice error occurred'}`, [], timeStr);
          }
        }
      } catch (err) {
        console.error("WebSocket message parse error:", err);
      }
    };

    liveVoiceWs.onerror = (err) => {
      console.error("Live Voice WebSocket error:", err);
      stopVoiceRecording();
    };

    liveVoiceWs.onclose = () => {
      stopVoiceRecording();
    };

  } catch (err) {
    console.error("Microphone capture error:", err);
    alert("Could not access microphone. Please check permissions in your browser.");
  }
}

function stopVoiceRecording() {
  if (!STATE.isRecording && !liveVoiceWs) return;
  STATE.isRecording = false;

  if (liveVoiceWs && liveVoiceWs.readyState === WebSocket.OPEN) {
    try {
      liveVoiceWs.send(JSON.stringify({ type: "stop" }));
    } catch (e) {}
  }

  if (scriptProcessorNode) {
    try { scriptProcessorNode.disconnect(); } catch (e) {}
    scriptProcessorNode = null;
  }

  if (STATE.audioStream) {
    STATE.audioStream.getTracks().forEach(track => track.stop());
    STATE.audioStream = null;
  }

  if (STATE.audioContext && STATE.audioContext.state !== 'closed') {
    try { STATE.audioContext.close(); } catch (e) {}
    STATE.audioContext = null;
  }

  if (STATE.animationFrameId) {
    cancelAnimationFrame(STATE.animationFrameId);
    STATE.animationFrameId = null;
  }

  const micBtn = document.getElementById('main-mic-btn');
  const listInd = document.getElementById('listening-indicator');
  const spkPill = document.getElementById('speak-prompt-pill');

  if (micBtn) micBtn.classList.remove('recording');
  if (listInd) listInd.style.display = 'none';
  if (spkPill) spkPill.style.display = 'flex';

  initWaveformCanvas();
}

function downsampleTo16kPCM(buffer, inputSampleRate, targetSampleRate = 16000) {
  if (inputSampleRate === targetSampleRate) {
    const pcm = new Int16Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
      let s = Math.max(-1, Math.min(1, buffer[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return pcm.buffer;
  }

  const ratio = inputSampleRate / targetSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const pcm = new Int16Array(newLength);
  
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < pcm.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    let s = count > 0 ? Math.max(-1, Math.min(1, accum / count)) : 0;
    pcm[offsetResult] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return pcm.buffer;
}

function triggerAudioUpload() {
  const input = document.getElementById('audio-file-input');
  if (input) input.click();
}

async function handleAudioFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = '';

  const evalToggle = document.getElementById('eval-toggle');
  const evaluate = evalToggle ? evalToggle.checked : false;
  const timeStr = formatTime(new Date());
  const activeLang = STATE.currentLanguage || 'gu';
  const langInfo = LANGUAGES[activeLang] || LANGUAGES.gu;

  const transcriptCard = document.getElementById('voice-transcript-card');
  const transcriptBody = document.getElementById('transcript-text');
  const transcriptTime = document.getElementById('transcript-timestamp');
  if (transcriptCard) transcriptCard.style.display = 'block';
  if (transcriptTime) transcriptTime.textContent = timeStr;
  if (transcriptBody) transcriptBody.textContent = 'Transcribing audio... ⏳';

  const botBubble = appendBotLoadingMessage(timeStr, langInfo.thinkingText);

  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', activeLang);
    formData.append('top_k', '5');
    formData.append('evaluate', evaluate ? 'true' : 'false');
    formData.append('voice_reply', 'true');

    const res = await fetch('/api/v1/query/voice', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server status ${res.status}`);
    }

    const data = await res.json();

    if (transcriptBody) transcriptBody.textContent = data.transcription || data.query;
    appendUserMessage(data.transcription || data.query, timeStr, true);
    saveHistoryItem(data.transcription || data.query, 'voice');

    if (data.audio_base64) {
      playAudioBase64(data.audio_base64);
    }

    updateBotMessage(botBubble, data.answer, data.sources, timeStr, data.evaluation, data.audio_base64);
    updateLatencySidebar(data.latency);
    updateEvaluationSidebar(data.evaluation);

  } catch (err) {
    console.error("Audio upload failed:", err);
    if (transcriptBody) transcriptBody.textContent = `❌ Transcription failed: ${err.message}`;
    updateBotMessage(botBubble, `⚠️ ${langInfo.errPrefix}: ${err.message}`, [], timeStr);
  } finally {
    STATE.isGenerating = false;
    setControlsDisabled(false);
  }
}

// ============================================================
// Waveform Canvas Visualization
// ============================================================
function initWaveformCanvas() {
  const canvas = document.getElementById('waveform-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
  grad.addColorStop(0.5, 'rgba(6, 182, 212, 0.7)');
  grad.addColorStop(1, 'rgba(139, 92, 246, 0.4)');
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  ctx.beginPath();

  const numBars = 45;
  const barWidth = w / numBars;

  for (let i = 0; i < numBars; i++) {
    const x = i * barWidth + barWidth / 2;
    const heightFactor = Math.sin((i / numBars) * Math.PI) * 0.7 + 0.15;
    const barH = 18 * heightFactor;
    ctx.moveTo(x, h / 2 - barH / 2);
    ctx.lineTo(x, h / 2 + barH / 2);
  }
  ctx.stroke();
}

function drawLiveWaveform() {
  if (!STATE.isRecording || !STATE.analyser) return;

  const canvas = document.getElementById('waveform-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  const bufferLength = STATE.analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  STATE.analyser.getByteFrequencyData(dataArray);

  ctx.clearRect(0, 0, w, h);

  const numBars = 45;
  const barWidth = w / numBars;

  for (let i = 0; i < numBars; i++) {
    const dataIdx = Math.floor((i / numBars) * bufferLength);
    const value = dataArray[dataIdx] || 0;
    const barH = Math.max(4, (value / 255) * h * 0.9);
    const x = i * barWidth + barWidth / 2;

    const grad = ctx.createLinearGradient(0, h / 2 - barH / 2, 0, h / 2 + barH / 2);
    grad.addColorStop(0, '#06b6d4');
    grad.addColorStop(0.5, '#6366f1');
    grad.addColorStop(1, '#ec4899');

    ctx.strokeStyle = grad;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    ctx.beginPath();
    ctx.moveTo(x, h / 2 - barH / 2);
    ctx.lineTo(x, h / 2 + barH / 2);
    ctx.stroke();
  }

  STATE.animationFrameId = requestAnimationFrame(drawLiveWaveform);
}

// ============================================================
// Chat Bubble Rendering Helpers
// ============================================================
function appendUserMessage(text, timeStr, isVoice = false) {
  const feed = document.getElementById('chat-feed');
  if (!feed) return;
  const row = document.createElement('div');
  row.className = 'message-row user-row';
  row.innerHTML = `
    <div class="message-avatar">👤</div>
    <div class="message-bubble">
      <div class="message-header">
        <span class="message-sender">${isVoice ? '🎙️ You' : 'You'}</span>
        <span class="message-time">${timeStr}</span>
      </div>
      <div class="message-text">${escapeHtml(text)}</div>
    </div>
  `;
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
}

function appendBotLoadingMessage(timeStr, thinkingText = 'Thinking... ⏳') {
  const feed = document.getElementById('chat-feed');
  if (!feed) return null;
  const row = document.createElement('div');
  row.className = 'message-row bot-row';
  row.innerHTML = `
    <div class="message-avatar">✦</div>
    <div class="message-bubble">
      <div class="message-header">
        <span class="message-sender">Answer</span>
        <span class="message-time">${timeStr}</span>
      </div>
      <div class="message-text bot-body-text">
        <span class="loading-dots">${thinkingText}</span>
      </div>
    </div>
  `;
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
  return row;
}

function updateBotMessage(botRow, answerText, sources, timeStr, evaluation = null, audioBase64 = null) {
  if (!botRow) return;
  const bubble = botRow.querySelector('.message-bubble');
  if (!bubble) return;
  const formattedAnswer = formatAnswerMarkdown(answerText);

  let evalBadgeHtml = '';
  if (evaluation && evaluation.scores && evaluation.scores.overall_score !== null && evaluation.scores.overall_score !== undefined) {
    const scoreVal = evaluation.scores.overall_score;
    const scorePercent = (scoreVal * 100).toFixed(0);
    const scoreClass = scoreVal >= 0.75 ? '' : (scoreVal >= 0.5 ? 'score-medium' : 'score-low');
    const reasonTitle = evaluation.scores.reason || evaluation.scores.explanation || 'DeepEval Score';
    evalBadgeHtml = `
      <span class="eval-chip-inline ${scoreClass}" title="${escapeHtml(reasonTitle)}">
        🎯 DeepEval: ${scorePercent}%
      </span>
    `;
  }

  let audioButtonHtml = '';
  if (audioBase64) {
    audioButtonHtml = `
      <div>
        <button class="audio-play-pill" onclick="playAudioBase64('${audioBase64}')" title="Play Voice Audio (Sarvam Bulbul v3)">
          🔊 Speak Answer
        </button>
      </div>
    `;
  } else {
    audioButtonHtml = `
      <div>
        <button class="audio-play-pill" onclick="synthesizeAndPlayMessage(this)" data-text="${escapeHtml(answerText)}" title="Generate & Listen in Voice (Bulbul v3)">
          🔊 Listen
        </button>
      </div>
    `;
  }

  let sourcesHtml = '';
  if (sources && sources.length > 0) {
    const listHtml = sources.map((s, idx) => `
      <div class="source-item">
        <strong>[${idx + 1}]</strong> ${escapeHtml(s.text.substring(0, 140))}... 
        <span style="color: #64748b; font-size: 0.7rem;">(RRF: ${s.rrf_score})</span>
      </div>
    `).join('');

    sourcesHtml = `
      <button class="sources-toggle-btn" onclick="toggleSourcesDrawer(this)">
        📑 References (${sources.length}) ▾
      </button>
      <div class="sources-drawer" style="display: none;">
        ${listHtml}
      </div>
    `;
  }

  bubble.innerHTML = `
    <div class="message-header">
      <span class="message-sender">Answer</span>
      <span class="message-time">${timeStr}</span>
      ${evalBadgeHtml}
    </div>
    <div class="message-text">${formattedAnswer}</div>
    ${audioButtonHtml}
    ${sourcesHtml}
    <div class="message-actions">
      <button class="msg-action-btn" title="Like response">👍</button>
      <button class="msg-action-btn" title="Dislike response">👎</button>
      <button class="msg-action-btn" title="Copy answer" onclick="copyTextToClipboard('${escapeJs(answerText)}')">📋</button>
      <button class="msg-action-btn" title="Speak answer aloud" onclick="speakText('${escapeJs(answerText)}')">🔊</button>
    </div>
  `;

  const feed = document.getElementById('chat-feed');
  if (feed) feed.scrollTop = feed.scrollHeight;
}

function appendBotMessage(data) {
  const feed = document.getElementById('chat-feed');
  if (!feed) return;
  const row = document.createElement('div');
  row.className = 'message-row bot-row';
  feed.appendChild(row);
  updateBotMessage(row, data.answer, data.sources, data.timestamp || formatTime(new Date()), data.evaluation);
}

function toggleSourcesDrawer(btn) {
  const drawer = btn.nextElementSibling;
  if (drawer.style.display === 'none') {
    drawer.style.display = 'block';
    btn.innerHTML = btn.innerHTML.replace('▾', '▴');
  } else {
    drawer.style.display = 'none';
    btn.innerHTML = btn.innerHTML.replace('▴', '▾');
  }
}

// ============================================================
// Latency Profiler Sidebar Updates
// ============================================================
function updateLatencySidebar(latency) {
  if (!latency) return;

  const pureResponseMs = latency.total_ms || 0;
  const totalSec = (pureResponseMs / 1000).toFixed(2);
  const totalValEl = document.getElementById('latency-total-val');
  if (totalValEl) totalValEl.textContent = `${totalSec} sec`;

  // Needle on 0-4s gauge bar
  const pct = Math.min(100, Math.max(0, (pureResponseMs / 4000) * 100));
  const needleEl = document.getElementById('gauge-needle');
  if (needleEl) needleEl.style.left = `${pct}%`;

  const ratingTag = document.getElementById('latency-rating-tag');
  const ratingText = document.getElementById('latency-rating-text');
  
  if (ratingTag && ratingText) {
    if (pureResponseMs < 1200) {
      ratingTag.className = 'latency-status-tag good';
      ratingText.textContent = 'Fast & Optimal 🟢';
    } else if (pureResponseMs < 2500) {
      ratingTag.className = 'latency-status-tag medium';
      ratingText.textContent = 'Good Latency 🟡';
    } else {
      ratingTag.className = 'latency-status-tag slow';
      ratingText.textContent = 'High Load 🔴';
    }
  }

  // Sub-timings
  const ret = latency.retrieval || {};
  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setEl('time-encoding', `${(ret.query_encoding_ms || 0).toFixed(1)} ms`);
  setEl('time-dense', `${(ret.dense_search_ms || 0).toFixed(1)} ms`);
  setEl('time-sparse', `${(ret.sparse_search_ms || 0).toFixed(1)} ms`);
  setEl('time-metadata', `${(ret.metadata_ms || 0).toFixed(1)} ms`);
  setEl('time-ttft', latency.ttft_ms ? `${latency.ttft_ms.toFixed(1)} ms` : '--');
  setEl('time-llm', `${(latency.llm_ms || 0).toFixed(1)} ms`);

  // STT Chip
  const chipStt = document.getElementById('chip-stt');
  if (chipStt) {
    if (latency.stt_ms) {
      chipStt.style.display = 'flex';
      setEl('time-stt', `${latency.stt_ms.toFixed(1)} ms`);
    } else {
      chipStt.style.display = 'none';
    }
  }

  // Eval Chip
  const chipEval = document.getElementById('chip-eval');
  if (chipEval) {
    if (latency.eval_ms) {
      chipEval.style.display = 'flex';
      setEl('time-eval', `${latency.eval_ms.toFixed(1)} ms`);
    } else {
      chipEval.style.display = 'none';
    }
  }
}

// ============================================================
// Evaluation Scorecard Sidebar Updates
// ============================================================
function updateEvaluationSidebar(evalData) {
  const emptyState = document.getElementById('eval-empty-state');
  const scoreContent = document.getElementById('eval-score-content');
  const goldenBadge = document.getElementById('badge-golden-status');
  const openBadge = document.getElementById('badge-open-status');

  if (!evalData || !evalData.scores) {
    if (emptyState) emptyState.style.display = 'flex';
    if (scoreContent) scoreContent.style.display = 'none';
    if (goldenBadge) goldenBadge.style.display = 'none';
    if (openBadge) openBadge.style.display = 'none';
    return;
  }

  if (emptyState) emptyState.style.display = 'none';
  if (scoreContent) scoreContent.style.display = 'flex';

  // Badges: Golden vs Open Context Check
  if (evalData.is_golden) {
    if (goldenBadge) goldenBadge.style.display = 'inline-flex';
    if (openBadge) openBadge.style.display = 'none';
  } else {
    if (goldenBadge) goldenBadge.style.display = 'none';
    if (openBadge) openBadge.style.display = 'inline-flex';
  }

  const scores = evalData.scores;
  const overall = scores.overall_score !== null && scores.overall_score !== undefined ? scores.overall_score : 0.0;
  
  // 1. Overall Score Ring & Number
  const overallScoreEl = document.getElementById('eval-overall-score');
  if (overallScoreEl) overallScoreEl.textContent = overall.toFixed(2);

  const ringBar = document.getElementById('score-ring-bar');
  if (ringBar) {
    const circumference = 2 * Math.PI * 26; // ~163.36
    const offset = circumference - (Math.min(1, Math.max(0, overall)) * circumference);
    ringBar.style.strokeDashoffset = offset;

    if (overall >= 0.85) {
      ringBar.style.stroke = '#10b981';
    } else if (overall >= 0.70) {
      ringBar.style.stroke = '#3b82f6';
    } else if (overall >= 0.50) {
      ringBar.style.stroke = '#f59e0b';
    } else {
      ringBar.style.stroke = '#ef4444';
    }
  }

  // 2. Verdict Tag & Meta Pills
  const verdictTag = document.getElementById('eval-verdict-tag');
  const verdictText = document.getElementById('eval-verdict-text');
  if (verdictTag && verdictText) {
    verdictTag.className = 'quality-verdict-tag';
    if (overall >= 0.85) {
      verdictTag.classList.add('excellent');
      verdictText.textContent = 'Excellent Quality ✨';
    } else if (overall >= 0.70) {
      verdictTag.classList.add('high');
      verdictText.textContent = 'High Quality 🎯';
    } else if (overall >= 0.50) {
      verdictTag.classList.add('fair');
      verdictText.textContent = 'Fair Alignment ⚠️';
    } else {
      verdictTag.classList.add('low');
      verdictText.textContent = 'Low Alignment ❌';
    }
  }

  const queryTypeEl = document.getElementById('eval-query-type');
  if (queryTypeEl) queryTypeEl.textContent = evalData.query_type || (evalData.is_golden ? 'GOLDEN' : 'GENERAL');

  const queryIdEl = document.getElementById('eval-query-id');
  if (queryIdEl) queryIdEl.textContent = evalData.query_id ? `#${evalData.query_id}` : 'N/A';

  // 3. Set 5 Individual Metric Cards & Progress Bars
  const setBar = (valId, barId, val) => {
    const valEl = document.getElementById(valId);
    const barEl = document.getElementById(barId);
    if (val !== null && val !== undefined) {
      if (valEl) valEl.textContent = val.toFixed(2);
      if (barEl) barEl.style.width = `${Math.min(100, Math.max(0, val * 100))}%`;
    } else {
      if (valEl) valEl.textContent = '--';
      if (barEl) barEl.style.width = '0%';
    }
  };

  setBar('m-correctness', 'bar-correctness', scores.answer_correctness);
  setBar('m-faithfulness', 'bar-faithfulness', scores.faithfulness);
  setBar('m-relevance', 'bar-relevance', scores.answer_relevance);
  setBar('m-similarity', 'bar-similarity', scores.answer_similarity);
  setBar('m-recall', 'bar-recall', scores.context_recall);

  // 4. Ground Truth reference card
  const gtBox = document.getElementById('gt-box');
  const gtText = document.getElementById('gt-text');
  const gtBadge = document.getElementById('gt-badge');
  const copyBtn = document.getElementById('gt-copy-btn');
  const speakBtn = document.getElementById('gt-speak-btn');

  if (gtBox) {
    gtBox.style.display = 'block';
    if (evalData.ground_truth_answer) {
      gtBox.classList.remove('open-mode');
      if (gtBadge) {
        gtBadge.textContent = 'Verified Target';
        gtBadge.style.display = 'inline-block';
      }
      if (gtText) gtText.textContent = evalData.ground_truth_answer;
      if (copyBtn) copyBtn.style.display = 'inline-block';
      if (speakBtn) speakBtn.style.display = 'inline-block';
    } else {
      gtBox.classList.add('open-mode');
      if (gtBadge) {
        gtBadge.textContent = 'Open Query';
        gtBadge.style.display = 'inline-block';
      }
      if (gtText) {
        gtText.innerHTML = '<span style="color:var(--text-muted); font-size:0.75rem;">ℹ️ This is an open/unlisted query with no pre-indexed ground truth answer. System quality was evaluated dynamically for context faithfulness & relevance against retrieved documents.</span>';
      }
      if (copyBtn) copyBtn.style.display = 'none';
      if (speakBtn) speakBtn.style.display = 'none';
    }
  }

  // 5. LLM Judge Explanation card (DeepEval Reasons)
  const expBox = document.getElementById('eval-explanation-box');
  const expText = document.getElementById('eval-explanation-text');
  if (expBox) {
    const reasonText = scores.reason || scores.explanation;
    if (reasonText) {
      expBox.style.display = 'flex';
      if (expText) expText.textContent = reasonText;
    } else {
      expBox.style.display = 'none';
    }
  }
}

function copyGroundTruth() {
  const gtText = document.getElementById('gt-text');
  if (gtText && gtText.textContent && gtText.textContent !== '--') {
    copyTextToClipboard(gtText.textContent);
  }
}

function speakGroundTruth() {
  const gtText = document.getElementById('gt-text');
  if (gtText && gtText.textContent && gtText.textContent !== '--') {
    speakText(gtText.textContent);
  }
}

// ============================================================
// Utilities: TTS, Markdown, Copy, Time
// ============================================================
function speakText(text) {
  if (!('speechSynthesis' in window)) {
    alert("Speech synthesis is not supported in this browser.");
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const activeLang = STATE.currentLanguage || 'gu';
  const langInfo = LANGUAGES[activeLang] || LANGUAGES.gu;
  utterance.lang = langInfo.locale;
  utterance.rate = 0.95;
  window.speechSynthesis.speak(utterance);
}

function copyTextToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    alert("Copied to clipboard!");
  }).catch(() => {
    alert("Could not copy text.");
  });
}

function formatAnswerMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\n\*\s+(.*)/g, '<br>• $1');
  html = html.replace(/\n/g, '<br>');
  return html;
}

function formatTime(d) {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
}

function escapeJs(str) {
  if (!str) return '';
  return str.replace(/\\/g, '\\\\')
            .replace(/'/g, "\\'")
            .replace(/"/g, '\\"')
            .replace(/\n/g, ' ');
}

function setControlsDisabled(disabled) {
  const sendBtn = document.getElementById('send-query-btn');
  const uploadBtn = document.getElementById('upload-audio-btn');
  const qInput = document.getElementById('query-input');

  if (sendBtn) sendBtn.disabled = disabled;
  if (uploadBtn) uploadBtn.disabled = disabled;
  if (qInput) qInput.disabled = disabled;
}

function toggleTheme() {
  document.body.classList.toggle('dark-theme');
  const isDark = document.body.classList.contains('dark-theme');
  const icon = document.getElementById('theme-icon');
  if (icon) icon.textContent = isDark ? '🌙' : '☀️';
}

function startNewSession() {
  const feed = document.getElementById('chat-feed');
  if (feed) {
    feed.innerHTML = '';
    const activeLang = STATE.currentLanguage || 'gu';
    const langInfo = LANGUAGES[activeLang] || LANGUAGES.gu;
    appendBotMessage({
      answer: langInfo.welcomeMsg,
      timestamp: formatTime(new Date()),
      sources: []
    });
  }
}

function toggleTelemetryPanel() {
  const layout = document.querySelector('.app-layout');
  const panel = document.getElementById('telemetry-panel');
  if (layout) {
    layout.classList.toggle('inspector-collapsed');
  }
  if (panel) {
    panel.classList.toggle('open');
  }
}

// ============================================================
// Global Window Binding for Inline HTML Handlers
// ============================================================
window.switchMode = switchMode;
window.toggleVoiceRecording = toggleVoiceRecording;
window.submitCurrentQuery = submitCurrentQuery;
window.submitTextQuery = submitTextQuery;
window.triggerAudioUpload = triggerAudioUpload;
window.handleAudioFileUpload = handleAudioFileUpload;
window.handleInputResize = handleInputResize;
window.handleInputKeyDown = handleInputKeyDown;
window.toggleTheme = toggleTheme;
window.startNewSession = startNewSession;
window.toggleTelemetryPanel = toggleTelemetryPanel;
window.openHistoryModal = openHistoryModal;
window.closeHistoryModal = closeHistoryModal;
window.clearHistory = clearHistory;
window.toggleSourcesDrawer = toggleSourcesDrawer;
window.copyTextToClipboard = copyTextToClipboard;
window.speakText = speakText;
window.fetchSampleQueries = fetchSampleQueries;
window.refreshModalSampleQueries = refreshModalSampleQueries;
window.fillRandomSampleQuery = fillRandomSampleQuery;
window.openAllQueriesModal = openAllQueriesModal;
window.closeAllQueriesModal = closeAllQueriesModal;
window.filterModalQueries = filterModalQueries;
window.filterModalByCategory = filterModalByCategory;
window.handleModalSearchInput = handleModalSearchInput;
window.useQueryInInput = useQueryInInput;
window.toggleLanguageMenu = toggleLanguageMenu;
window.selectLanguage = selectLanguage;
window.copyGroundTruth = copyGroundTruth;
window.speakGroundTruth = speakGroundTruth;
