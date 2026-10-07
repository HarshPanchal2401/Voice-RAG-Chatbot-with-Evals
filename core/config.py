"""
Core Configuration & Environment Management
===========================================
Central settings, directory paths, model names, and pipeline parameters.
Every tunable is read from the environment (``.env`` is loaded first), so this module is the
single source of truth for model names and retrieval / generation knobs.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Root of v3
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env before reading any setting (including RAG_ALLOW_PANDAS below)
load_dotenv(dotenv_path=BASE_DIR / ".env")


# ------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------

def _clean(value: str) -> str:
    return value.strip().strip('"').strip("'").strip()


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = _clean(value)
    return value if value else default


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not _clean(value):
        return default
    return _clean(value).lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_optional_float(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if value is None or not _clean(value):
        return None
    try:
        return float(_clean(value))
    except ValueError:
        return None


# ------------------------------------------------------------
# Python 3.14 + Windows workaround
# ------------------------------------------------------------
# On Python 3.14 / Windows, importing sklearn / pandas / pyarrow / datasets next to faiss
# crashes the interpreter. Block them only there (they work on Linux / Docker and on older
# Pythons). Set RAG_ALLOW_PANDAS=1 to opt out of the block.
CRASH_PACKAGES_BLOCKED = (
    sys.platform == "win32"
    and sys.version_info >= (3, 14)
    and not _env_flag("RAG_ALLOW_PANDAS", False)
)
if CRASH_PACKAGES_BLOCKED:
    for _crash_pkg in ("sklearn", "pandas", "pyarrow", "datasets"):
        if _crash_pkg not in sys.modules:
            sys.modules[_crash_pkg] = None  # type: ignore[assignment]

# Compute Device
import torch  # noqa: E402

# RAG_DEVICE forces a device (e.g. "cpu" on Hugging Face ZeroGPU, where CUDA only works inside @spaces.GPU calls).
DEVICE = (os.environ.get("RAG_DEVICE") or "").strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
USE_FP16 = DEVICE.startswith("cuda")


def get_default_torch_threads() -> int:
    env = os.environ.get("RAG_TORCH_THREADS")
    if env:
        return max(1, int(env))
    return max(1, os.cpu_count() or 4)


# ------------------------------------------------------------
# Models (roles must stay distinct: generator / reranker != judge)
# ------------------------------------------------------------
BGE_MODEL_NAME = _env_str("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_MODEL_NAME = BGE_MODEL_NAME
GROQ_MODEL_NAME = _env_str("RAG_LLM_MODEL", "qwen/qwen3.8-27b")
LLM_MODEL_NAME = GROQ_MODEL_NAME
DEFAULT_JUDGE_MODEL = _env_str("RAG_JUDGE_MODEL", "openai/gpt-oss-120b")

# Re-ranking backends:
#   "llm"           (default) a Groq chat model grades every candidate 0-3 in one call
#   "cross_encoder" (opt-in) local transformers cross-encoder, lazy-loaded (~2.3 GB RAM fp32)
#   "lexical"       zero-dependency whole-token overlap scorer (also the fallback of the others)
RERANKER_BACKENDS = ("llm", "cross_encoder", "lexical")
RERANKER_BACKEND = _env_str("RAG_RERANKER_BACKEND", "llm").lower().replace("-", "_")
if RERANKER_BACKEND not in RERANKER_BACKENDS:
    print(f"⚠️ [config] Unknown RAG_RERANKER_BACKEND={RERANKER_BACKEND!r}; using 'llm'.")
    RERANKER_BACKEND = "llm"
# LLM reranker model; must differ from the judge model. Latency check (10 Gujarati passages,
# one call each, 2026-10-06): qwen/qwen3.8-27b 1.8 s, graded the relevant passage 2/3, no
# reasoning tokens; openai/gpt-oss-20b 11.7 s cold / 1.8 s warm, 147 reasoning tokens even at
# effort=low, graded every passage 0 (would have forced a no-answer). -> qwen/qwen3.8-27b.
RERANKER_MODEL_NAME = _env_str("RAG_RERANKER_MODEL", "qwen/qwen3.8-27b")
CROSS_ENCODER_MODEL_NAME = _env_str("RAG_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
CROSS_ENCODER_QUANTIZE = _env_flag("RAG_CROSS_ENCODER_QUANTIZE", True)  # int8 dynamic quantization on CPU
CROSS_ENCODER_BATCH_SIZE = max(1, _env_int("RAG_CROSS_ENCODER_BATCH", 8))
CROSS_ENCODER_MAX_LENGTH = max(32, _env_int("RAG_CROSS_ENCODER_MAX_LEN", 512))

# Minimum rerank score kept after re-ranking. Unset -> per-backend default:
#   llm 0.3 (drops grade 0 = unrelated; scores are grade/3 = 0, .33, .67, 1),
#   cross_encoder 0.01 (sigmoid relevance), lexical 0.0 (keeps everything).
# If every candidate falls below it, retrieval returns no documents -> no-answer path.
RERANK_MIN_SCORE = _env_optional_float("RAG_RERANK_MIN_SCORE")
RERANK_DEFAULT_MIN_SCORES = {"llm": 0.3, "cross_encoder": 0.01, "lexical": 0.0}
RERANK_POOL = max(1, _env_int("RAG_RERANK_POOL", 10))             # candidates sent to the reranker
RERANK_TIMEOUT = max(1.0, _env_float("RAG_RERANK_TIMEOUT", 8.0))  # seconds, one LLM attempt, no retries
RERANK_MAX_TOKENS = max(64, _env_int("RAG_RERANK_MAX_TOKENS", 512))
RERANK_PASSAGE_CHARS = max(100, _env_int("RAG_RERANK_PASSAGE_CHARS", 400))

# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------
LLM_MAX_TOKENS = _env_int("RAG_LLM_MAX_TOKENS", 512)
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT = max(5.0, _env_float("RAG_LLM_TIMEOUT", 30.0))
LLM_MAX_RETRIES = max(0, _env_int("RAG_LLM_MAX_RETRIES", 3))
# Reasoning control for Groq reasoning models: none | low | medium | high | default | auto | off.
# "auto" -> "none" for Qwen3 models, "low" for gpt-oss models (they cannot disable reasoning),
# nothing for other models. "off" never sends the parameter.
LLM_REASONING_EFFORT = _env_str("RAG_LLM_REASONING_EFFORT", "auto").lower()
RERANK_REASONING_EFFORT = _env_str("RAG_RERANK_REASONING_EFFORT", LLM_REASONING_EFFORT).lower()
# Optional Groq `reasoning_format` (hidden | parsed | raw); empty = not sent. Never sent to gpt-oss.
LLM_REASONING_FORMAT = _env_str("RAG_LLM_REASONING_FORMAT", "").lower()

# Conversation memory
HISTORY_TURNS = max(0, _env_int("RAG_HISTORY_TURNS", 6))               # messages (user + assistant)
HISTORY_MAX_CHARS = max(100, _env_int("RAG_HISTORY_MAX_CHARS", 1200))  # per message
CONDENSE_QUERY = _env_flag("RAG_CONDENSE_QUERY", True)
CONDENSE_MODEL_NAME = _env_str("RAG_CONDENSE_MODEL", GROQ_MODEL_NAME)
CONDENSE_MAX_TOKENS = max(16, _env_int("RAG_CONDENSE_MAX_TOKENS", 128))
CONDENSE_TIMEOUT = max(1.0, _env_float("RAG_CONDENSE_TIMEOUT", 6.0))

# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------
# Query truncation length in BGE-M3 tokens (incl. special tokens). Measured on the 2 x 500
# golden questions (Data/<lang>/golden_dataset_sample_500.jsonl): p50=12, p95=21, p99=32,
# max=497 (one outlier); gu p99=33, hi p99=26. 48 covers 99.9%. Single queries are not
# padded, so a larger value costs nothing for short questions.
BGE_QUERY_MAX_LENGTH = _env_int("RAG_MAX_QUERY_LEN", 48)
QUERY_CACHE_SIZE = max(0, _env_int("RAG_QUERY_CACHE_SIZE", 256))
DENSE_TOP_N = _env_int("RAG_DENSE_TOP_N", 50)
SPARSE_TOP_N = _env_int("RAG_SPARSE_TOP_N", 50)
RRF_TOP_N = _env_int("RAG_RRF_TOP_N", 30)
RRF_K = _env_int("RAG_RRF_K", 60)
FINAL_TOP_K = _env_int("RAG_FINAL_TOP_K", 5)
DEFAULT_HNSW_EF_SEARCH = _env_int("RAG_HNSW_EF_SEARCH", 24)
# Drop candidates whose dense (cosine) score is below this value. 0 = off (default).
MIN_DENSE_SCORE = _env_float("RAG_MIN_DENSE_SCORE", 0.0)

# ------------------------------------------------------------
# Languages
# ------------------------------------------------------------
# Index languages only (other code iterates this dict to discover pipelines).
LANGUAGE_METADATA = {
    "gu": {
        "code": "gu",
        "aliases": ["gu", "gujarati", "gu-in", "guj_gujr"],
        "name": "Gujarati",
        "native_name": "ગુજરાતી",
        "locale": "gu-IN",
        "folder": "gujarati",
        "context_header": "સંદર્ભ",
        "question_header": "પ્રશ્ન",
        "answer_header": "જવાબ",
    },
    "hi": {
        "code": "hi",
        "aliases": ["hi", "hindi", "hi-in", "hin_deva"],
        "name": "Hindi",
        "native_name": "हिन्दी",
        "locale": "hi-IN",
        "folder": "hindi",
        "context_header": "संदर्भ",
        "question_header": "प्रश्न",
        "answer_header": "उत्तर",
    }
}

# Exact refusal sentences (other code detects these strings - do not reword).
NO_ANSWER_MESSAGES = {
    "gu": "આપેલ સંદર્ભમાં આ અંગે પૂરતી માહિતી ઉપલબ્ધ નથી.",
    "hi": "दिए गए संदर्भ में इस बारे में पर्याप्त जानकारी उपलब्ध नहीं है।",
    "en": "There is not enough information in the provided context to answer this.",
}

# Answer languages = index languages + English (English questions answered from the Gujarati /
# Hindi index). `context_label` is the inline citation tag the model must use.
ANSWER_LANGUAGES = ("gu", "hi", "en")
ANSWER_LANGUAGE_META = {
    "gu": {
        "code": "gu", "name": "Gujarati", "locale": "gu-IN",
        "context_header": "સંદર્ભ", "context_label": "સંદર્ભ",
        "question_header": "પ્રશ્ન", "answer_header": "જવાબ",
        "no_answer": NO_ANSWER_MESSAGES["gu"],
    },
    "hi": {
        "code": "hi", "name": "Hindi", "locale": "hi-IN",
        "context_header": "संदर्भ", "context_label": "संदर्भ",
        "question_header": "प्रश्न", "answer_header": "उत्तर",
        "no_answer": NO_ANSWER_MESSAGES["hi"],
    },
    "en": {
        "code": "en", "name": "English", "locale": "en-IN",
        "context_header": "Context", "context_label": "Source",
        "question_header": "Question", "answer_header": "Answer",
        "no_answer": NO_ANSWER_MESSAGES["en"],
    },
}

# System prompts, keyed by answer language ("en" is used for English questions).
SYSTEM_PROMPTS = {
    "gu": f"""
તમે એક બુદ્ધિશાળી ગુજરાતી AI સહાયક છો.

નિયમો:
1. આપેલા સંદર્ભ (Context) ના આધારે જ સરળ, કુદરતી અને સચોટ ગુજરાતીમાં ટૂંકો જવાબ આપો (સામાન્ય રીતે 1 થી 3 વાક્યો).
2. જે સંદર્ભ પરથી માહિતી લીધી હોય તેનો ઉલ્લેખ વાક્યમાં જ [સંદર્ભ N] સ્વરૂપે કરો, જ્યાં N એ સંદર્ભનો નંબર છે (ઉદા. [સંદર્ભ 2]).
3. જો સંદર્ભમાં પૂરતી માહિતી ન હોય તો પ્રામાણિકપણે ફક્ત આટલું જ જણાવો: "{NO_ANSWER_MESSAGES['gu']}"
4. સંદર્ભ બહારનું અનુમાન કે અસત્ય વિગતો ઉમેરશો નહીં.
5. અગાઉની વાતચીત ફક્ત પ્રશ્ન સમજવા માટે છે; હકીકતો ફક્ત આપેલા સંદર્ભમાંથી જ લો.
""".strip(),

    "hi": f"""
आप एक बुद्धिमान हिंदी AI सहायक हैं।

नियम:
1. दिए गए संदर्भ (Context) के आधार पर ही सरल, स्वाभाविक और सटीक हिंदी में संक्षिप्त उत्तर दें (आमतौर पर 1 से 3 वाक्य)।
2. जिस संदर्भ से जानकारी ली है, उसका उल्लेख वाक्य में ही [संदर्भ N] के रूप में करें, जहाँ N संदर्भ की संख्या है (जैसे [संदर्भ 2])।
3. यदि संदर्भ में पर्याप्त जानकारी नहीं है, तो ईमानदारी से केवल यह बताएं: "{NO_ANSWER_MESSAGES['hi']}"
4. संदर्भ के बाहर का कोई अनुमान या असत्य विवरण न जोड़ें।
5. पिछली बातचीत केवल प्रश्न समझने के लिए है; तथ्य केवल दिए गए संदर्भ से लें।
""".strip(),

    "en": f"""
You are a helpful multilingual AI assistant. The context passages are in Gujarati or Hindi; the user asked in English.

Rules:
1. Answer ONLY from the given context, in clear, natural and concise English (usually 1 to 3 sentences).
2. Cite the supporting passage inline as [Source N], where N is the passage number (e.g. [Source 2]).
3. If the context does not contain enough information, reply with exactly: "{NO_ANSWER_MESSAGES['en']}"
4. Do not add guesses or facts that are not in the context.
5. Earlier conversation is only for understanding the question; take facts only from the given context.
""".strip(),
}

# Directories
BUILDER_DIR = BASE_DIR / "voice_rag_builder"
DATA_DIR = BASE_DIR / "Data"
STATIC_DIR = BASE_DIR / "static"
