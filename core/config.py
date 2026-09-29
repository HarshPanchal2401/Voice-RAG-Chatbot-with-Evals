"""
Core Configuration & Environment Management
===========================================
Central settings, directory paths, model names, and pipeline parameters.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Block crash packages on Python 3.14 + Windows before faiss
for _crash_pkg in ("sklearn", "pandas", "pyarrow", "datasets"):
    if _crash_pkg not in sys.modules:
        sys.modules[_crash_pkg] = None  # type: ignore[assignment]

# Root of v3
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Compute Device
import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_FP16 = torch.cuda.is_available()

def get_default_torch_threads() -> int:
    env = os.environ.get("RAG_TORCH_THREADS")
    if env:
        return max(1, int(env))
    return max(1, (os.cpu_count() or 2) // 2)

# Models
BGE_MODEL_NAME = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
GROQ_MODEL_NAME = os.environ.get("RAG_LLM_MODEL", "qwen/qwen3.8-27b")
RERANKER_MODEL_NAME = os.environ.get("RAG_RERANKER_MODEL", "openai/gpt-oss-120b")
DEFAULT_JUDGE_MODEL = os.environ.get("RAG_JUDGE_MODEL", "openai/gpt-oss-120b")

# Generation Hyperparameters
LLM_MAX_TOKENS = int(os.environ.get("RAG_LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = 0.0

# Retrieval Hyperparameters
BGE_QUERY_MAX_LENGTH = int(os.environ.get("RAG_MAX_QUERY_LEN", "128"))
DENSE_TOP_N = int(os.environ.get("RAG_DENSE_TOP_N", "50"))
SPARSE_TOP_N = int(os.environ.get("RAG_SPARSE_TOP_N", "50"))
RRF_TOP_N = int(os.environ.get("RAG_RRF_TOP_N", "30"))
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
FINAL_TOP_K = int(os.environ.get("RAG_FINAL_TOP_K", "5"))
DEFAULT_HNSW_EF_SEARCH = int(os.environ.get("RAG_HNSW_EF_SEARCH", "64"))

# Multilingual Language Metadata
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

# System Prompts
SYSTEM_PROMPTS = {
    "gu": """
તમે એક બુદ્ધિશાળી ગુજરાતી AI સહાયક છો.

નિયમો:
1. આપેલા સંદર્ભ (Context) ના આધારે જ સરળ, કુદરતી અને સચોટ ગુજરાતીમાં જવાબ આપો.
2. જો સંદર્ભમાં પૂરતી માહિતી ન હોય તો પ્રામાણિકપણે જણાવો: "આપેલ સંદર્ભમાં આ અંગે પૂરતી માહિતી ઉપલબ્ધ નથી."
3. સંદર્ભ બહારનું અનુમાન કે અસત્ય વિગતો ઉમેરશો નહીં.
""".strip(),

    "hi": """
आप एक बुद्धिमान हिंदी AI सहायक हैं।

नियम:
1. दिए गए संदर्भ (Context) के आधार पर ही सरल, स्वाभाविक और सटीक हिंदी में उत्तर दें।
2. यदि संदर्भ में पर्याप्त जानकारी नहीं है, तो ईमानदारी से बताएं: "दिए गए संदर्भ में इस बारे में पर्याप्त जानकारी उपलब्ध नहीं है।"
3. संदर्भ के बाहर का कोई अनुमान या असत्य विवरण न जोड़ें।
""".strip()
}

# Directories
BUILDER_DIR = BASE_DIR / "voice_rag_builder"
DATA_DIR = BASE_DIR / "Data"
STATIC_DIR = BASE_DIR / "static"
