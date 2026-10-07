"""
Verification Suite for Multi-Language Voice & Text RAG System
=============================================================
Tests LanguageRouter, Gujarati & Hindi pipelines, DeepEval Evaluators, and FastAPI Endpoints.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# UTF-8 encoding support for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 60)
print("🧪 [1/6] Initializing LanguageRouter...")
print("=" * 60)
t0 = time.time()
from pipeline.router import LanguageRouter
router = LanguageRouter(verbose=True)
print(f"✅ Router initialized in {time.time() - t0:.2f}s")

# 1. Check Available Languages
print("\n" + "=" * 60)
print("🌐 [2/6] Checking Registered Languages...")
print("=" * 60)
langs = router.get_available_languages()
print("Registered Languages:")
for l in langs:
    print(f"  - [{l['code'].upper()}] {l['name']} ({l['native_name']}): {l['indexed_passages']:,} passages, {l['golden_dataset_records']:,} golden records")

assert len(langs) >= 2, f"Expected at least 2 languages, got {len(langs)}"
codes = [l['code'] for l in langs]
assert "gu" in codes, "Gujarati pipeline not registered"
assert "hi" in codes, "Hindi pipeline not registered"
print("✅ Language registration check passed!")

# 2. Test Gujarati Retrieval & Generation
print("\n" + "=" * 60)
print("🇬🇺 [3/6] Testing Gujarati Hybrid RAG & Golden Evaluation...")
print("=" * 60)
gu_query = "વાઇનની બોટલમાં કેટલા ઔંસ હોય છે"
t0 = time.time()
gu_res = router.ask_and_eval(gu_query, lang="gu", final_k=3)
print(f"Query: {gu_query}")
print(f"Answer: {gu_res['answer']}")
print(f"Language: {gu_res['language']}")
print(f"Retrieved Chunks: {len(gu_res['documents'])}")
print(f"Is Golden: {gu_res['evaluation']['is_golden']}")
print(f"Overall Score: {gu_res['evaluation']['scores']['overall_score'] if gu_res['evaluation']['scores'] else 'N/A'}")
print(f"Total Time: {time.time() - t0:.2f}s")
assert gu_res['language'] == 'gu'
assert len(gu_res['documents']) > 0

# 3. Test Hindi Retrieval & Generation
print("\n" + "=" * 60)
print("🇮🇳 [4/6] Testing Hindi Hybrid RAG & Golden Evaluation...")
print("=" * 60)
hi_query = "क्या वह शादीशुदा है?"
t0 = time.time()
hi_res = router.ask_and_eval(hi_query, lang="hi", final_k=3)
print(f"Query: {hi_query}")
print(f"Answer: {hi_res['answer']}")
print(f"Language: {hi_res['language']}")
print(f"Retrieved Chunks: {len(hi_res['documents'])}")
print(f"Is Golden: {hi_res['evaluation']['is_golden']}")
print(f"Overall Score: {hi_res['evaluation']['scores']['overall_score'] if hi_res['evaluation']['scores'] else 'N/A'}")
print(f"Total Time: {time.time() - t0:.2f}s")
assert hi_res['language'] == 'hi'
assert len(hi_res['documents']) > 0

# 4. Test Sample Queries for both languages
print("\n" + "=" * 60)
print("🎯 [5/6] Testing Dynamic Sample Queries...")
print("=" * 60)
gu_samples = router.get_sample_queries(lang="gu", count=5)
hi_samples = router.get_sample_queries(lang="hi", count=5)
print(f"Gujarati Samples ({len(gu_samples)}):", [s['question'] for s in gu_samples[:2]])
print(f"Hindi Samples ({len(hi_samples)}):", [s['question'] for s in hi_samples[:2]])
assert len(gu_samples) > 0
assert len(hi_samples) > 0

# 5. Test FastAPI Endpoints via TestClient
print("\n" + "=" * 60)
print("🚀 [6/6] Testing FastAPI Server Endpoints via TestClient...")
print("=" * 60)
from starlette.testclient import TestClient
from app import app
from services.voice_service import VoiceService

app.state.router = router
app.state.pipeline = router.get_pipeline("gu")
app.state.voice_service = VoiceService(groq_client=router.groq_client)

with TestClient(app) as client:
    # Test /api/v1/languages
    r = client.get("/api/v1/languages")
    assert r.status_code == 200, f"Languages endpoint failed: {r.text}"
    print("GET /api/v1/languages ->", r.json())

    # Test /health
    r = client.get("/health")
    assert r.status_code == 200, f"Health endpoint failed: {r.text}"
    print("GET /health ->", r.json())

    # Test /api/v1/sample-queries?lang=hi
    r = client.get("/api/v1/sample-queries?lang=hi&count=3")
    assert r.status_code == 200, f"Sample queries failed: {r.text}"
    print("GET /api/v1/sample-queries?lang=hi ->", len(r.json()["queries"]), "queries returned")

    # Test POST /api/v1/query/text (Hindi with auto-detection)
    r = client.post("/api/v1/query/text", json={
        "query": "गूगल पिनयिन इनपुट क्या है",
        "language": "auto",
        "top_k": 3,
        "evaluate": True
    })
    assert r.status_code == 200, f"Text query failed: {r.text}"
    res_json = r.json()
    print("POST /api/v1/query/text (Hindi) ->")
    print("  Answer:", res_json["answer"])
    print("  Language:", res_json["language"])
    assert res_json["language"] == "hi"
    print("  Evaluation:", res_json["evaluation"]["scores"]["overall_score"] if res_json["evaluation"] and res_json["evaluation"]["scores"] else "None")

print("\n" + "=" * 60)
print("🎉 ALL MULTILINGUAL RAG & LANGUAGE ROUTER TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
