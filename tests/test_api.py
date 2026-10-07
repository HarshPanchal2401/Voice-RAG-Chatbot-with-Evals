"""
Test Suite for Gujarati & Hindi Voice & Text RAG FastAPI Application
=====================================================================
Tests endpoints, sorting modes, script auto-detection, and response contracts.
"""

import sys
import time
import json
from pathlib import Path
from starlette.testclient import TestClient

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app

print("🚀 Initializing FastAPI Test Client (Triggering Lifespan)...")

with TestClient(app) as client:
    print("\n✅ FastAPI application loaded successfully!\n")

    # 0. Test GET / (Web UI HTML)
    print("--- [0/7] Testing GET / (Web Dashboard HTML) ---")
    resp_ui = client.get("/")
    assert resp_ui.status_code == 200, f"UI serving failed: {resp_ui.text}"
    assert "Gujarati Voice RAG" in resp_ui.text or "Voice" in resp_ui.text
    print("✅ GET / (Web UI) PASSED\n")

    # 1. Test /health
    print("--- [1/7] Testing GET /health ---")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    print(f"Health Response: {json.dumps(health_data, indent=2)}")
    assert health_data["status"] == "healthy"
    assert health_data["indexed_passages"] > 0
    print("✅ GET /health PASSED\n")

    # 2. Test /api/v1/sample-queries
    print("--- [2/7] Testing GET /api/v1/sample-queries ---")
    resp_sq = client.get("/api/v1/sample-queries?count=5")
    assert resp_sq.status_code == 200, f"Sample queries failed: {resp_sq.text}"
    sq_data = resp_sq.json()
    print(f"Fetched {len(sq_data['queries'])} sample queries from Golden Dataset")
    assert len(sq_data["queries"]) == 5
    print("✅ GET /api/v1/sample-queries PASSED\n")

    # 3. Test POST /api/v1/query/text (Gujarati Query with RRF sorting)
    print("--- [3/7] Testing POST /api/v1/query/text (Gujarati Query) ---")
    query_payload = {
        "query": "વાઇનની બોટલમાં કેટલા ઔંસ હોય છે",
        "language": "gu",
        "top_k": 3,
        "sort_by": "rrf",
        "evaluate": True
    }
    t0 = time.time()
    resp = client.post("/api/v1/query/text", json=query_payload)
    elapsed = time.time() - t0
    assert resp.status_code == 200, f"Query failed: {resp.text}"
    rag_data = resp.json()
    print(f"Query: {rag_data['query']}")
    print(f"Answer: {rag_data['answer']}")
    print(f"Language: {rag_data['language']}")
    print(f"Latency: {json.dumps(rag_data['latency'], indent=2)}")
    assert len(rag_data["sources"]) == 3
    assert rag_data["language"] == "gu"
    assert rag_data["latency"]["total_ms"] > 0
    print(f"✅ POST /api/v1/query/text (Gujarati) PASSED in {elapsed:.2f}s\n")

    # 4. Test POST /api/v1/query/text with Auto Script Detection (Hindi text with language='auto')
    print("--- [4/7] Testing POST /api/v1/query/text (Auto Script Detection for Hindi) ---")
    hi_payload = {
        "query": "क्या वह शादीशुदा है?",
        "language": "auto",
        "auto_detect_language": True,
        "top_k": 3,
        "sort_by": "rrf",
        "evaluate": False
    }
    resp = client.post("/api/v1/query/text", json=hi_payload)
    assert resp.status_code == 200, f"Hindi auto-detect failed: {resp.text}"
    hi_data = resp.json()
    print(f"Detected Language: {hi_data['language']}")
    assert hi_data["language"] == "hi", f"Expected language 'hi', got '{hi_data['language']}'"
    assert len(hi_data["sources"]) == 3
    print("✅ Hindi Auto Script Detection PASSED\n")

    # 5. Test POST /api/v1/retrieve (Retrieve-only with dense sorting)
    print("--- [5/7] Testing POST /api/v1/retrieve (Dense sorting mode) ---")
    ret_payload = {
        "query": "રોબર્ટ ઓપનહેઇમર કોણ હતા?",
        "top_k": 3,
        "sort_by": "dense"
    }
    resp = client.post("/api/v1/retrieve", json=ret_payload)
    assert resp.status_code == 200, f"Retrieve failed: {resp.text}"
    ret_data = resp.json()
    print(f"Retrieved {len(ret_data['sources'])} passages")
    assert len(ret_data["sources"]) == 3
    # Check that dense scores are sorted descending
    dense_scores = [s["dense_score"] for s in ret_data["sources"] if s.get("dense_score") is not None]
    assert dense_scores == sorted(dense_scores, reverse=True)
    print("✅ POST /api/v1/retrieve PASSED\n")

    # 6. Test POST /api/v1/query/text/stream (SSE Streaming)
    print("--- [6/7] Testing POST /api/v1/query/text/stream ---")
    stream_payload = {
        "query": "તે પરિણીત છે?",
        "top_k": 2,
        "evaluate": False
    }
    token_count = 0
    received_meta = False
    with client.stream("POST", "/api/v1/query/text/stream", json=stream_payload) as stream_resp:
        assert stream_resp.status_code == 200
        for line in stream_resp.iter_lines():
            if line.startswith("event: token"):
                token_count += 1
            elif line.startswith("event: meta"):
                received_meta = True

    print(f"Streamed tokens count: {token_count}, Received Final Meta: {received_meta}")
    assert token_count > 0
    assert received_meta
    print("✅ POST /api/v1/query/text/stream PASSED\n")

    # 7. Test POST /api/v1/evaluate (Standalone Evaluation)
    print("--- [7/7] Testing POST /api/v1/evaluate ---")
    eval_payload = {
        "question": "વાઇનની બોટલમાં કેટલા ઔંસ હોય છે",
        "generated_answer": "વાઇનની બોટલમાં આશરે 25 ઔંસ (24.75 ઔંસ) હોય છે.",
        "retrieved_contexts": ["વાઇનની નિયમિત બોટલ 750 મિલીલિટર હોય છે. તેથી, વાઇનની નિયમિત બોટલ આશરે 25 ઔંસ હોય છે."],
        "language": "gu"
    }
    resp = client.post("/api/v1/evaluate", json=eval_payload)
    assert resp.status_code == 200, f"Evaluate failed: {resp.text}"
    eval_data = resp.json()
    print(f"Standalone Eval Result: {json.dumps(eval_data, indent=2, ensure_ascii=False)}")
    assert eval_data["is_golden"] is True
    print("✅ POST /api/v1/evaluate PASSED\n")

print("🎉🎉 ALL FASTAPI ENDPOINT TESTS PASSED SUCCESSFULLY! 🎉🎉")
