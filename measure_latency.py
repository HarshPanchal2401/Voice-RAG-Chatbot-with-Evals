import time
import sys
import numpy as np
from pipeline import GujaratiHybridRAGNoReranker

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("⏳ Initializing RAG pipeline...")
pipeline = GujaratiHybridRAGNoReranker(load_groq=True, verbose=False)

test_queries = [
    "મેનહટન પ્રોજેક્ટ શું હતો?",
    "રોબર્ટ ઓપનહેઇમર કોણ હતા?",
    "પરમાણુ બોમ્બ ક્યારે બનાવવામાં આવ્યો હતો?",
]

print("🔥 Warming up...")
_ = pipeline.ask(test_queries[0])

records = {
    "query_encoding_ms": [],
    "dense_search_ms": [],
    "sparse_search_ms": [],
    "rrf_ms": [],
    "metadata_ms": [],
    "retrieval_total_ms": [],
    "llm_ms": [],
    "total_ms": [],
}

print("📊 Running latency benchmark across test queries...\n")
for q in test_queries:
    res = pipeline.ask(q)
    for k in ["query_encoding_ms", "dense_search_ms", "sparse_search_ms", "rrf_ms", "metadata_ms", "retrieval_total_ms"]:
        records[k].append(res["retrieval_timings"][k])
    records["llm_ms"].append(res["llm_ms"])
    records["total_ms"].append(res["total_ms"])

print("=" * 60)
print(f"{'STAGE':<28} | {'AVG TIME (ms)':<15} | {'SHARE OF TOTAL'}")
print("=" * 60)

total_avg = np.mean(records["total_ms"])

for stage, key in [
    ("1. BGE-M3 Query Encoding", "query_encoding_ms"),
    ("2. FAISS Dense Search", "dense_search_ms"),
    ("3. Sparse Lexical Search", "sparse_search_ms"),
    ("4. RRF Rank Fusion", "rrf_ms"),
    ("5. SQLite Metadata Fetch", "metadata_ms"),
    ("👉 RETRIEVAL SUBTOTAL", "retrieval_total_ms"),
    ("6. Groq LLM Generation", "llm_ms"),
    ("👉 TOTAL END-TO-END", "total_ms"),
]:
    avg_val = np.mean(records[key])
    pct = (avg_val / total_avg) * 100
    if "TOTAL" in stage:
        print("-" * 60)
    print(f"{stage:<28} | {avg_val:>10.2f} ms   | {pct:>6.1f}%")
print("=" * 60)
