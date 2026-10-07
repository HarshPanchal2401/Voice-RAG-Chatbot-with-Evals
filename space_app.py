"""
Hugging Face Spaces entry point (Gradio SDK on free ZeroGPU hardware).

Runs the normal FastAPI app and web UI on port 7860:
  1. downloads the indexes from the dataset in the RAG_INDEX_REPO secret (if missing),
  2. starts uvicorn with app:app.
Everything runs on CPU; ZeroGPU only requires one @spaces.GPU function to exist.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# CUDA is only usable inside @spaces.GPU functions on ZeroGPU, so keep the pipeline on CPU.
os.environ.setdefault("RAG_DEVICE", "cpu")
# Spaces sits behind a proxy: use X-Forwarded-For to tell users apart (voice quota).
os.environ.setdefault("RAG_TRUST_PROXY", "1")
# Where the prebuilt indexes live (private dataset; needs the HF_TOKEN secret).
os.environ.setdefault("RAG_INDEX_REPO", "harshpanchal241/voice-rag-indexes")
# Spaces hosts report ~192 CPUs; too many torch threads slows single-query encoding.
os.environ.setdefault("RAG_TORCH_THREADS", "8")
# Same-origin UI on *.hf.space; keep CORS closed to other sites.
os.environ.setdefault("RAG_CORS_ORIGINS", os.environ.get("SPACE_HOST", "") and f"https://{os.environ['SPACE_HOST']}")

try:  # ZeroGPU refuses to start without at least one registered GPU function.
    import spaces

    @spaces.GPU(duration=5)
    def gpu_probe() -> bool:
        import torch
        return torch.cuda.is_available()

    # `spaces` normally reports startup when a Gradio app launches; this app serves FastAPI
    # directly, so send the ZeroGPU startup report ourselves (only defined on ZeroGPU hardware).
    from spaces import zero as _zero
    if hasattr(_zero, "startup"):
        _zero.startup()
        print("✅ [space_app] ZeroGPU startup reported.")
except Exception as exc:  # not on ZeroGPU (local run / CPU Space)
    print(f"ℹ️ [space_app] spaces package unavailable ({exc}); running without ZeroGPU hooks.")

if __name__ == "__main__":
    rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_indexes.py")]).returncode
    if rc != 0:
        print("⚠️ [space_app] Index download incomplete; the API will start without some languages.")
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), workers=1)
