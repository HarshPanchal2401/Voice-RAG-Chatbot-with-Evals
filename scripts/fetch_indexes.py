"""
Download the prebuilt indexes (voice_rag_builder/) from a Hugging Face dataset repo
when they are not already present. Used by the Docker image on startup.

Env:
  RAG_INDEX_REPO   e.g. "your-username/voice-rag-indexes" (unset = skip download)
  HF_TOKEN         needed when the dataset repo is private
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "voice_rag_builder"
NEEDED = ["pipeline_manifest.json", "bge_m3_sparse_lexical.npz", "passage_metadata.sqlite"]
LANGS = [l.strip() for l in os.environ.get("RAG_INDEX_FOLDERS", "gujarati,hindi").split(",") if l.strip()]


def present() -> bool:
    for lang in LANGS:
        d = TARGET / lang
        if not all((d / f).exists() for f in NEEDED):
            return False
        if not list(d.glob("bge_m3_dense_hnsw*.index")):
            return False
    return True


def main() -> int:
    if present():
        print("✅ [fetch_indexes] Indexes already present.")
        return 0
    repo = os.environ.get("RAG_INDEX_REPO", "").strip()
    if not repo:
        print("⚠️ [fetch_indexes] Indexes missing and RAG_INDEX_REPO is not set.")
        return 0
    from huggingface_hub import snapshot_download
    token = (os.environ.get("HF_TOKEN") or "").strip().strip("\"'").strip() or None
    has_token = bool(token)
    print(f"ℹ️ [fetch_indexes] HF_TOKEN secret present: {'yes' if has_token else 'NO'}")
    if token:
        try:
            from huggingface_hub import whoami
            info = whoami(token=token)
            print(f"ℹ️ [fetch_indexes] HF_TOKEN belongs to '{info.get('name')}' (token role: {info.get('auth', {}).get('accessToken', {}).get('role', '?')})")
        except Exception as e:
            print(f"❌ [fetch_indexes] HF_TOKEN is invalid: {type(e).__name__}. Create a new token and update the secret.")
    if not has_token:
        print("❌ [fetch_indexes] A private dataset needs the HF_TOKEN secret (Space Settings → Variables and secrets → New secret).")
    print(f"⏳ [fetch_indexes] Downloading indexes from dataset '{repo}' ...")
    patterns = []
    for lang in LANGS:
        patterns += [f"{lang}/{f}" for f in NEEDED] + [f"{lang}/bge_m3_dense_hnsw*.index"]
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(TARGET),
        allow_patterns=patterns,
        token=token,
    )
    ok = present()
    print("✅ [fetch_indexes] Done." if ok else "❌ [fetch_indexes] Some files are still missing.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
