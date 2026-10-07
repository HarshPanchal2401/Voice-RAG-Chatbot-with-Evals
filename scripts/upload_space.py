"""
Upload this project to a Hugging Face Space without the shell expanding wildcards
(Windows expands `*` before `hf upload` sees it).

Usage:
    python scripts/upload_space.py                       # -> harshpanchal241/voice-rag
    python scripts/upload_space.py --repo USER/SPACE --dry-run
"""
import argparse
import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IGNORE = [
    ".env", ".env.*", "*.env",                 # secrets never leave this machine
    "voice_rag_builder/*",                     # indexes come from the private dataset
    "Data/*/golden_dataset_full.jsonl",
    "Data/*/*.sqlite",
    "Data/threads_store.json",
    "Data/voice_quota.json",
    "adiitional_test_notebook/*",
    "evaluation/reports/*",
    "evaluation/.judge_cache/*",
    "**/__pycache__/*", "*.pyc",
    ".pytest_cache/*", ".venv/*", "venv/*", ".git/*",
    "Dockerfile", ".dockerignore",
    "*.wav", "*.mp3", "*.webm", "*.log",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="harshpanchal241/voice-rag")
    ap.add_argument("--dry-run", action="store_true", help="only list the files that would be uploaded")
    args = ap.parse_args()

    if args.dry_run:
        files = []
        for p in ROOT.rglob("*"):
            if p.is_file():
                rel = p.relative_to(ROOT).as_posix()
                if not any(fnmatch.fnmatch(rel, pat) for pat in IGNORE):
                    files.append(rel)
        print("\n".join(sorted(files)))
        print(f"\n{len(files)} files would be uploaded to {args.repo}")
        return

    from huggingface_hub import HfApi
    info = HfApi().upload_folder(
        repo_id=args.repo,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=IGNORE,
        commit_message="Deploy Voice RAG",
    )
    print(f"✅ Uploaded: {info}")
    print(f"🔗 https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
