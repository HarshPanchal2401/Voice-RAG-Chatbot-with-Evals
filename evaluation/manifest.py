"""
Provenance / reproducibility helpers
====================================
* ``sha256_file``         - content hash of datasets and source files
* ``git_info``            - commit + dirty flag of the repository (None when unavailable)
* ``code_version``        - sha256 over the evaluation code files that produced an artifact
* ``dataset_manifest``    - manifest written next to every generated dataset
* ``run_manifest``        - manifest block embedded in every evaluation report
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_info(repo: Path = BASE_DIR) -> Dict[str, Any]:
    def _run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def code_version(files: Iterable[Path | str] | None = None) -> Dict[str, Any]:
    """sha256 over the evaluation package source (or the given files) + git info."""
    paths = sorted(Path(p) for p in files) if files else sorted(EVAL_DIR.rglob("*.py"))
    h = hashlib.sha256()
    for p in paths:
        try:
            h.update(p.relative_to(BASE_DIR).as_posix().encode("utf-8"))
        except ValueError:
            h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return {"code_sha256": h.hexdigest(), "files": len(paths), **git_info()}


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return p.as_posix()


def file_entry(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    return {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def write_json(path: Path | str, data: Dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return p


def manifest_path_for(dataset_path: Path | str) -> Path:
    p = Path(dataset_path)
    return p.with_name(p.stem + ".manifest.json")


def dataset_manifest(output: Path | str, sources: Iterable[Path | str], *, generator: str, seed: Optional[int],
                     counts: Dict[str, Any], params: Dict[str, Any], created: Optional[str] = None,
                     code_files: Iterable[Path | str] | None = None) -> Dict[str, Any]:
    return {
        "dataset": file_entry(output),
        "sources": [file_entry(s) for s in sources],
        "generator": generator,
        "seed": seed,
        "params": params,
        "counts": counts,
        "code_version": code_version(code_files),
        "created": created or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "llm_calls": 0,
    }


def package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def run_manifest(*, timestamp: str, roles: Dict[str, Any], dataset_path: Optional[Path | str], mode: str,
                 use_cached: bool, seed: Optional[int], args: Dict[str, Any], judge: Optional[Dict[str, Any]] = None,
                 extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Manifest block embedded in every report (timestamp is passed in, not taken here)."""
    ds = None
    if dataset_path:
        p = Path(dataset_path)
        ds = {"path": rel(p), "sha256": sha256_file(p) if p.exists() else None}
    return {
        "timestamp": timestamp,
        "models": roles,
        "judge": judge,
        "dataset": ds,
        "mode": mode,
        "use_cached": use_cached,
        "seed": seed,
        "git": git_info(),
        "code_sha256": code_version()["code_sha256"],
        "deepeval_version": package_version("deepeval"),
        "groq_sdk_version": package_version("groq"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "args": args,
        **(extra or {}),
    }
