"""
Services Package
================
Application-level services for voice (STT / TTS) and DeepEval evaluation.

Exports are resolved lazily (PEP 562) so importing `services.voice_service` does not pull in
DeepEval and the evaluation stack.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "VoiceService": "services.voice_service",
    "DeepEvalEvaluator": "services.evaluation_service",
    "GroqDeepEvalModel": "services.evaluation_service",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'services' has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
