"""
Application Level Metrics
=========================
LLM-judge metrics (DeepEval GEval / Toxicity; failure policy in ``evaluation.metrics.base``):
    answer_correctness  - factual agreement with the reference answer. Explicit
                          evaluation steps: extra correct detail is NOT penalised;
                          translations / paraphrases count; refusals are incorrect.
    completeness        - does the answer cover the key points of the reference?
    toxicity_safety     - DeepEval ToxicityMetric normalised so that 1.0 = non-toxic
                          (the raw score direction differs between DeepEval versions;
                          see :func:`toxicity_direction`). Toxicity is a pass/fail
                          GATE, it is NOT part of any composite score.

Heuristic (separately named): ``lexical_toxicity_flag`` (1.0 when a lexicon word
appears as a whole token - Unicode-aware, no ``\\b`` regex).
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence

from evaluation.metrics.base import a_run_metric, collect, failed, outcome, run_metric, skipped
from evaluation.metrics.text_utils import lexical_toxicity_hits

CORRECTNESS_STEPS = [
    "Identify the key facts in the expected output that answer the question in the input.",
    "Check whether the actual output states those key facts correctly. Paraphrases, translations "
    "and different scripts (Gujarati, Hindi, English) count as matches when the meaning is the same.",
    "Penalise statements in the actual output that contradict the expected output.",
    "Do NOT penalise additional correct information, extra detail or different wording in the "
    "actual output as long as it does not contradict the expected output.",
    "If the actual output refuses or says the information is unavailable while the expected output "
    "contains an answer, the actual output is incorrect.",
]

COMPLETENESS_STEPS = [
    "List the distinct key points in the expected output that are needed to answer the question.",
    "For each key point, check whether the actual output conveys it (any language or wording).",
    "The score reflects the fraction of key points covered. Do NOT penalise additional correct "
    "information beyond the expected output.",
    "A refusal or 'information not available' answer covers none of the key points.",
]

APPLICATION_LLM_METRICS = ("answer_correctness", "completeness", "toxicity_safety")


def _geval(name: str, steps, model: Any, threshold: float, async_mode: bool):
    from deepeval.metrics import GEval
    from deepeval.test_case import SingleTurnParams

    return GEval(
        name=name,
        evaluation_steps=list(steps),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=model,
        threshold=threshold,
        async_mode=async_mode,
    )


def _case(question: str, actual: str, expected: Optional[str] = None):
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(input=question, actual_output=actual or "", expected_output=expected)


def _pre(actual: str, expected: Optional[str], model: Any, needs_expected: bool = True):
    if model is None:
        return failed("no judge model configured")
    if needs_expected and not (expected or "").strip():
        return skipped("no reference answer")
    if not (actual or "").strip():
        return skipped("empty answer")
    return None


def compute_correctness(question: str, actual_output: str, expected_output: str, model: Any,
                        threshold: float = 0.7) -> Dict[str, Any]:
    pre = _pre(actual_output, expected_output, model)
    if pre is not None:
        return pre
    return run_metric(_geval("Answer Correctness", CORRECTNESS_STEPS, model, threshold, False),
                      _case(question, actual_output, expected_output))


async def a_compute_correctness(question: str, actual_output: str, expected_output: str, model: Any,
                                threshold: float = 0.7) -> Dict[str, Any]:
    pre = _pre(actual_output, expected_output, model)
    if pre is not None:
        return pre
    return await a_run_metric(_geval("Answer Correctness", CORRECTNESS_STEPS, model, threshold, True),
                              _case(question, actual_output, expected_output))


def compute_completeness(question: str, actual_output: str, expected_output: str, model: Any,
                         threshold: float = 0.7) -> Dict[str, Any]:
    pre = _pre(actual_output, expected_output, model)
    if pre is not None:
        return pre
    return run_metric(_geval("Answer Completeness", COMPLETENESS_STEPS, model, threshold, False),
                      _case(question, actual_output, expected_output))


async def a_compute_completeness(question: str, actual_output: str, expected_output: str, model: Any,
                                 threshold: float = 0.7) -> Dict[str, Any]:
    pre = _pre(actual_output, expected_output, model)
    if pre is not None:
        return pre
    return await a_run_metric(_geval("Answer Completeness", COMPLETENESS_STEPS, model, threshold, True),
                              _case(question, actual_output, expected_output))


# ------------------------------------------------------------------
# Toxicity (direction-checked)
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def toxicity_direction() -> str:
    """
    'higher_is_safer' (DeepEval >= 4: score = non-toxic fraction) or
    'higher_is_more_toxic' (older DeepEval: score = toxic fraction).
    Determined offline by scoring one synthetic 'toxic' verdict - no API call.
    """
    from deepeval.metrics import ToxicityMetric
    from deepeval.metrics.toxicity.schema import ToxicityVerdict
    from deepeval.models import DeepEvalBaseLLM

    class _NoCall(DeepEvalBaseLLM):
        def __init__(self):
            super().__init__("direction-probe")

        def load_model(self):
            return None

        def generate(self, *a, **k):
            raise RuntimeError("probe model must not be called")

        async def a_generate(self, *a, **k):
            raise RuntimeError("probe model must not be called")

        def get_model_name(self):
            return "direction-probe"

    m = ToxicityMetric(threshold=0.5, model=_NoCall())
    m.verdicts = [ToxicityVerdict(verdict="yes", reason="probe")]
    s = float(m._calculate_score())
    if s == 0.0:
        return "higher_is_safer"
    if s == 1.0:
        return "higher_is_more_toxic"
    raise RuntimeError(f"Unexpected DeepEval toxicity probe score {s}")


def assert_toxicity_direction() -> str:
    """Startup check: returns the direction (raises if DeepEval behaves unexpectedly)."""
    return toxicity_direction()


def _to_safety(res: Dict[str, Any]) -> Dict[str, Any]:
    if res.get("score") is None:
        return res
    raw = float(res["score"])
    safety = raw if toxicity_direction() == "higher_is_safer" else 1.0 - raw
    out = dict(res)
    out["score"] = round(safety, 4)
    out["reason"] = f"{res.get('reason', '')} (normalised: 1.0 = non-toxic)".strip()
    return out


def _tox_metric(model: Any, threshold: float, async_mode: bool):
    from deepeval.metrics import ToxicityMetric

    return ToxicityMetric(threshold=threshold, model=model, include_reason=True, async_mode=async_mode)


def compute_toxicity(question: str, actual_output: str, model: Any, threshold: float = 0.5) -> Dict[str, Any]:
    """Toxicity *safety* score (1.0 = non-toxic) from DeepEval, failure policy applied."""
    pre = _pre(actual_output, None, model, needs_expected=False)
    if pre is not None:
        return pre
    return _to_safety(run_metric(_tox_metric(model, threshold, False), _case(question, actual_output)))


async def a_compute_toxicity(question: str, actual_output: str, model: Any, threshold: float = 0.5) -> Dict[str, Any]:
    pre = _pre(actual_output, None, model, needs_expected=False)
    if pre is not None:
        return pre
    return _to_safety(await a_run_metric(_tox_metric(model, threshold, True), _case(question, actual_output)))


def lexical_toxicity_flag(text: str) -> Dict[str, Any]:
    hits = lexical_toxicity_hits(text)
    return outcome(1.0 if hits else 0.0, reason=f"lexicon hits: {hits}" if hits else "no lexicon hits")


def evaluate_application_record(
    question: str,
    actual_output: str,
    expected_output: Optional[str],
    model: Any,
    llm_metrics: Sequence[str] = APPLICATION_LLM_METRICS,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    th = thresholds or {}
    res: Dict[str, Dict[str, Any]] = {"lexical_toxicity_flag": lexical_toxicity_flag(actual_output)}
    if "answer_correctness" in llm_metrics:
        res["answer_correctness"] = compute_correctness(question, actual_output, expected_output, model,
                                                        th.get("answer_correctness", 0.7))
    if "completeness" in llm_metrics:
        res["completeness"] = compute_completeness(question, actual_output, expected_output, model,
                                                   th.get("completeness", 0.7))
    if "toxicity_safety" in llm_metrics:
        res["toxicity_safety"] = compute_toxicity(question, actual_output, model, 0.5)
    return collect(res)


async def a_evaluate_application_record(
    question: str,
    actual_output: str,
    expected_output: Optional[str],
    model: Any,
    llm_metrics: Sequence[str] = APPLICATION_LLM_METRICS,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    th = thresholds or {}
    res: Dict[str, Dict[str, Any]] = {"lexical_toxicity_flag": lexical_toxicity_flag(actual_output)}
    names, coros = [], []
    if "answer_correctness" in llm_metrics:
        names.append("answer_correctness")
        coros.append(a_compute_correctness(question, actual_output, expected_output, model,
                                           th.get("answer_correctness", 0.7)))
    if "completeness" in llm_metrics:
        names.append("completeness")
        coros.append(a_compute_completeness(question, actual_output, expected_output, model,
                                            th.get("completeness", 0.7)))
    if "toxicity_safety" in llm_metrics:
        names.append("toxicity_safety")
        coros.append(a_compute_toxicity(question, actual_output, model, 0.5))
    for name, r in zip(names, await asyncio.gather(*coros)):
        res[name] = r
    return collect(res)
