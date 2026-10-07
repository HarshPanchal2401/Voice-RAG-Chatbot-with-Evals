"""
Feedback Routes
===============
User thumbs-up / thumbs-down on an answer, attached to its LangSmith trace when tracing is on.
"""

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from core.security import PROTECTED_LIMITED
from core.tracing import tracing_enabled, _get_ls_client
from schemas.requests import FeedbackRequest
from schemas.responses import FeedbackResponse

router = APIRouter()

FEEDBACK_KEY = "user_rating"


def _log_feedback(trace_id: str, score: int, comment):
    client = _get_ls_client()
    if client is None:
        return False
    client.create_feedback(
        run_id=trace_id,
        key=FEEDBACK_KEY,
        score=float(score),
        comment=comment or None,
        source_info={"source": "web_ui"},
    )
    return True


@router.post("/api/v1/feedback", response_model=FeedbackResponse, tags=["Feedback"], dependencies=PROTECTED_LIMITED)
async def submit_feedback(payload: FeedbackRequest):
    """
    Records a user rating (`score` 1 = good, 0 = bad) for the answer whose `trace_id` came with it.
    `logged` is true only when the rating was written to LangSmith (tracing on and a trace id given).
    """
    logged = False
    if payload.trace_id and tracing_enabled():
        try:
            logged = await run_in_threadpool(_log_feedback, payload.trace_id, payload.score, payload.comment)
        except Exception as e:
            print(f"⚠️ [LangSmith] Failed to log user feedback for {payload.trace_id}: {e}")
            logged = False
    else:
        print(f"ℹ️ [feedback] score={payload.score} trace_id={payload.trace_id} (not logged: tracing off or no trace id)")
    return FeedbackResponse(status="ok", logged=logged)
