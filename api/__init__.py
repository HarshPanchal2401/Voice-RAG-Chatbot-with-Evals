"""
API Package
===========
Modular FastAPI routers and helpers.
"""

from api.routes_system import router as system_router
from api.routes_query import router as query_router
from api.routes_stream import router as stream_router
from api.routes_retrieval import router as retrieval_router
from api.routes_evaluation import router as evaluation_router

__all__ = [
    "system_router",
    "query_router",
    "stream_router",
    "retrieval_router",
    "evaluation_router",
]
