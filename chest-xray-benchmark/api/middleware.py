"""
Middleware — Chest X-ray API
==============================
CORS, logging, and error handling middleware.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def setup_cors(app: FastAPI) -> None:
    """Add CORS middleware allowing all origins.

    Parameters
    ----------
    app : FastAPI
        FastAPI application instance.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log request method, path, and response time.

    Parameters
    ----------
    request : Request
        Incoming request.
    call_next : Callable
        Next middleware / route handler.

    Returns
    -------
    Response
        The response.
    """
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path} → "
        f"{response.status_code} ({elapsed_ms:.1f}ms)"
    )

    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global error handler.

    Parameters
    ----------
    request : Request
        Incoming request.
    exc : Exception
        Raised exception.

    Returns
    -------
    JSONResponse
        JSON error response.
    """
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
