from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

logger = logging.getLogger(__name__)

def setup_middleware(app):
    """Configure middleware for Unreal Engine compatibility"""
    
    # CORS for Unreal Engine HTTP requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins including UE
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-Request-ID"]
    )
    
    # Performance monitoring middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"{request.url.path} - {process_time:.3f}s")
        return response
    
    # Request ID middleware for tracking
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(time.time()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response