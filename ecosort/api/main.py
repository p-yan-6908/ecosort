"""EcoSort FastAPI Application.

This module provides the main FastAPI application with endpoints
for waste classification using the trained model.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging

from ecosort.api.routes import predict, health, batch, metrics
from ecosort.api.dependencies import load_model
from ecosort.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events.
    
    Loads the model on startup and cleans up on shutdown.
    """
    logger.info("Loading model...")
    try:
        config = Config.from_yaml("config.yaml")
        load_model(config)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="EcoSort API",
    description="Ontario Waste Classification API (2026 Standards). "
    "Upload images to classify waste into Blue Bin, Green Bin, Garbage, and more.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(batch.router, tags=["Batch Prediction"])
app.include_router(metrics.router, tags=["Metrics"])

# Mount static files
web_dir = Path(__file__).parent.parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get(
    "/",
    summary="API root",
    description="Returns a welcome message and links to the web interface.",
)
async def root():
    """Root endpoint with API information.
    
    Returns:
        Dict with welcome message and links.
    """
    return {
        "message": "EcoSort API is running. Visit /static/index.html for web interface.",
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "top_k": "/predict/top-k",
            "classes": "/classes",
            "health": "/health",
        },
    }
