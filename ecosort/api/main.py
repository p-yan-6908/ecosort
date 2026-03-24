"""EcoSort FastAPI Application"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging

from ecosort.api.routes import predict, health
from ecosort.api.dependencies import load_model
from ecosort.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    description="Ontario Waste Classification API (2026 Standards)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])

web_dir = Path(__file__).parent.parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/")
async def root():
    return {
        "message": "EcoSort API is running. Visit /static/index.html for web interface."
    }
