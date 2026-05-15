"""
AgroSense AI — Backend FastAPI
Servicio de análisis ecológico y predicción ML
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analysis import router as analysis_router
from ml.predictor import ModelPredictor

app = FastAPI(
    title="AgroSense AI API",
    description="Plataforma de análisis ecológico y ML predictivo para restauración ambiental",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelos al arrancar
@app.on_event("startup")
async def startup_event():
    print("Cargando modelos ML...")
    predictor = ModelPredictor()
    predictor.load()
    app.state.predictor = predictor
    print("Modelos cargados OK.")

app.include_router(analysis_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ok", "message": "AgroSense AI API v1.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": hasattr(app.state, "predictor")}
