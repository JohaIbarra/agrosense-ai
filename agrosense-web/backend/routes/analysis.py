"""
Rutas del API de análisis
"""
import os
import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from utils.validators import validate_dataset
from services.pipeline import run_pipeline

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_and_analyze(request: Request, file: UploadFile = File(...)):
    """
    Recibe un archivo .xlsx o .csv, lo valida y ejecuta el pipeline ML completo.
    Devuelve todos los resultados en formato JSON.
    """
    # Validar extensión
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in (".xlsx", ".csv"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .xlsx o .csv")

    # Leer archivo
    contents = await file.read()
    try:
        if ext == ".xlsx":
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo: {str(e)}")

    # Validar estructura
    valid, errors = validate_dataset(df)
    if not valid:
        return JSONResponse(
            status_code=422,
            content={"valid": False, "errors": errors, "data": None}
        )

    # Ejecutar pipeline
    predictor = request.app.state.predictor
    try:
        results = run_pipeline(df, predictor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el pipeline: {str(e)}")

    return {
        "valid":    True,
        "filename": filename,
        "errors":   [],
        "data":     results
    }


@router.get("/model-info")
async def model_info(request: Request):
    """Retorna métricas del modelo y estadísticas del dataset de entrenamiento."""
    predictor = request.app.state.predictor
    return {
        "metrics":          predictor.metrics,
        "dataset_stats":    predictor.dataset_stats,
        "feature_importance": predictor.feature_importance[:10],
    }


@router.get("/status")
async def status(request: Request):
    predictor = request.app.state.predictor
    return {"models_loaded": predictor.loaded, "status": "ok"}
