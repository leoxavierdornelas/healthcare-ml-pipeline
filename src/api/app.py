from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("models/xgboost_lung_cancer.pkl")

app = FastAPI(title="Lung Cancer Prediction API", version="1.0.0")

class PatientInput(BaseModel):
    GENDER: int
    AGE: int
    SMOKING: int
    YELLOW_FINGERS: int
    ANXIETY: int
    PEER_PRESSURE: int
    CHRONIC_DISEASE: int
    FATIGUE: int
    ALLERGY: int
    WHEEZING: int
    ALCOHOL_CONSUMING: int
    COUGHING: int
    SHORTNESS_OF_BREATH: int
    SWALLOWING_DIFFICULTY: int
    CHEST_PAIN: int

@app.post("/predict")
def predict(patient: PatientInput):
    smoking_age = patient.SMOKING * patient.AGE
    
    sintomas = (
        patient.YELLOW_FINGERS + patient.ANXIETY + patient.PEER_PRESSURE +
        patient.CHRONIC_DISEASE + patient.FATIGUE + patient.ALLERGY +
        patient.WHEEZING + patient.ALCOHOL_CONSUMING + patient.COUGHING +
        patient.SHORTNESS_OF_BREATH + patient.SWALLOWING_DIFFICULTY +
        patient.CHEST_PAIN
    )
    
    risk_score = (
        patient.SMOKING * 2 +
        patient.AGE / 10 +
        patient.CHRONIC_DISEASE * 2 +
        patient.WHEEZING * 1.5 +
        patient.COUGHING * 1.5 +
        patient.SHORTNESS_OF_BREATH * 2
    )
    
    # Regra médica: se tem mais de 8 sintomas, força CANCER
    if sintomas >= 8:
        prediction = 1  # CANCER
        probability = 0.95
    else:
        features = np.array([[
            patient.GENDER, patient.AGE, patient.SMOKING,
            patient.YELLOW_FINGERS, patient.ANXIETY, patient.PEER_PRESSURE,
            patient.CHRONIC_DISEASE, patient.FATIGUE, patient.ALLERGY,
            patient.WHEEZING, patient.ALCOHOL_CONSUMING, patient.COUGHING,
            patient.SHORTNESS_OF_BREATH, patient.SWALLOWING_DIFFICULTY,
            patient.CHEST_PAIN, smoking_age, sintomas, risk_score
        ]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][prediction]
    
    return {
        "prediction": "CANCER" if prediction == 1 else "NO_CANCER",
        "probability": float(probability),
        "confidence": f"{probability*100:.2f}%"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost Lung Cancer"}

@app.get("/")
def info():
    return {
        "title": "Lung Cancer Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Saude da API",
            "/predict": "Prever risco de cancer de pulmao (POST)"
        }
    }
