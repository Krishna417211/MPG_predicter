"""
Car MPG Predictor API
Run: uvicorn main:app --reload
"""

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load the saved model, scaler and polynomial features (already trained)
model = joblib.load("lasso_model.pkl")
scaler = joblib.load("std_scaler.pkl")
poly = joblib.load("poly_feats.pkl")

# Feature order the model was trained on (do not change)
FEATURE_ORDER = [
    "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin",
]

app = FastAPI(title="MPG Predictor API")


class Car(BaseModel):
    cylinders: int
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: int
    origin: int


@app.get("/")
def root():
    return {"message": "MPG Predictor API. Go to /docs to try it."}


@app.post("/predict")
def predict(car: Car):
    data = car.model_dump()
    row = np.array([[data[f] for f in FEATURE_ORDER]])
    poly_features = poly.transform(scaler.transform(row))
    prediction = model.predict(poly_features)[0]
    return {"predicted_mpg": round(float(prediction), 2)}