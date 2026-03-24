import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# --- Load Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# --- Load Artifacts ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model artifacts: {str(e)}")

# --- Pydantic Request Model ---
class Request(BaseModel):
    Age: float
    Sleep_Duration: float
    Quality_of_Sleep: float
    Physical_Activity_Level: float
    Stress_Level: float
    Heart_Rate: float
    Daily_Steps: float

# --- Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(request: Request):
    try:
        # Convert input to numpy array (order MUST match training)
        features = np.array([
            request.Age,
            request.Sleep_Duration,
            request.Quality_of_Sleep,
            request.Physical_Activity_Level,
            request.Stress_Level,
            request.Heart_Rate,
            request.Daily_Steps
        ]).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(features)

        # Predict
        prediction_encoded = model.predict(scaled_features)

        # Decode label
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        return {"prediction": prediction_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
