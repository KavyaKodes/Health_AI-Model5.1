import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "Sleep_health.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# --- Load ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading files: {str(e)}")

# --- Request Model ---
class Request(BaseModel):
    Age: float
    Sleep_Duration: float
    Quality_of_Sleep: float
    Physical_Activity_Level: float
    Stress_Level: float
    Heart_Rate: float
    Daily_Steps: float

# --- Health ---
@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Predict ---
@app.post("/predict")
async def predict(request: Request):
    try:
        features = np.array([
            request.Age,
            request.Sleep_Duration,
            request.Quality_of_Sleep,
            request.Physical_Activity_Level,
            request.Stress_Level,
            request.Heart_Rate,
            request.Daily_Steps
        ]).reshape(1, -1)

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        # --- Handle both cases ---
        if isinstance(prediction, (np.integer, int)):
            # ⚠️ If encoded, map manually (FIX THIS BASED ON YOUR DATA)
            mapping = {
                0: "No Disorder",
                1: "Insomnia",
                2: "Sleep Apnea"
            }
            prediction = mapping.get(prediction, str(prediction))

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
