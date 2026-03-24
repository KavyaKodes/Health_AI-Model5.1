import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize FastAPI app
app = FastAPI()

# --- Load Model and Prepare Scaler/LabelEncoder ---
# Define the path to the model and dataset
MODEL_PATH = "/content/Sleep_health.pkl"
DATASET_PATH = "/content/Sleep_health_and_lifestyle_dataset.csv"

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

# Define the feature columns used during training
FEATURE_COLUMNS = [
    'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
    'Stress Level', 'Heart Rate', 'Daily Steps'
]
TARGET_COLUMN = 'Sleep Disorder'

# Load the original dataset to fit the scaler and label encoder
try:
    df_original = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Dataset file not found at {DATASET_PATH}")

# Preprocess the original dataset to match training preprocessing
df_processed = df_original.dropna().copy() # .copy() to avoid SettingWithCopyWarning

# Initialize and fit the StandardScaler
scaler = StandardScaler()
scaler.fit(df_processed[FEATURE_COLUMNS])

# Initialize and fit the LabelEncoder for the target variable
label_encoder = LabelEncoder()
label_encoder.fit(df_processed[TARGET_COLUMN])

# --- Pydantic Request Model ---
class Request(BaseModel):
    Age: float
    Sleep_Duration: float
    Quality_of_Sleep: float
    Physical_Activity_Level: float
    Stress_Level: float
    Heart_Rate: float
    Daily_Steps: float

# --- FastAPI Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    """Predict sleep disorder based on input features"""
    try:
        # Convert input features from Pydantic model to a NumPy array
        features = np.array([
            request.Age,
            request.Sleep_Duration,
            request.Quality_of_Sleep,
            request.Physical_Activity_Level,
            request.Stress_Level,
            request.Heart_Rate,
            request.Daily_Steps
        ]).reshape(1, -1) # Reshape for single prediction

        # Scale the input features using the fitted scaler
        scaled_features = scaler.transform(features)

        # Make prediction using the loaded model
        prediction_encoded = model.predict(scaled_features)

        # Inverse transform the prediction to get the original label
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        return {"prediction": prediction_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# --- Uvicorn Run Configuration ---
if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
