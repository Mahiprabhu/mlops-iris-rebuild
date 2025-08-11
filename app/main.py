import mlflow
import uvicorn
import logging
import os
import pickle
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import subprocess

# --- Prometheus metrics definitions ---
REQUEST_COUNT = Counter(
    "iris_api_requests_total",
    "Total number of prediction requests."
)

PREDICTION_TIME = Histogram(
    "iris_prediction_duration_seconds",
    "Duration of prediction requests in seconds."
)

# --- Logging Configuration ---
# Define the log file path inside the container
# LOG_FILE = "app.log"
LOG_FILE = "/app/logs/app.log"

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a rotating file handler to store logs
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=1024 * 1024 * 10,
    backupCount=5
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Add a console handler to show logs in the terminal
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


# --- Load the Model Directly from its specific artifact path ---
MODEL_PATH_IN_CONTAINER = "/app/mlruns/240103695117661407/models/m-45c328d3a9074c39a6ad1206a99d1b06/artifacts/"

try:
    # Construct the full path to the pickle file
    model_file_path = os.path.join(MODEL_PATH_IN_CONTAINER, "model.pkl")

    # Load the model directly using pickle
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Successfully loaded model from path '{model_file_path}'.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# --- Define our FastAPI application instance ---
app = FastAPI(
    title="Iris Flower Classifier API",
    description="A simple API to classify Iris flowers.",
    version="1.0.0"
)

# --- Pydantic model for request body validation ---
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- API Endpoints ---

@app.get("/")
def home():
    """Returns a welcome message."""
    return {"message": "Welcome to the Iris Classifier API. Use the /predict endpoint."}

# /metrics endpoint for monitoring
@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest().decode("utf-8")

@app.post("/retrain")
def retrain():
    try:
        # We run the training script as a subprocess
        result = subprocess.run(
            ["python", "scripts/train.py"],
            capture_output=True,
            text=True,
            check=True
        )
        return {"message": "Model retraining triggered successfully."}
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed: {e.stderr}")
        return {"error": "Model retraining failed."}, 500
    
# /predict endpoint with metrics tracking
@app.post("/predict")
@PREDICTION_TIME.time()
def predict(features: IrisFeatures):
    REQUEST_COUNT.inc()
    """
    Accepts a JSON payload with iris features and returns a prediction.
    """
    if model is None:
        logger.error("Model not loaded. Cannot make a prediction.")
        return {"error": "Model not available"}, 500

    # Convert the Pydantic object into a format our model expects
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]

    logger.info(f"Received prediction request with features: {input_data}")

    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data).tolist()[0]
        
        prediction_labels = ["setosa", "versicolor", "virginica"]
        prediction_label = prediction_labels[prediction]

        response = {
            "prediction": prediction_label,
            "probabilities": {
                "setosa": prediction_proba[0],
                "versicolor": prediction_proba[1],
                "virginica": prediction_proba[2]
            }
        }
        
        logger.info(f"Prediction result: {response}")
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}, 500