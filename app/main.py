import mlflow
import uvicorn
import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load the Model from MLflow Model Registry ---

# --- Load the Model Directly from the run's artifacts ---
# RUN_ID = "749318566ed94622822a31878fbcb10a" 

# try:
#     model_uri = f"runs:/{RUN_ID}/model"
#     model = mlflow.sklearn.load_model(model_uri=model_uri)
#     logger.info(f"Successfully loaded model from run '{RUN_ID}'.")
# except Exception as e:
#     logger.error(f"Error loading model: {e}")
#     model = None


# MODEL_NAME = "IrisClassifier"
# STAGE = "Production" 

# try:
#     model = mlflow.sklearn.load_model(model_uri=f"models:/{MODEL_NAME}/{STAGE}")
#     logger.info(f"Successfully loaded model '{MODEL_NAME}' in '{STAGE}' stage.")
# except Exception as e:
#     logger.error(f"Error loading model: {e}")
#     model = None

EXPERIMENT_ID = "240103695117661407"
RUN_ID = "749318566ed94622822a31878fbcb10a"  

# Construct the full, absolute path inside the container
# MODEL_PATH_IN_CONTAINER = f"/app/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"

MODEL_PATH_IN_CONTAINER = f"/app/mlruns/models"

try:
    # We tell MLflow to load the model from this specific directory.
    model = mlflow.sklearn.load_model(model_uri=MODEL_PATH_IN_CONTAINER)
    logger.info(f"Successfully loaded model from path '{MODEL_PATH_IN_CONTAINER}'.")
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

@app.post("/predict")
def predict(features: IrisFeatures):
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