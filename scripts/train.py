import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Explicitly set the MLflow tracking URI to a relative path
tracking_uri = "file://" + os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(tracking_uri)
logging.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

# Set the experiment name
mlflow.set_experiment("Iris Rebuild Experiment")

def train_and_log_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Trains a given model and logs its parameters, metrics, and the model artifact
    to MLflow.
    """
    with mlflow.start_run():
        logging.info(f"Starting MLflow run for {model_name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate the accuracy metric
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model artifact
        mlflow.sklearn.log_model(model, "model")

        logging.info(f"Finished run for {model_name}. Accuracy: {accuracy:.4f}")
        logging.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    logging.info("Loading and splitting the Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and track Logistic Regression
    logging.info("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=200, solver='liblinear')
    train_and_log_model(lr_model, X_train, X_test, y_train, y_test, "LogisticRegression")
    
    # Train and track Random Forest
    logging.info("Training Random Forest Classifier model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_log_model(rf_model, X_train, X_test, y_train, y_test, "RandomForestClassifier")
    
    logging.info("Training process complete.")