FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code AND the MLflow artifacts
COPY app /app/app
COPY mlruns /app/mlruns

# Create a log file and ensure permissions
# RUN touch /app/app.log && chmod 666 /app/app.log
RUN mkdir -p /app/logs && touch /app/logs/app.log && chmod 666 /app/logs/app.log

# Expose the port the API will run on
EXPOSE 8000

# Command that runs the application when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


