# Specifying base image
FROM python:3.10

# Specifying working directory
WORKDIR /test-technique-vif

# Copying the whole project
COPY . .

# Installing requirements
RUN pip install -r requirements.txt

# Changing working directory
WORKDIR /test-technique-vif/test_technique_vif

# Set the mlflow tracking URI
ENV MLFLOW_TRACKING_URI https://dagshub.com/zerippeur/test-technique-vif.mlflow
ENV MLFLOW_TRACKING_USERNAME zerippeur
ENV MLFLOW_TRACKING_PASSWORD paste_token_here

# Run the model fitting script
CMD ["python", "model_fit.py"]