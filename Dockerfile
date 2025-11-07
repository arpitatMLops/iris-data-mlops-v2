# Base image - Python 3.9 (lightweight)
FROM python:3.9-slim

# Set the working directory
WORKDIR /opt/ml

# Install basic system dependencies (build-essential is often needed by sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --upgrade pip && pip install -r /opt/ml/requirements.txt

# Copy all source code into the image
COPY src/ /opt/ml/

# Environment variables for clean logging
ENV PYTHONPATH=/opt/ml
ENV PYTHONUNBUFFERED=TRUE

# Define the entrypoint — SageMaker calls this
ENTRYPOINT ["python", "-m", "src.main"]

