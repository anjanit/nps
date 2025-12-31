# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies in batches with retries to avoid network issues
RUN pip install --no-cache-dir Flask && \
    pip install --no-cache-dir SQLAlchemy && \
    pip install --no-cache-dir Flask-SQLAlchemy && \
    pip install --no-cache-dir Flask-Migrate && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir matplotlib && \
    pip install --no-cache-dir scikit-learn && \
    pip install --no-cache-dir mysql-connector-python

# Copy application code
COPY . .

# Create directory for models if it doesn't exist
RUN mkdir -p models

# Expose port 5000 for Flask
EXPOSE 5000

# Run the application with gunicorn for production
RUN pip install gunicorn

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
