# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Requirements are in ml_component
COPY ml_component/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and ML component
COPY backend/ ./backend/
COPY ml_component/ ./ml_component/

# Create directories for uploads and reports
RUN mkdir -p backend/uploads backend/reports

# Expose the port Hugging Face expects (7860)
EXPOSE 7860

# Command to run the application using gunicorn for production stability
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860", "backend.main:app"]
