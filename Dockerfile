# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Install in development mode
RUN pip install -e .

# Create directory for logs
RUN mkdir -p logs

# Download external modules
RUN python scripts/download_modules.py

# Expose port if needed (for future web interface)
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["python", "main.py"]