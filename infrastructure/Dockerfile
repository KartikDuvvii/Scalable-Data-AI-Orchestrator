# Stage 1: Build Dependencies
FROM python:3.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# Stage 2: Spark Environment (Final Image)
FROM bitnami/spark:3.5.0

USER root

WORKDIR /app

# Install Python dependencies from the builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code and other artifacts
COPY src/ /app/src/
COPY README.md /app/

# Environment Variables for Spark
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PYTHONPATH=/app/src

# Expose Spark UI (optional)
EXPOSE 4040

# Entry point for the pipeline (example)
# ENTRYPOINT ["spark-submit", "/app/src/pipelines/orchestrator.py"]
CMD ["tail", "-f", "/dev/null"]
