# Build stage
FROM python:3.11-slim-bullseye AS builder

WORKDIR /app

# Install build dependencies only in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy and install dependencies
COPY pyproject.toml .
RUN uv sync

# Runtime stage
FROM python:3.11-slim-bullseye

WORKDIR /app

# Copy only the installed dependencies from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Set PATH to use venv
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["python", "src/sample.py"]