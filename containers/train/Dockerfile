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
RUN uv sync --no-dev

# Runtime stage
FROM python:3.11-slim-bullseye

WORKDIR /app

# Copy only the installed dependencies from builder (exclude unnecessary files)
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Set PATH to use venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Clean up pip cache and unnecessary files
RUN find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyc" -delete && \
    find /app/.venv -type f -name "*.pyo" -delete && \
    find /app/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true

# Run the application
CMD ["python", "src/run_tuning.py"]