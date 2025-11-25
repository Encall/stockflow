# Build stage
FROM python:3.11-slim-bullseye AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Runtime stage
FROM python:3.11-slim-bullseye

WORKDIR /app

# Copy application code
COPY app ./app

# Copy installed dependencies from builder and ensure they're on PATH
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# uvicorn app.main:app --host 0.0.0.0 --port 8000
