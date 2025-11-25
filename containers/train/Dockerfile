# Build stage
FROM python:3.11-slim-bullseye AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-install-project

# Runtime stage
FROM python:3.11-slim-bullseye

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the installed dependencies from builder (exclude unnecessary files)
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Copy entrypoint script
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Set PATH to use venv
ENV PATH="/app/.venv/bin:$PATH"

# Use entrypoint script to pass arguments flexibly
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []