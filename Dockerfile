# --- Base image ---
# Slim Python 3.14 image to keep the image small.
FROM python:3.14-slim

# Show logs immediately and let `src.*` imports work for every entry point
# (API, monitoring dashboard and chat UI).
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# --- Package manager ---
RUN pip install --no-cache-dir uv

WORKDIR /app

# --- Dependencies first (cached layer while only the code changes) ---
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# --- Application code ---
COPY . .

EXPOSE 8000

# Shell form so ${PORT} can be overridden (e.g. by Cloud Run).
CMD uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
