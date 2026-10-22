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

# One image, several roles: APP_MODE=api (default), ui or dashboard.
# start.sh reads PORT, which Cloud Run sets automatically.
CMD ["sh", "start.sh"]
