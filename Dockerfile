# --- Imagen base ---

# Use a slim version of Python 3.14 as the base image to keep the image size small.   
FROM python:3.14-slim


# --- Set environment variables ---
RUN pip install --no-cache-dir uv

# -- Set the working directory inside the container ---
WORKDIR /app

# --- Copy dependency files ---
COPY pyproject.toml uv.lock ./

# --- Install dependencies ---
RUN uv sync --frozen --no-dev

# --- Copy the rest of the application code ---
COPY . .

# --- Port that exposes the app ---
EXPOSE 8000

# --- Command to run the app ---
#CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
