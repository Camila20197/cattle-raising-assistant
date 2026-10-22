#!/bin/sh
# Container entry point. One image, three roles, chosen with APP_MODE:
#   api (default) -> FastAPI          ui -> chat (Streamlit)          dashboard -> monitoring
# PORT is set by Cloud Run; locally it defaults to each service's usual port.
set -e

case "${APP_MODE:-api}" in
  api)
    exec uv run uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
    ;;
  ui)
    exec uv run python -m streamlit run src/ui/chat_app.py \
      --server.address 0.0.0.0 --server.port "${PORT:-8502}" --server.headless true
    ;;
  dashboard)
    exec uv run python -m streamlit run src/monitoring/dashboard.py \
      --server.address 0.0.0.0 --server.port "${PORT:-8501}" --server.headless true
    ;;
  *)
    echo "Unknown APP_MODE '${APP_MODE}'. Use: api, ui or dashboard." >&2
    exit 1
    ;;
esac
