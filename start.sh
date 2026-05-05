#!/bin/sh
# Production entrypoint — used by Railway, Render, Docker.
# 'sh' expands $PORT before gunicorn receives it.
exec gunicorn \
  --workers 1 \
  --threads 4 \
  --timeout 120 \
  --bind "0.0.0.0:${PORT:-8080}" \
  api:app
