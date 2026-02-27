#!/usr/bin/env bash

# Development startup script for AI4CH
# - Runs Django development server on port 8090 under PM2

set -euo pipefail

APP_NAME="ai4ch-dev"
PORT=8090

cd "$(dirname "$0")"

export DJANGO_SETTINGS_MODULE="ai4ch.settings"
export PYTHONUNBUFFERED=1

echo "Starting ${APP_NAME} (runserver) on port ${PORT}..."

if ! command -v pm2 >/dev/null 2>&1; then
  echo "Error: pm2 is not installed or not on PATH."
  echo "Install with: npm install -g pm2"
  exit 1
fi

echo "Stopping existing PM2 process (if any)..."
pm2 delete "${APP_NAME}" >/dev/null 2>&1 || true

pm2 start "python manage.py runserver 0.0.0.0:${PORT}" --name "${APP_NAME}"

echo "Use 'pm2 logs ${APP_NAME}' to see logs and 'pm2 stop ${APP_NAME}' to stop."

