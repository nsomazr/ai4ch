#!/usr/bin/env bash

# Development startup script for AI4CH
# - Runs Django development server on port 8090 under PM2

set -euo pipefail

APP_NAME="ai4ch-dev"
PORT=8090

cd "$(dirname "$0")"

if [ -f "config/.env" ]; then
  echo "Loading environment variables from config/.env"
  set -a
  # shellcheck disable=SC1091
  . "config/.env"
  set +a
fi

if [ -n "${VENV_PATH:-}" ]; then
  echo "Using virtualenv at ${VENV_PATH}"
  if [ ! -d "${VENV_PATH}" ]; then
    echo "Virtualenv not found, creating with python3 -m venv \"${VENV_PATH}\"..."
    python3 -m venv "${VENV_PATH}"
  fi
  PYTHON_BIN="${VENV_PATH}/bin/python3"
else
  PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
fi

export DJANGO_SETTINGS_MODULE="ai4ch.settings"
export PYTHONUNBUFFERED=1

echo "Using Python interpreter: ${PYTHON_BIN}"
echo "Starting ${APP_NAME} (runserver) on port ${PORT}..."

if [ -f "requirements.txt" ]; then
  echo "Installing Python requirements from requirements.txt..."
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

if ! command -v pm2 >/dev/null 2>&1; then
  echo "Error: pm2 is not installed or not on PATH."
  echo "Install with: npm install -g pm2"
  exit 1
fi

echo "Stopping existing PM2 process (if any)..."
pm2 delete "${APP_NAME}" >/dev/null 2>&1 || true

pm2 start "\"${PYTHON_BIN}\" manage.py runserver 0.0.0.0:${PORT}" --name "${APP_NAME}"

echo "Use 'pm2 logs ${APP_NAME}' to see logs and 'pm2 stop ${APP_NAME}' to stop."

