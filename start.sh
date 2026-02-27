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

ENV_DIR="env"

if [ ! -d "${ENV_DIR}" ]; then
  echo "Creating virtualenv at ${ENV_DIR}..."
  if ! python3 -m venv "${ENV_DIR}"; then
    echo "Failed to create virtualenv. Make sure 'python3-venv' or 'python3-full' is installed on this server."
    exit 1
  fi
fi

# Activate the virtualenv so python3/pip3 refer to it
# shellcheck disable=SC1091
. "${ENV_DIR}/bin/activate"

export DJANGO_SETTINGS_MODULE="ai4ch.settings"
export PYTHONUNBUFFERED=1
export PIP_BREAK_SYSTEM_PACKAGES=1

echo "Using Python interpreter: $(command -v python3)"
echo "Starting ${APP_NAME} (runserver) on port ${PORT}..."

echo "Upgrading pip and setuptools in the virtualenv..."
pip3 install --upgrade pip setuptools

if [ -f "requirements.txt" ]; then
  echo "Installing Python requirements from requirements.txt using ${PIP_BIN}..."
  pip3 install -r requirements.txt
fi

if ! command -v pm2 >/dev/null 2>&1; then
  echo "Error: pm2 is not installed or not on PATH."
  echo "Install with: npm install -g pm2"
  exit 1
fi

echo "Stopping existing PM2 process (if any)..."
pm2 delete "${APP_NAME}" >/dev/null 2>&1 || true

pm2 start "python3 manage.py runserver 0.0.0.0:${PORT}" --name "${APP_NAME}"

echo "Use 'pm2 logs ${APP_NAME}' to see logs and 'pm2 stop ${APP_NAME}' to stop."

