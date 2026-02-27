#!/usr/bin/env bash

# Production deployment script for AI4CH
# - Runs Django on port 8090 under PM2 using gunicorn
# - Intended to sit behind an HTTPS reverse proxy (e.g. nginx) for:
#   https://portal.ai4crophealth.or.tz

set -euo pipefail

APP_NAME="ai4ch-portal"
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
  if ! python3 -m venv --system-site-packages "${ENV_DIR}"; then
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

echo "Using DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE}"
echo "Using Python interpreter: $(command -v python3)"
echo "Starting ${APP_NAME} on port ${PORT} (behind https://portal.ai4crophealth.or.tz)"

# echo "Upgrading pip and setuptools in the virtualenv..."
# pip3 install --upgrade pip setuptools

if [ -f "requirements.txt" ]; then
  echo "Installing Python requirements from requirements.txt using pip3..."
  pip3 install -r requirements.txt
fi

echo "Applying migrations..."
python3 manage.py migrate --noinput

echo "Collecting static files..."
python3 manage.py collectstatic --noinput

echo "Stopping existing PM2 process (if any)..."
if command -v pm2 >/dev/null 2>&1; then
  pm2 delete "${APP_NAME}" >/dev/null 2>&1 || true
else
  echo "Error: pm2 is not installed or not on PATH."
  echo "Install with: npm install -g pm2"
  exit 1
fi

if ! command -v gunicorn >/dev/null 2>&1; then
  echo "Error: gunicorn is not installed in the virtualenv."
  echo "Install it with: pip3 install gunicorn"
  exit 1
fi

echo "Starting gunicorn under PM2..."
pm2 start "gunicorn ai4ch.wsgi:application --bind 0.0.0.0:${PORT} --workers 3" --name "${APP_NAME}"

echo "Saving PM2 process list (for pm2 resurrect)..."
pm2 save

echo "Deployment complete. Ensure your reverse proxy forwards https://portal.ai4crophealth.or.tz to 127.0.0.1:${PORT}."

