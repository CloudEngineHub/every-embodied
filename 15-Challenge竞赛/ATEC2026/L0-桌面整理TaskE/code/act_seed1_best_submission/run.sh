#!/bin/bash
set -e
APP_ROOT="${APP_ROOT:-.}"
LOG_ROOT="${LOG_ROOT:-logs/solution}"
mkdir -p "$LOG_ROOT"
cd "$APP_ROOT"
source ./venv/bin/activate
python solution/server.py 2>&1 | tee "$LOG_ROOT/server.stdout"
