#!/usr/bin/env bash
# Helper to activate venv and run demo or tests
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT_DIR/.venv"

if [ ! -d "$VENV" ]; then
  echo "Virtualenv not found at $VENV. Creating..."
  /opt/homebrew/bin/python3.11 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VENV/bin/pip" install -r "$ROOT_DIR/env/requirements.txt"
fi

if [ $# -eq 0 ]; then
  echo "Usage: $0 {demo|test|tests}"
  exit 1
fi

case "$1" in
  demo)
    echo "Starting demo..."
    "$VENV/bin/python" "$ROOT_DIR/env/main.py"
    ;;
  test|tests)
    echo "Running tests..."
    "$VENV/bin/python" -m pytest "$ROOT_DIR/env/tests" -q
    ;;
  stop-demo)
    if [ -f "$VENV/demo.pid" ]; then
      PID=$(cat "$VENV/demo.pid")
      echo "Stopping demo pid=$PID"
      kill "$PID" || echo "Process already stopped"
      rm -f "$VENV/demo.pid"
    else
      echo "No demo PID file found at $VENV/demo.pid"
    fi
    ;;
  *)
    echo "Unknown command: $1"
    exit 2
    ;;
esac
