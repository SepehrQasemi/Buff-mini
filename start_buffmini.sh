#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  python3 launch_app.py
elif command -v python >/dev/null 2>&1; then
  python launch_app.py
else
  echo "Python is not installed. Install Python 3.11+ and retry."
  exit 1
fi
