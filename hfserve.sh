#!/usr/bin/env bash
# Launcher for Linux/macOS: run hfserve with the project's virtualenv.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python";
elif [ -x "$ROOT/.vemv/bin/python" ]; then PY="$ROOT/.vemv/bin/python";
else PY=python3; fi
cd "$ROOT" && exec "$PY" -m hfserve "$@"
