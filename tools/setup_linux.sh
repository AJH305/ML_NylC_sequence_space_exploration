#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tools/bootstrap_linux.py
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-linux"
export UV_PYTHON_INSTALL_DIR="$PWD/.python-linux"
export UV_CACHE_DIR="$PWD/.uv-cache-linux"
.tools-linux/uv sync --locked --group workflow "$@"
