#!/bin/bash
# Auto-format all Python source files with black

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/backend"

echo "==> Formatting with black..."
black "$BACKEND_DIR"

echo "Done."
