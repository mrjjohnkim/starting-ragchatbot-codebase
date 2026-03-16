#!/bin/bash
# Development quality checks

set -e

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/backend"

echo "==> Checking formatting with black..."
black --check "$BACKEND_DIR"

echo ""
echo "==> Running tests..."
cd "$BACKEND_DIR" && uv run pytest tests/ -v

echo ""
echo "All checks passed."
