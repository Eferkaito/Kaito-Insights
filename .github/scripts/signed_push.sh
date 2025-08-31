#!/usr/bin/env bash
set -euo pipefail
MSG="${1:-Auto commit}"

echo "🧹 Cleaning temp artifacts before commit..."
find . -path ./.git -prune -o -type d \
  \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' -o -name '.ipynb_checkpoints' \) \
  -exec rm -rf {} +
find . -path ./.git -prune -o -type f \
  \( -name '*.tmp' -o -name '*.temp' -o -name '*.log' -o -name '*.cache' \) \
  -exec rm -f {} +

echo "➕ Staging changes..."
git add -A

if git diff --cached --quiet; then
  echo "🟢 No changes to commit for: $MSG"
  exit 0
fi

echo "🖊️ Committing (GPG-signed) with [skip ci] to avoid loops..."
git commit -S -m "[Auto][skip ci] ${MSG}"

echo "🚀 Pushing..."
git push origin HEAD
