#!/usr/bin/env bash
# One-command bootstrap: venv + dependencies + notebook-clean commit hook.
# Idempotent: safe to re-run; the venv is reused if it already exists.
set -euo pipefail
cd "$(dirname "$0")"

# 1. Virtual environment (created once, reused afterwards)
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# 2. Dependencies. Everything lives under tp/:
#    - tp/requirements-dev.txt pulls in both the runtime pins and the
#      tooling (pytest, ruff, pre-commit).
#    - the project itself is installed editable so `import daa` works.
#    - nbstripout does NOT need installing here: pre-commit builds an
#      isolated environment for it from .pre-commit-config.yaml.
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r tp/requirements-dev.txt
./.venv/bin/pip install -e tp --no-deps

# 3. Activate the git commit hook in this clone
./.venv/bin/pre-commit install

echo "Done. Activate the venv with: source .venv/bin/activate"
