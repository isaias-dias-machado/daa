#!/usr/bin/env bash
# One-command bootstrap: venv + dependencies + notebook-clean commit hook.
# Idempotent: safe to re-run; the venv is reused if it already exists.
#
# The virtualenv lives OUTSIDE the repository, one directory up (../.venv),
# so the repo tree only ever contains project files. Override the location
# with VENV_DIR=/some/path ./setup.sh.
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-../.venv}"

# 1. Virtual environment (created once, reused afterwards)
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# 2. Dependencies:
#    - requirements-dev.txt pulls in both the runtime pins and the
#      tooling (pytest, ruff, pre-commit).
#    - the project itself is installed editable so `import daa` works.
#    - nbstripout does NOT need installing here: pre-commit builds an
#      isolated environment for it from .pre-commit-config.yaml.
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements-dev.txt
"$VENV_DIR/bin/pip" install -e . --no-deps

# 3. Activate the git commit hook in this clone
"$VENV_DIR/bin/pre-commit" install

echo "Done. Activate the venv with: source $VENV_DIR/bin/activate"
