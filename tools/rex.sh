#!/usr/bin/env bash
# rex.sh — shortcut: bash tools/rex.sh "remote command"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MSYS_NO_PATHCONV=1
python "$SCRIPT_DIR/remote_exec.py" "$1"
