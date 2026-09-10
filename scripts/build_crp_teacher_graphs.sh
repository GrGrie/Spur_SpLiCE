#!/usr/bin/env bash
# Backward-compatible wrapper; use scripts/build_cospro_teacher_graphs.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/build_cospro_teacher_graphs.sh" "$@"
