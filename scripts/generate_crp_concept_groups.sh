#!/usr/bin/env bash
# Backward-compatible wrapper; use scripts/generate_cospro_concept_groups.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/generate_cospro_concept_groups.sh" "$@"
