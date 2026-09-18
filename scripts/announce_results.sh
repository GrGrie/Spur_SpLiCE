#!/usr/bin/env bash
# Sourced by every launcher so the first lines of each Slurm .out file say where the job writes.
#
#   source scripts/announce_results.sh
#   announce_results "outputs/reports/<study>/results.json" "checkpoints: ${SPUR_SPLICE_SCRATCH_ROOT}/checkpoints"
#
# The first argument is the main result location; further arguments add one line each.

announce_results() {
  local line
  echo "================================================================"
  echo "Results expected in: $1"
  shift
  for line in "$@"; do
    echo "  ${line}"
  done
  echo "Job: ${SLURM_JOB_ID:-local} on $(hostname) at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "================================================================"
}
