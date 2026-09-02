#!/bin/bash

# Load a trusted, repository-local shell configuration for a Slurm entry point.
# Shell assignments keep arrays and Slurm defaults easy to edit without adding
# a JSON parser dependency.
load_splice_config() {
    local config_file="$1"
    [[ -f "${config_file}" ]] || {
        echo "ERROR: configuration file does not exist: ${config_file}" >&2
        return 2
    }
    CONFIG_DIR="$(cd "$(dirname "${config_file}")" && pwd)"
    # shellcheck disable=SC1090
    source "${config_file}"
}
