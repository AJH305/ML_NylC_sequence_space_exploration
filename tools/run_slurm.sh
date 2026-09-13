#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-linux}"
PROFILE="${NYLC_PROFILE:-slurm-cpu}"
CONFIG="${NYLC_CONFIG:-configs/default.yaml}"
DEVICE="${NYLC_DEVICE:-cpu}"
# Install the locked environment on the login node before this launcher is used.
# The controller remains on the login node; Snakemake submits the individual jobs.
exec "$UV_PROJECT_ENVIRONMENT/bin/snakemake" --snakefile workflow/Snakefile \
    --profile "workflow/profiles/$PROFILE" \
    --config "experiment_config=$CONFIG" "device=$DEVICE" "$@"
