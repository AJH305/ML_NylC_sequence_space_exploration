The authoritative Python dependency resolution is `../uv.lock`.

Core: `uv sync --locked --no-dev`
Development: `uv sync --locked`
Linux workflow: `uv sync --locked --no-dev --group workflow`
ESM-2: add `--extra protein`
METL: add `--extra metl`
Interactive notebooks: add `--group notebooks`

Use Python 3.12.14 (`.python-version`) and uv 0.12.13. The project restricts
Python to 3.12, rather than assuming the copied Python 3.14 environment is portable.
All optional dependencies are resolved in the same universal lock, but are only
installed when selected. The lock fixes wheels, source distributions, transitive
dependencies and the METL source commit. Linux and Windows receive the appropriate
platform distributions. CUDA drivers remain a documented host dependency.

The Snakemake controller and workers use this preinstalled environment. No worker
installs packages or modifies the shared environment during a run. Prepare the
environment before submitting cluster jobs, on a filesystem visible to all nodes.
For simultaneous Windows/WSL use set UV_PROJECT_ENVIRONMENT=.venv-linux in WSL;
never share the Windows virtual environment with Linux.

These uv environments replace the initially proposed Conda-per-rule setup: there
are no external simulation binaries in the current scope, and one authoritative
lock avoids maintaining two conflicting dependency resolutions. Snakemake still
provides scheduling, dependencies, resource allocation and restart.
