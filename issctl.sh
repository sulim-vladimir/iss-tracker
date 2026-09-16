#!/usr/bin/env bash
# Run issctl with the project's virtualenv, from anywhere:  ./issctl.sh passes
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="$here/.venv/bin/python"
if [[ ! -x "$py" ]]; then
    echo "No virtualenv at $here/.venv - create it with:" >&2
    echo "  python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
exec "$py" -m issctl "$@"
