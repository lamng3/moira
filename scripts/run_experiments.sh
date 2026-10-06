#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$root/.venv/bin/python"

if [[ ! -x "$python" ]]; then
  echo "MOIRA is not installed. Run ./setup.sh first." >&2
  exit 1
fi

suite="${1:-quickstart}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ "$suite" != */* && "$suite" != *.json ]]; then
  suite="$root/experiments/configs/$suite.json"
fi

exec "$python" -m experiments "$suite" --project-root "$root" "$@"
