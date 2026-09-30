#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 INPUT.d2 [OUTPUT.svg] [d2 watch options...]" >&2; exit 2; }
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
D2="$SKILL_ROOT/.diagram-tools/d2/bin/d2"
[[ -x "$D2" ]] || "$SCRIPT_DIR/setup.sh" d2
input="$1"; shift
output="${1:-${input%.d2}.svg}"
if [[ $# -gt 0 ]]; then shift; fi
[[ -f "$input" ]] || { echo "input not found: $input" >&2; exit 1; }
mkdir -p "$(dirname "$output")"
exec "$D2" --watch "$@" "$input" "$output"
