#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: $0 INPUT.mmd OUTPUT.{svg|png|pdf} [mmdc options...]" >&2; exit 2; }
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MMDC="$SKILL_ROOT/.diagram-tools/mermaid/node_modules/.bin/mmdc"
[[ -x "$MMDC" ]] || "$SCRIPT_DIR/setup.sh" mermaid
input="$1"; output="$2"; shift 2
[[ -f "$input" ]] || { echo "input not found: $input" >&2; exit 1; }
mkdir -p "$(dirname "$output")"
THEME_CONFIG="$SKILL_ROOT/assets/mermaid-theme.json"
[[ -f "$THEME_CONFIG" ]] || { echo "missing theme config: $THEME_CONFIG" >&2; exit 1; }
# Default look: dark theme + Catppuccin-style canvas (override by passing extra mmdc flags after the output path).
"$MMDC" -c "$THEME_CONFIG" -b "#1e1e2e" -i "$input" -o "$output" "$@"
[[ -s "$output" ]] || { echo "render produced no output: $output" >&2; exit 1; }
printf '%s\n' "$output"
