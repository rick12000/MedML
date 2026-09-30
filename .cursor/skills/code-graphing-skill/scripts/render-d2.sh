#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: $0 INPUT.d2 OUTPUT.{svg|png|gif|pdf|pptx|txt} [d2 options...]" >&2; exit 2; }
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -x "$SKILL_ROOT/.diagram-tools/d2/bin/d2" ]]; then
  D2="$SKILL_ROOT/.diagram-tools/d2/bin/d2"
elif [[ -x "$SKILL_ROOT/.diagram-tools/d2/bin/d2.exe" ]]; then
  D2="$SKILL_ROOT/.diagram-tools/d2/bin/d2.exe"
else
  D2="$SKILL_ROOT/.diagram-tools/d2/bin/d2"
fi
[[ -x "$D2" ]] || "$SCRIPT_DIR/setup.sh" d2
input="$1"; output="$2"; shift 2
[[ -f "$input" ]] || { echo "input not found: $input" >&2; exit 1; }
mkdir -p "$(dirname "$output")"
# Default look: D2 theme 200 (dark). Override with e.g. --theme=0 after the output path.
"$D2" --theme=200 "$@" "$input" "$output"

primary_output="$output"
if [[ ! -s "$primary_output" ]]; then
  bundle_dir="${output%.*}"
  ext="${output##*.}"
  if [[ -f "$bundle_dir/index.svg" ]]; then
    primary_output="$bundle_dir/index.svg"
  elif [[ -f "$bundle_dir/index.$ext" ]]; then
    primary_output="$bundle_dir/index.$ext"
  fi
fi

if [[ -s "$primary_output" ]]; then
  printf '%s\n' "$primary_output"
  if [[ "$primary_output" != "$output" ]]; then
    printf 'multi-board bundle directory: %s\n' "$(dirname "$primary_output")" >&2
  fi
  exit 0
fi

echo "render produced no output: $output" >&2
exit 1
