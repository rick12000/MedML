#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME="$SKILL_ROOT/.diagram-tools"
MERMAID_DIR="$RUNTIME/mermaid"
D2_PREFIX="$RUNTIME/d2"
MERMAID_VERSION="11.17.0"
D2_VERSION="v0.9.0"

want="${1:-all}"
case "$want" in all|mermaid|d2) ;; *) echo "usage: $0 [all|mermaid|d2]" >&2; exit 2;; esac
mkdir -p "$RUNTIME"

d2_bin() {
  if [[ -x "$D2_PREFIX/bin/d2" ]]; then
    echo "$D2_PREFIX/bin/d2"
  elif [[ -x "$D2_PREFIX/bin/d2.exe" ]]; then
    echo "$D2_PREFIX/bin/d2.exe"
  else
    echo "$D2_PREFIX/bin/d2"
  fi
}

d2_ready() {
  local bin
  bin="$(d2_bin)"
  [[ -x "$bin" ]] && "$bin" version 2>/dev/null | grep -q "0.9.0"
}

d2_install_from_extract() {
  local extracted=""
  for d in "$D2_PREFIX/lib/d2"/d2-*; do
    if [[ -d "$d" && -f "$d/scripts/install.sh" ]]; then
      extracted="$d"
      break
    fi
  done
  [[ -n "$extracted" ]] || return 1
  (cd "$extracted" && PREFIX="$D2_PREFIX" ./scripts/install.sh)
}

setup_mermaid() {
  command -v node >/dev/null || { echo "Node.js >=18 is required" >&2; exit 1; }
  command -v npm >/dev/null || { echo "npm is required" >&2; exit 1; }
  node_major="$(node -p 'Number(process.versions.node.split(".")[0])')"
  (( node_major >= 18 )) || { echo "Node.js >=18 is required; found $(node --version)" >&2; exit 1; }

  mkdir -p "$MERMAID_DIR"
  local pkg="$MERMAID_DIR/node_modules/@mermaid-js/mermaid-cli/package.json"
  if [[ -f "$pkg" ]]; then
    local installed_ver=""
    installed_ver="$(node -p "require('$pkg').version" 2>/dev/null || true)"
    if [[ "$installed_ver" == "$MERMAID_VERSION" ]]; then
      echo "Mermaid CLI $MERMAID_VERSION already present: $MERMAID_DIR"
      return
    fi
  fi

  cat > "$MERMAID_DIR/package.json" <<JSON
{"private":true,"dependencies":{"@mermaid-js/mermaid-cli":"$MERMAID_VERSION"}}
JSON
  (cd "$MERMAID_DIR" && npm install --omit=dev --no-audit --no-fund --save-exact)
  "$MERMAID_DIR/node_modules/.bin/mmdc" --version >/dev/null
  echo "Installed Mermaid CLI $MERMAID_VERSION: $MERMAID_DIR"
}

setup_d2() {
  command -v curl >/dev/null || { echo "curl is required for first-time D2 setup" >&2; exit 1; }

  if d2_ready; then
    echo "D2 $D2_VERSION already present: $D2_PREFIX"
    return
  fi

  rm -rf "$D2_PREFIX"
  mkdir -p "$D2_PREFIX"
  # Official installer; --prefix keeps the installation inside this skill.
  if ! curl -fsSL https://d2lang.com/install.sh | sh -s -- --prefix "$D2_PREFIX" --version "$D2_VERSION"; then
    echo "D2 installer exited with an error; trying bundled install.sh from extracted release..." >&2
  fi

  if ! d2_ready; then
    d2_install_from_extract || { echo "D2 setup failed: binary not found under $D2_PREFIX/bin" >&2; exit 1; }
  fi

  "$(d2_bin)" version
  echo "Installed D2 $D2_VERSION: $D2_PREFIX"
}

[[ "$want" == all || "$want" == mermaid ]] && setup_mermaid
[[ "$want" == all || "$want" == d2 ]] && setup_d2
