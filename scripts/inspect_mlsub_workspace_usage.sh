#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_HOME=${WORKSPACE_HOME:-/home/jovyan}

echo "=== filesystem ==="
df -hT "$WORKSPACE_HOME"

echo "=== top-level workspace usage ==="
du -xhd1 "$WORKSPACE_HOME" 2>/dev/null | sort -hr

echo "=== largest files (top 100) ==="
find "$WORKSPACE_HOME" -xdev -type f -printf '%s\t%TY-%Tm-%Td %p\n' 2>/dev/null \
    | sort -nr \
    | head -100 \
    | numfmt --field=1 --to=iec-i --suffix=B
