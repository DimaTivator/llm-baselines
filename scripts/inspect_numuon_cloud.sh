#!/bin/bash
set -eu

LOG_ROOT=${LOG_ROOT:-"/home/jovyan/logs/numuon"}
LATEST=$(find "$LOG_ROOT" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-)

echo "LATEST_LOG=$LATEST"
python - "$LATEST" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(errors="replace").replace("\r", "\n")
lines = text.splitlines()
print("\n".join(lines[-300:]))
PY
