#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PYTHON=".venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

if ! "$VENV_PYTHON" -c "import streamlit" >/dev/null 2>&1; then
  "$VENV_PYTHON" -m pip install -r requirements.txt
fi

PORT="${STREAMLIT_PORT:-}"

if [[ -z "$PORT" ]]; then
  PORT="$("$VENV_PYTHON" - <<'PY'
import socket


def can_bind(port: int) -> bool:
    sockets = []
    checks = [
        (socket.AF_INET, ("0.0.0.0", port)),
        (socket.AF_INET6, ("::", port)),
    ]

    try:
        for family, address in checks:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.bind(address)
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()


for candidate in range(8501, 8600):
    if can_bind(candidate):
        print(candidate)
        break
else:
    raise SystemExit("No free Streamlit port found in 8501-8599.")
PY
)"
fi

echo "Starting Streamlit on http://localhost:${PORT}"
exec "$VENV_PYTHON" -m streamlit run streamlit_app.py --server.port "$PORT" "$@"
