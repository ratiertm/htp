#!/usr/bin/env bash
# Hub Topology Runtime 3D Dashboard
# skild conda env로 서버 실행
SKILD_PY="$HOME/anaconda3/envs/skild/python.exe"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  Hub Topology Runtime 3D Dashboard"
echo "========================================"
echo "  URL: http://localhost:8765"
echo "========================================"

cd "$SCRIPT_DIR"
"$SKILD_PY" server.py
