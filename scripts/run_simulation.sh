#!/usr/bin/env bash
# Script direto para rodar a simulação e gerar estatísticas e gráficos
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
    echo "==> Criando ambiente virtual uv..."
    uv venv --python python3.12
    uv pip install --python .venv/bin/python -e .
fi

echo "==> Executando Simulação Científica Pareada (N=30)..."
.venv/bin/python -m src.cli "$@"
