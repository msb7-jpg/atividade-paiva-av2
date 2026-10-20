#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

INPUT_MD="${1:-paper/artigo.md}"
OUTPUT_PDF="${2:-paper/artigo.pdf}"

echo "==> Compilando Artigo IEEE..."
echo "Entrada: $INPUT_MD"
echo "Saída:   $OUTPUT_PDF"

pandoc "$INPUT_MD" \
  --template=latex/template-ieee.tex \
  --pdf-engine=pdflatex \
  --citeproc \
  --bibliography=latex/referencias.bib \
  -o "$OUTPUT_PDF"

echo "==> Sucesso! PDF gerado em: $OUTPUT_PDF"
