#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

INPUT_MD="${1:-paper/artigo.md}"
OUTPUT_PDF="${2:-paper/artigo.pdf}"

echo "==> Compilando Artigo IEEE com CSL oficial..."
echo "Entrada: $INPUT_MD"
echo "Saída:   $OUTPUT_PDF"

pandoc "$INPUT_MD" \
  --template=latex/template-ieee.tex \
  --pdf-engine=pdflatex \
  --citeproc \
  --csl=latex/ieee.csl \
  --bibliography=latex/referencias.bib \
  -o "$OUTPUT_PDF"

echo "==> Sucesso! PDF gerado em: $OUTPUT_PDF"

# Renderização automática de páginas para inspeção visual
echo "==> Gerando imagens PNG das páginas do artigo para inspeção visual..."
mkdir -p paper/pages
rm -f paper/pages/page-*.png
if command -v pdftoppm >/dev/null 2>&1; then
    pdftoppm -png -r 150 "$OUTPUT_PDF" paper/pages/page
elif command -v gs >/dev/null 2>&1; then
    gs -sDEVICE=png16m -dSAFER -dBATCH -dNOPAUSE -r150 -sOutputFile="paper/pages/page-%02d.png" "$OUTPUT_PDF" >/dev/null 2>&1
fi
echo "==> Imagens geradas em paper/pages/:"
ls -lh paper/pages/page-*.png
