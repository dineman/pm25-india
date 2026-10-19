#!/usr/bin/env bash
# Build the beginner's booklet PDF from the chapter files and the project glossary.
#
# Usage (from anywhere):  bash docs/booklet/build.sh [--pages DIR]
#   --pages DIR  also render every page as PNG into DIR (for checking the layout)
# Needs pandoc and typst from the pm25 conda env (environment.yml).
set -euo pipefail

cd "$(dirname "$0")"
out=../../results/12_report/pm25_india_booklet.pdf
inputs=(00_front.md 01_big_picture.md 02_steps.md 03_code.md 04_using.md ../steps/glossary.md)
opts=(--resource-path=. -V papersize=a4 -V margin-x=2.2cm -V margin-y=2.2cm -V fontsize=11pt)

conda run -n pm25 pandoc "${inputs[@]}" "${opts[@]}" --pdf-engine=typst -o "$out"
echo "Wrote $out"

if [[ "${1:-}" == "--pages" ]]; then
    pages=$2
    mkdir -p "$pages"
    conda run -n pm25 pandoc "${inputs[@]}" "${opts[@]}" -s -t typst -o _pages.typ
    conda run -n pm25 typst compile --root ../.. _pages.typ "$pages/p-{0p}.png" --ppi 70
    rm -f _pages.typ
    echo "Rendered pages to $pages"
fi
