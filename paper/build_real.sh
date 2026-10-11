#!/usr/bin/env bash
# Full build with real TS1 fonts (generated locally with Metafont) -- no font substitution.
# usage: build_real.sh [texfile-stem=main] ; the document class line is whatever the .tex contains.
set -uo pipefail
cd "$(dirname "$0")"; S=${1:-main}
export TEXMFCNF=/usr/share/texlive/texmf-dist/web2c TEXMFVAR="$PWD/.texmf-var"
export TEXMF="{$TEXMFVAR,/usr/share/texlive/texmf-dist,/usr/share/texmf}" TEXMFDBS='' TEXFORMATS=".fmt:$PWD/.fmt"
if [ ! -f .fmt/pdflatex.fmt ]; then mkdir -p .fmt && ( cd .fmt && pdftex -ini -etex -jobname=pdflatex '\input pdftexconfig.tex \input latex.ltx \dump' >/dev/null ); fi
MAPD="$TEXMFVAR/fonts/map/pdftex/updmap"
if [ ! -s "$MAPD/pdftex.map" ]; then mkdir -p "$MAPD"; find /usr/share/texlive/texmf-dist/fonts/map/dvips /usr/share/texmf/fonts/map/dvips -name '*.map' -type f -print0 | xargs -0 cat | grep -v '^%' | grep -v '^[[:space:]]*$' | sort -u > "$MAPD/pdftex.map"; fi
mkpk() { for i in $(seq 1 30); do rm -f missfont.log; pdflatex -interaction=nonstopmode $S.tex > b_$S.1.log 2>&1; [ -s missfont.log ] || return 0
  sort -u missfont.log | while read -r line; do n=${line##* }; d=$(echo "$line" | sed -E 's/.*--dpi ([0-9]+).*/\1/'); ./genpk.sh $n $d | tail -1; done; done; }
mkpk
bibtex $S > b_$S.bib.log 2>&1 || true
pdflatex -interaction=nonstopmode $S.tex > b_$S.2.log 2>&1
pdflatex -interaction=nonstopmode $S.tex > b_$S.3.log 2>&1
pdflatex -interaction=nonstopmode $S.tex > b_$S.3.log 2>&1
grep -o 'Output written on.*' b_$S.3.log || echo FAIL
echo "errors: $(grep -c '^! ' b_$S.3.log)  undefined: $(grep -cE 'undefined' b_$S.3.log)  missing-char: $(grep -c 'Missing character' b_$S.3.log)  missfont: $( [ -s missfont.log ] && wc -l < missfont.log || echo 0)"
