#!/usr/bin/env bash
# generate a 600-dpi PK for a Metafont font with the local TeX tree (no downloads)
set -e
name=$1; dpi=${2:-600}
W=$(cd "$(dirname "$0")" && pwd)/mfwork; cd $W
export TEXMFCNF=/usr/share/texlive/texmf-dist/web2c TEXMF="{/usr/share/texlive/texmf-dist,/usr/share/texmf}" TEXMFDBS=''
export MFINPUTS=".:/usr/share/texlive/texmf-dist/metafont/base:/usr/share/texlive/texmf-dist/fonts/source/jknappen/ec:/usr/share/texlive/texmf-dist/fonts/source/public//:/usr/share/texlive/texmf-dist/fonts/source/jknappen//"
export MFBASES=. 
mf -progname=mf -base=plain "\\mode:=ljfour; mag:=$dpi/600; nonstopmode; input $name" >/dev/null 2>&1 || true
gf=$(ls ${name}.*gf | head -1)
gftopk $gf ${name}.${dpi}pk >/dev/null
D=../.texmf-var/fonts/pk/ljfour/jknappen/ec/dpi${dpi}; mkdir -p $D; mv ${name}.${dpi}pk $D/; rm -f ${name}.*gf ${name}.log
echo "made $name @${dpi}"
