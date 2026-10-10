import json, re
H = '../handoff/'
ax = json.load(open(H + 'arxiv_hits.json')); cr = json.load(open(H + 'crossref_hits.json'))
KEY = dict(s4='gu2022s4', hippo='gu2020hippo', mamba2='dao2024mamba2', s5='smith2023s5', lru='orvieto2023lru', hyena='poli2023hyena',
 glu='dauphin2017glu', gluvar='shazeer2020glu', highway='srivastava2015highway', senet='hu2018senet', swish='ramachandran2017swish',
 gelu='hendrycks2016gelu', transformer='vaswani2017attention', vim='zhu2024vim', mambaout='yu2024mambaout', copying='jelassi2024copying',
 illusion='merrill2024illusion', corrupt='hendrycks2019corruptions', domainbed='gulrajani2021domainbed', shortcut='geirhos2020shortcut',
 troubling='lipton2018troubling', variance='bouthillier2021variance', drlmatters='henderson2018drl', sgdr='loshchilov2017sgdr',
 adamw='loshchilov2019adamw', dann='ganin2016dann', zhao2020bench='zhao2020benchmark', zoology='arora2024zoology', bn='ioffe2015bn')
def esc(s): return s.replace('&amp;', '&').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#').replace('_', r'\_')
def authors(names):
    out = []
    for n in names[:10]:
        p = n.split(); out.append(p[-1] + ', ' + ' '.join(p[:-1]) if len(p) > 1 else n)
    return ' and '.join(out) + (' and others' if len(names) > 10 else '')
bib = []; new_keys = []
for k, v in ax.items():
    key = KEY[k]; new_keys.append(key)
    bib.append(f"@article{{{key},\n  title   = {{{esc(v['title'])}}},\n  author  = {{{authors(v['authors'])}}},\n  journal = {{arXiv preprint arXiv:{v['id']}}},\n  year    = {{{v['year']}}},\n  eprint  = {{{v['id']}}},\n  archivePrefix = {{arXiv}},\n  url     = {{https://arxiv.org/abs/{v['id']}}}\n}}\n")
CK = dict(lstm=('hochreiter1997lstm', 0), smith2015cwru=('smith2015cwru', 0), zhang2017wdcnn=('zhang2017wdcnn', 0), lei2020review=('lei2020review', 0), neupane2020=('neupane2020cwru', 0))
for k, (key, i) in CK.items():
    it = cr[k][i]; new_keys.append(key)
    au = ' and '.join(f"{a['family']}, {a.get('given','')}".rstrip(', ') for a in it['author'])
    f = [f"  title   = {{{esc(it['title'][0])}}}", f"  author  = {{{au}}}", f"  journal = {{{esc(it['container-title'][0])}}}", f"  year    = {{{it['issued']['date-parts'][0][0]}}}"]
    if it.get('volume'): f.append(f"  volume  = {{{it['volume']}}}")
    if it.get('issue'): f.append(f"  number  = {{{it['issue']}}}")
    if it.get('page'): f.append(f"  pages   = {{{it['page'].replace('-', '--')}}}")
    elif it.get('article-number'): f.append(f"  pages   = {{{it['article-number']}}}")
    f.append(f"  doi     = {{{it['DOI']}}}")
    bib.append(f"@article{{{key},\n" + ',\n'.join(f) + "\n}\n")
assert len(new_keys) == 34 and len(set(new_keys)) == 34
b = open('references.bib', encoding='utf8').read()
for k in new_keys: assert '{' + k + ',' not in b, k
open('references.bib', 'w', encoding='utf8').write(b.rstrip('\n') + '\n\n' + '\n'.join(bib))
def rd(f): return open(f, encoding='utf8').read()
def wr(f, s): open(f, 'w', encoding='utf8').write(s)
def rep(f, o, n):
    s = rd(f); assert s.count(o) == 1, (f, o[:70], s.count(o)); wr(f, s.replace(o, n))
BG = 'sections/background.tex'
rep(BG, r"Diagonal structured SSMs (S4D)~\cite{gu2022s4d} evolve a linear state", r"Diagonal structured SSMs (S4D)~\cite{gu2022s4d}, a simplification of S4~\cite{gu2022s4} whose initialisation derives from HiPPO~\cite{gu2020hippo}, evolve a linear state")
rep(BG, r"acting as a bank of learned linear filters. Mamba-style blocks", r"acting as a bank of learned linear filters. Related alternatives to attention~\cite{vaswani2017attention} include S5, a multi-input multi-output linear SSM~\cite{smith2023s5}, the linear recurrent unit~\cite{orvieto2023lru}, and Hyena, which interleaves implicitly parametrised long convolutions with data-controlled gating~\cite{poli2023hyena}. Mamba-style blocks")
rep(BG, r"follows the Mamba-3 formulation~\cite{lahoti2026mamba3}.", r"follows the Mamba-3 formulation~\cite{lahoti2026mamba3} (see also Mamba-2~\cite{dao2024mamba2}).")
rep(BG, r"does not by itself determine which component earns empirical robustness in a trained block.", r"does not by itself determine which component earns empirical robustness in a trained block.\n\n\\textbf{Multiplicative gating} is not specific to Mamba. It has been used in recurrent networks since the LSTM~\\cite{hochreiter1997lstm}, in highway networks~\\cite{srivastava2015highway}, in gated convolutional language models~\\cite{dauphin2017glu} and their transformer variants~\\cite{shazeer2020glu}, and as channel-wise recalibration in convolutional networks~\\cite{hu2018senet}. The gate of a Mamba-style block applies the SiLU activation~\\cite{hendrycks2016gelu,ramachandran2017swish} to a parallel branch and multiplies the result onto the scan output; this is the operation whose role the present paper isolates.".replace('\\\\n','\n') if False else r"does not by itself determine which component earns empirical robustness in a trained block."+"\n\n"+r"\textbf{Multiplicative gating} is not specific to Mamba. It has been used in recurrent networks since the LSTM~\cite{hochreiter1997lstm}, in highway networks~\cite{srivastava2015highway}, in gated convolutional language models~\cite{dauphin2017glu} and their transformer variants~\cite{shazeer2020glu}, and as channel-wise recalibration in convolutional networks~\cite{hu2018senet}. The gate of a Mamba-style block applies the SiLU activation~\cite{hendrycks2016gelu,ramachandran2017swish} to a parallel branch and multiplies the result onto the scan output; this is the operation whose role the present paper isolates.")
rep(BG, r"theoretical work identifies long-range regimes where Mamba is weaker than S4D~\cite{yu2025blockbiased}.", r"theoretical work identifies long-range regimes where Mamba is weaker than S4D~\cite{yu2025blockbiased}, and copying~\cite{jelassi2024copying}, state-tracking~\cite{merrill2024illusion} and associative-recall~\cite{arora2024zoology} studies delimit tasks on which fixed-state sequence models lag attention. In vision, where Mamba backbones have been adopted~\cite{zhu2024vim}, it has been hypothesised that Mamba is not necessary for image classification~\cite{yu2024mambaout}.")
rep(BG, r"The testbed is two-class (details in Section~\ref{sec:protocol:testbed});", r"Bearing diagnosis with machine learning is reviewed in~\cite{lei2020review,neupane2020cwru}; the Case Western Reserve University data are the best-known benchmark~\cite{smith2015cwru} and have been used to study noise and load changes with convolutional networks~\cite{zhang2017wdcnn}, while open-source benchmark studies stress comparable protocols~\cite{zhao2020benchmark}. Operating-condition shift is usually addressed with domain adaptation such as adversarial training~\cite{ganin2016dann}; we use none, because the question is which component of one block carries robustness, not how to close a domain gap. Additive noise at a fixed SNR is applied in the manner of corruption benchmarks~\cite{hendrycks2019corruptions}. The testbed is two-class (details in Section~\ref{sec:protocol:testbed});")
PR = 'sections/protocol.tex'
rep(PR, r"with a cosine learning-rate schedule that anneals to zero,", r"with AdamW~\cite{loshchilov2019adamw} and a cosine learning-rate schedule~\cite{loshchilov2017sgdr} that anneals to zero,")
rep(PR, r"(the harness defines no separate validation split), and", r"(the harness defines no separate validation split; the choice of model-selection rule can change conclusions in domain-generalisation benchmarks~\cite{gulrajani2021domainbed}), and")
rep(PR, r"both SSM arms use LayerNorm and contain no BatchNorm;", r"both SSM arms use LayerNorm and contain no BatchNorm~\cite{ioffe2015bn};")
rep('sections/results.tex', r"so a model can succeed partly by recognising bearing-specific signatures;", r"so a model can succeed partly by recognising bearing-specific signatures (a form of shortcut learning~\cite{geirhos2020shortcut});")
rep('sections/boundaries.tex', r"Every cell mean uses five seeds.", r"Every cell mean uses five seeds; run-to-run variance is a known confounder of benchmark conclusions~\cite{bouthillier2021variance,henderson2018drl}.")
rep('sections/intro.tex', r"Our approach is deliberately \emph{attributional} rather than \emph{methodological}:", r"Failure to identify the source of an empirical gain is a recurring weakness of machine-learning scholarship~\cite{lipton2018troubling}. Our approach is deliberately \emph{attributional} rather than \emph{methodological}:")
print('applied', len(new_keys))
