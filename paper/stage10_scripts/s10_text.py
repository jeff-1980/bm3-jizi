from s10_common import *
import glob
APPX = ['falsify', 'fullfreeze', 'localise', 'graft', 'graftwm', 'nativeadd', 'reverse_old', 'pu', 'sensitivity', 'perclass']
# ---- "Section" -> "Appendix" for references into Appendix D ----
for fn in ['main.tex'] + glob.glob('sections/*.tex') + glob.glob('appendix/*.tex'):
    s = rd(fn)
    for x in APPX:
        s = s.replace(r"Sections~\ref{sec:results:%s}" % x, r"Appendix~\ref{sec:results:%s}" % x)
        s = s.replace(r"Section~\ref{sec:results:%s}" % x, r"Appendix~\ref{sec:results:%s}" % x)
        s = s.replace(r"Table~\ref{tab:%s}" % x, r"Table~\ref{tab:%s}" % x)
    wr(fn, s)
# ---- main.tex: appendix D, abstract ----
rep('main.tex', r"\input{appendix/appendixC}", r"\input{appendix/appendixC}" + "\n" + r"\input{appendix/appendixD}")
Lf1, Lp1 = pc(A, "frozen_clti", "s4d"), pc(A, "bm3_frozen", "s4d")
Rf = pc(D, "frozen_clti", "s4d")
ABS = r"""
The robustness of Mamba-style selective state-space models is often attributed to input-dependent selectivity. We test this with controlled ablations of one Mamba-3 block on bearing vibration data under cross-condition shift (XJTU-SY, one speed-and-load pair run in both directions, training and evaluation bearings disjoint), with matched-noise training and final-epoch macro-F1 under a fixed budget. All central contrasts were run with training and evaluation noise drawn independently. Making the scan's dynamics and read/write coefficients input-independent does not remove the block's advantage over a time-invariant S4D: on the original transition the fully frozen block leads by """ + pp3(Lf1) + r"""\,pp at $0$, $-2$ and $-6$\,dB, and on the reverse transition by """ + pp3(Rf) + r"""\,pp. Removing the multiplicative output gate costs """ + pp3(pc(AC, "frozen_clti", "frozen_clti_nogate")) + r"""\,pp in the fully frozen block, resolved only at $-6$\,dB; only in this host and direction does the cost grow significantly with noise depth, and without its gate the block still exceeds S4D at $0$ and $-2$\,dB but not at $-6$\,dB. Multiplicative gating exceeds an additive branch of identical size by """ + sg(pc(AC, "frozen_clti", "frozen_clti_add")[2][0]) + r"""\,pp at $-6$\,dB. On Paderborn bearings the gate-removal cost replicates when training and evaluation share bearings. Grafting a gate onto S4D, per-class results and Paderborn with disjoint bearings were run only under an earlier noise protocol and are descriptive. The evidence concerns one block, one shift pair per dataset and five seeds.
"""
s = rd('main.tex'); a = s.index(r"\begin{abstract}") + len(r"\begin{abstract}"); e = s.index(r"\end{abstract}"); wr('main.tex', s[:a] + ABS + s[e:])
import re
ab = re.sub(r'%.*', '', ABS); print('abstract words', len(re.sub(r'\\[a-zA-Z]+|[{}$]', ' ', ab).split()))
# ---- intro ----
I = 'sections/intro.tex'; s = rd(I)
a = s.index(r"Our approach is deliberately"); 
NEW = r"""Our approach is deliberately \emph{attributional} rather than \emph{methodological}: we propose no new architecture and claim no state-of-the-art. We take a regime in which a Mamba-style block shows a large, stable robustness advantage over a time-invariant SSM, and ask which ingredient earns it, in three steps with adjudication thresholds fixed before each grid was run:

\begin{enumerate}
\item \textbf{Freeze the scan.} The input-dependent parts of $\Delta$, $A$, $B$, $C$ are replaced by learned constants; in a second arm the trapezoidal weight and rotation angles are as well, so that the scan's dynamics and read/write coefficients no longer depend on the input. The value stream and the gate input remain input-dependent (Section~\ref{sec:protocol:freeze}).
\item \textbf{Remove the gate.} On the frozen base the multiplicative SiLU gate is removed, with the other components and the budget unchanged.
\item \textbf{Replace or transplant the gate.} In place, the gate is replaced by an additive branch of identical size and initial weights; across hosts, it is grafted onto plain S4D with an additive control.
\end{enumerate}

Steps 1 and 2 and the in-place replacement were tested with training and evaluation noise drawn independently (Section~\ref{sec:results}); the graft onto S4D, the per-class analysis, the stem and $A$ ablations and the disjoint-bearing Paderborn setting were run only under an earlier protocol in which the two noise streams were shared by window index, and are reported in Appendix~\ref{app:earlier} as descriptive evidence (Table~\ref{tab:coverage}). The outcome is a conditional pattern. Making the scan input-independent does not remove the block's advantage over S4D, on either transition (""" + pp3(Lf1) + r"""\,pp and """ + pp3(Rf) + r"""\,pp for the fully frozen block). Removing the gate costs """ + pp3(pc(A, "bm3_frozen", "frozen_nogate")) + r"""\,pp in the partially frozen block on the original transition, resolved only at $-6$\,dB, and """ + pl(pc(D, "bm3_frozen", "frozen_nogate")) + r"""\,pp on the reverse transition, resolved at every level. Only in the fully frozen block on the original transition does the gate-removal cost grow significantly with noise depth: without its gate that block still exceeds S4D at $0$ and $-2$\,dB but not at $-6$\,dB. An additive branch of identical size does not recover the gate's effect at $-6$\,dB in either host, although the pre-specified rescue test is not assessable. On Paderborn bearings the gate-removal cost replicates when training and evaluation share bearings.

\textbf{Contributions.} (i) A layered freezing experiment on a Mamba-3 block, down to a scan with input-independent dynamics and read/write coefficients, in which the advantage over S4D survives on both transitions of one dataset under independent training and evaluation noise; the frozen coefficients are verified input-independent at the kernel interface. (ii) Same-host tests of the output gate (removal and in-place additive replacement) in a partially and a fully frozen block, together with a descriptive analysis of how the gate's contribution changes with noise depth. (iii) A coverage table that states, for every operation, whether it was run under independent noise, and a correction of the original noise protocol with a pre-specified re-run. (iv) A released pre-specification and review trail, including the incidents in which automated adjudication departed from the pre-specified criteria.
"""
wr(I, s[:a] + NEW)
s = rd(I); s = s.replace(r"the evaluation labels are identical across all 865 cells of the original grids (hash-verified)", r"the evaluation labels are identical across all cells of each transition (hash-verified)"); wr(I, s)
print('intro ok')
