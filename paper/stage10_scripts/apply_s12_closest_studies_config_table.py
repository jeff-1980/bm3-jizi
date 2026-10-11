def rd(f): return open(f, encoding='utf8').read()
def wr(f, s): open(f, 'w', encoding='utf8').write(s)
def rep(f, o, n):
    s = rd(f); assert s.count(o) == 1, (f, o[:80], s.count(o)); wr(f, s.replace(o, n))
# ---- 1. closest studies ----
BG = 'sections/background.tex'
rep(BG, r"In vision, where Mamba backbones have been adopted~\cite{zhu2024vim}, it has been hypothesised that Mamba is not necessary for image classification~\cite{yu2024mambaout}. ", r"In vision, where Mamba backbones have been adopted~\cite{zhu2024vim}, it has been hypothesised that Mamba is not necessary for image classification (discussed below). ")
s = rd(BG); a = s.index(r"\subsection{Testbed rationale}")
NEW = r"""\textbf{Relation to the closest studies.} Three studies are closest in question or method, and the present work differs from each in a specific way. (i) MambaOut~\cite{yu2024mambaout} stacks Mamba blocks with their SSM token mixer removed, leaving a gated convolutional block, and reports that on ImageNet classification such models surpass visual Mamba models; it asks, for whole vision models, whether the SSM is needed. Here the selective-scan layer is kept in every arm, its input-dependent parameters are replaced by learned constants step by step (Section~\ref{sec:protocol:freeze}), and the output gate is then removed or replaced by an additive branch inside the same host, on one-dimensional vibration signals under matched noise and operating-condition shift; nothing is claimed about vision. (ii) The 59-dataset comparison~\cite{saadatmand2026s4d} compares complete S4D-based and Mamba-based architectures on clean benchmarks; the differences are the evaluation regime (noise and condition shift) and the level of intervention (components within one block family rather than complete models). (iii) The selectivity theory~\cite{huang2025selectivity,yu2025blockbiased} concerns what input-dependent dynamics can represent; the experiments here are empirical and do not test representability, so they neither confirm nor contradict those results. What this paper adds is therefore bounded: a component-level attribution inside one block design, in one noisy cross-condition regime.

"""
wr(BG, s[:a] + NEW + s[a:])
# ---- 2. configuration table ----
P = 'sections/protocol.tex'
rep(P, r"The principal arms are summarised in Table~\ref{tab:arms}.", r"The principal arms are summarised in Table~\ref{tab:arms}, and the configuration that the code fixes for all of them in Table~\ref{tab:config}.")
s = rd(P); e = s.index(r"\end{longtable}\endgroup")  if r"\end{longtable}\endgroup" in s else None
if e is None:
    e = s.index(r"\end{longtable}"); e = s.index("\n", s.index(r"\endgroup", e))
else:
    e += len(r"\end{longtable}\endgroup")
TAB = r"""

\begin{table}[htbp]
\centering\footnotesize
\caption{Configuration fixed by the training code for all arms (differences between arms are noted). OR: outer race, IR: inner race.}
\label{tab:config}
\begin{tabular}{@{}lp{0.74\linewidth}@{}}
\toprule
Setting & Value \\
\midrule
Signal & Horizontal channel at 25.6\,kHz; non-overlapping 2048-sample windows (16 per 32{,}768-sample file) from the fault phase. \\
Fault phase & Starts at the earlier of (a) the first file at which the rolling mean over 5 files of the per-file kurtosis exceeds 5.0 and (b) the first file at which the rolling mean of the per-file standard deviation exceeds twice its baseline (mean over the first $\min(20, N/5)$ files, $N$ files in the run); if neither occurs, the last 15\% of the files. \\
Scaling and noise & Each window is z-scored, then white Gaussian noise is added at the stated SNR to training and evaluation windows. \\
Network & $d_{\mathrm{model}}=64$, 4 layers, state size 128, LayerNorm; input stem Conv1d($1\!\rightarrow\!64$), stride 2, padding 3, kernel 7 in the Mamba-3 arms and 8 in S4D; Mamba-3 expansion 2 ($d_{\mathrm{inner}}=128$). \\
Optimisation & AdamW, learning rate $3\times10^{-4}$, weight decay $10^{-4}$, cosine annealing to zero over 50 epochs, batch size 64 (evaluation 128), gradient-norm clipping at 1.0. \\
Sampling & Training windows are drawn with replacement with probability inversely proportional to class frequency, as many draws per epoch as training windows. With 4{,}624 OR and 624 IR training windows (original transition) the expected class shares are equal instead of 11.9\% IR, so IR windows are drawn about 4.2 times as often as their natural frequency. This holds for every arm; its interaction with the gate was not tested. \\
Loss and seeds & Cross-entropy only in all attribution arms. Seeds 0--4 set initialisation, sampler draws and noise; they do not resample bearings. \\
\bottomrule
\end{tabular}
\end{table}
"""
wr(P, s[:e] + TAB + s[e:])
# ---- 3. abstract ----
rep('main.tex', r"The evidence concerns one block, one shift pair per dataset and five seeds.", r"The evidence concerns one block design, one shift pair per dataset and five seeds.")
# ---- 4. compress duplicated process text ----
s = rd(P); a = s.index(r"Adjudication was mechanical: an analysis pass"); b = s.index(r"In the grids of the original design the adjudication criteria")
wr(P, s[:a] + r"Adjudication was mechanical (an analysis pass applies the thresholds and a separate reviewer pass re-computes the aggregates); the incidents in which automated adjudication departed from the pre-specified criteria, and the human adjudication gates, are described in \ref{app:incidents}. " + s[b:])
B = 'sections/boundaries.tex'
rep(B, r"\emph{Coverage by protocol.} Table~\ref{tab:coverage} states which operations were run under independent noise. Not run under independent noise: the graft onto S4D, the stem and $A$ ablations, additive substitution on the reverse transition, the per-class grid and the disjoint-bearing Paderborn setting. Not run at all:", r"\emph{Coverage by protocol.} Table~\ref{tab:coverage} lists, for every operation, whether it was run under independent noise; the graft onto S4D, the stem and $A$ ablations, additive substitution on the reverse transition, the per-class grid and the disjoint-bearing Paderborn setting were not. Not run at all:")
s = rd(B); a = s.index(r"\emph{Strength of the pre-specification evidence.}"); b = s.index("\n", a)
wr(B, s[:a] + r"\emph{Pre-specification.} Thresholds were fixed before each grid: internally for the original design (file times, not tamper-evident) and, from the width-matched graft onwards, also in a public repository before the first cell (Section~\ref{sec:protocol:prereg}). The noise-depth analysis and the multiplicative--additive differences are post hoc." + s[b:])
print('ok')
