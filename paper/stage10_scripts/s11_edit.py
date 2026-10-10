from s10_common import *
import re
Lf1, Lp1 = pc(A, "frozen_clti", "s4d"), pc(A, "bm3_frozen", "s4d")
Rf1, Rp1 = pc(D, "frozen_clti", "s4d"), pc(D, "bm3_frozen", "s4d")
allr = [r1(x[0]) for x in Lf1 + Lp1 + Rf1 + Rp1]; lo, hi = f"{min(allr):.0f}", f"{max(allr):.0f}"
dfull = drop_fracs(AC, "frozen_clti_add", "frozen_clti", "frozen_clti_nogate"); dpart = drop_fracs(A, "bm3_frozen_add", "bm3_frozen", "frozen_nogate")
assert abs(dfull[0] - 507/3125) < 1e-9 and abs(dpart[0] - 16/3125) < 1e-9, (dfull, dpart)
R_ = 'sections/results.tex'
# ---- (1) swapped exclusion fractions ----
s = rd(R_); a = s.index(r"The ratio $\rho$ is a conditional summary"); e = s.index("\n", a)
s = s[:a] + (r"The ratio $\rho$ is a conditional summary: among the 3125 paired seed resamples, at least one noise level was excluded because its denominator fell below $2\pp$ (as pre-specified) in " + f"{100*dfull[0]:.1f}" + r"\% (507) of the resamples for the fully frozen host and in " + f"{100*dpart[0]:.1f}" + r"\% (16) for the partially frozen host; no resample excluded all three levels (the repository field \texttt{paired\_frac\_dropped} is this second statistic, $0$ for both).") + s[e:]
wr(R_, s)
# ---- (2) cell allocation, eight arms ----
rep(R_, r"A pre-specified grid of 150 cells (thresholds, labels and writing consequences committed to the public repository before the first cell) re-ran the central arms with independent training and evaluation noise (Section~\ref{sec:protocol:confounds}); everything else is unchanged.",
    r"A pre-specified grid of 150 cells (thresholds, labels and writing consequences committed to the public repository before the first cell) re-ran the central arms with independent training and evaluation noise (Section~\ref{sec:protocol:confounds}): 105 cells on XJTU-SY (the seven arms of Table~\ref{tab:indep}, three levels, five seeds) and 45 on Paderborn shared bearings (three arms, Section~\ref{sec:results:pu_shared}); everything else is unchanged.")
rep(R_, r"in the eight re-run arms; no contrast changed direction.", r"over the eight arm-by-setting pairs that have an earlier-protocol counterpart (five XJTU-SY arms and three Paderborn arms); no contrast changed direction.")
# ---- (3) Table 6 caption and text ----
rep(R_, r"Descriptive; not pre-specified.}", r"Descriptive; not pre-specified; the eight differences are not corrected for multiple comparisons.}")
rep(R_, r"The statement that the gate's contribution grows with noise depth is therefore supported only by the fully frozen block on the forward transition,",
    r"The $-2$ versus $0$\,dB change for that block has a lower interval bound close to zero and is not treated as a separate finding. The statement that the gate's contribution grows with noise depth is therefore supported only by the fully frozen block on the forward transition,")
# ---- figure 1 caption ----
rep(R_, r"Steps 1 and 2 are tested under independent training and evaluation noise (Sections~\ref{sec:results:indep}--\ref{sec:results:depth}). The additive substitution of step 3 is tested under independent noise in Section~\ref{sec:results:indep}; the graft onto S4D (right box) was run only under the earlier noise protocol (\ref{app:earlier}) and is descriptive. Numbers in the boxes are final-epoch differences in percentage points.",
    r"Steps 1 to 3 are tested under independent training and evaluation noise (Sections~\ref{sec:results:indep}--\ref{sec:results:depth}); step 3 is the in-host replacement of the multiplicative gate by an additive branch, on the partially and the fully frozen host. The graft of the gate onto S4D was run only under the earlier noise protocol and is an exploratory branch in \ref{app:earlier}. Numbers in the boxes are final-epoch differences in percentage points.")
# ---- intro ----
I = 'sections/intro.tex'; s = rd(I)
s = s.replace(r"in which the robustness gap is large (up to 18$\pp$)", r"in which the robustness gap is large (" + f"{lo}--{hi}" + r"$\pp$ between S4D and the frozen Mamba-3 blocks in the independent-noise results)")
a = s.index(r"\textbf{Contributions.}")
s = s[:a] + r"""\textbf{Contributions.} (i) A layered freezing experiment on a Mamba-3 block, down to a scan with input-independent dynamics and read/write coefficients (verified at the kernel interface), in which the advantage over S4D survives on both transitions of one dataset under independent training and evaluation noise. (ii) Same-host tests of the output gate (removal and in-place additive replacement) in a partially and a fully frozen block, with a descriptive analysis of how the gate's contribution changes with noise depth. As supporting material, the study reports a coverage table by protocol, the correction of an initial noise-protocol defect, and a released pre-specification and review trail, including the incidents in which automated adjudication departed from the pre-specified criteria (Appendix~C)."""+"\n"
wr(I, s)
rep('sections/background.tex', r"is large (up to 20$\pp$)", r"is large (" + f"{lo}--{hi}" + r"$\pp$ in the independent-noise results)")
# ---- discussion ----
Dd = 'sections/discussion.tex'
rep(Dd, r"this is the weakest of the three observations, since the gateless--S4D gap is not resolved with five seeds (\ref{sec:results:sensitivity}).",
    r"this is the weakest of the three observations. The gateless--S4D comparison depends on host and protocol: in the fully frozen block under independent noise the gateless arm is above S4D at $0$ and $-2$\,dB and below it at $-6$\,dB (Table~\ref{tab:indep}), whereas for the partially frozen block under the earlier protocol it is not resolved against S4D with five seeds (\ref{sec:results:sensitivity}).")
rep(Dd, r"(i) Making the scan's dynamics and read/write coefficients input-independent did not remove the block's advantage over S4D on either transition of one dataset:",
    r"(i) Making the scan's dynamics and read/write coefficients input-independent did not remove the block's advantage over S4D on either transition of one dataset (the independent-noise grids contain no unfrozen BM3 arm, so they do not show that freezing preserves all of the unfrozen block's performance):")
# ---- conclusion ----
C_ = 'sections/conclusion.tex'
rep(C_, r"The multiplicative output gate costs most when removed, but this cost is resolved only at $-6$\,dB on the original transition", r"Removing the multiplicative output gate causes a substantial loss at $-6$\,dB (resolved there on the original transition, where the earlier-protocol ranking against stem and $A$ removal is descriptive only", 1) if False else None
s = rd(C_); o = r"The multiplicative output gate costs most when removed, but this cost is resolved only at $-6$\,dB on the original transition (at every level for the partially frozen block on the reverse one),"
assert s.count(o) == 1
wr(C_, s.replace(o, r"Removing the multiplicative output gate causes a substantial loss at $-6$\,dB, and this cost is resolved only at that level on the original transition (at every level for the partially frozen block on the reverse one; the ranking of the gate against the stem and $A$ ablations comes from the earlier protocol and is descriptive),"))
rep(C_, r"We offer the chain", r"The independent-noise grids contain no unfrozen BM3 arm, so they establish that the frozen block beats S4D, not that freezing preserves the unfrozen block's performance. We offer the chain")
# ---- boundaries: merge into themes + gateless wording + freezing sentence ----
B = 'sections/boundaries.tex'; s = rd(B)
s = s.replace(r"several secondary statements (conv stem, $A$ parameterisation, gateless versus S4D) are not resolved at this size", r"several secondary statements (conv stem and $A$ parameterisation; gateless versus S4D in the partially frozen block under the earlier protocol) are not resolved at this size")
a = s.index(r"\begin{enumerate}"); b = s.index(r"\end{enumerate}")
items = re.findall(r"\\item \\textbf\{([^}]*)\} (.*)", s[a:b])
d = {k.rstrip('.'): v for k, v in items}
assert len(d) == 15, list(d)
def g(*ks): return " ".join(r"\emph{" + k + r".} " + d[k] for k in ks)
themes = [
 ("Testbed and datasets", ["Testbed scope", "Second dataset", "Matched-SNR scope", "Noise-regime coverage"]),
 ("Noise protocol and coverage", ["Noise protocol", "Coverage by protocol", "Transplant results are earlier-protocol and budget-conditional"]),
 ("What the interventions can and cannot show", ["Scope of the freezing intervention", "Gate interventions", "Attribution level", "Conv-stem dual reading", "Construction checks"]),
 ("Statistics, reporting and pre-specification", ["Sample size", "Reporting rule", "Strength of the pre-specification evidence"]),
]
assert sum(len(v) for _, v in themes) == 15
body = "\n".join(r"\item \textbf{" + t + r".} " + g(*ks) for t, ks in themes)
body = body.replace(r"not equivalence. The value stream", r"not equivalence; the independent-noise grids contain no unfrozen BM3 arm, so they do not show that freezing preserves the unfrozen block's performance. The value stream")
s = s[:a] + r"\begin{enumerate}" + "\n" + body + "\n" + s[b:]
wr(B, s)
# ---- data availability ----
rep('main.tex', r"and the harness's own imported modules, so the training entry point runs from a fresh checkout.",
    r"and the harness's own imported modules. The analysis can be recomputed from the released per-cell records alone (it was re-run from a fresh clone and reproduced the archived decisions byte for byte). Retraining requires the XJTU-SY and Paderborn data, a GPU and the Mamba-3 source and kernels (upstream commit \texttt{e9594ce}); a portable loader resolves the harness's imports to the repository copies, and a 2-epoch smoke run of the reverse-transition driver from a clean copy of the repository (not the author's directory tree) loaded those copies and reproduced the evaluation-label hash, but a full retraining from a clean environment has not been performed.")
# ---- appendix wording ----
Ad = 'appendix/appendixD.tex'
rep(Ad, r"so it must be carried by other parts of the block or by the remaining input-dependent scan terms.", r"so this freeze does not identify where the advantage comes from; the other parts of the block and the remaining input-dependent scan terms are candidates.")
rep(Ad, r"None of the three freezing steps costs accuracy:", r"None of the three freezing steps shows an accuracy loss in the point estimates:")
rep(Ad, r"Removing the gate does not cost accuracy:", r"Removing the gate shows no accuracy cost in this setting:")
print('ok', lo, hi)
