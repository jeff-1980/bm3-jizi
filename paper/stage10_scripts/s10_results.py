from s10_common import *
import re
RS = 'sections/results.tex'
cur = rd(RS).split('\n')
# lines are 1-indexed in earlier greps: sub-sections located by markers
txt = rd(RS)
def block(start_marker, end_marker=None):
    a = txt.index(start_marker); b = txt.index(end_marker, a + 1) if end_marker else len(txt); return txt[a:b]
S_fal = block(r"\subsection{Freezing the input-dependent $\Delta$", r"\subsection{Replacing the gate in its native host}")
S_nat = block(r"\subsection{Replacing the gate in its native host}", r"\subsection{The reverse transition}")
S_pu_on = block(r"\subsection{External validity: two settings on a second dataset}")
old_draft = rd('../p9_build_v11draft_snapshot/sections/results.tex')
a = old_draft.index(r"\subsection{The reverse transition}"); b = old_draft.index(r"\subsection{External validity", a)
S_rev_old = old_draft[a:b]
# ---- relabel reverse (earlier protocol) ----
S_rev_old = S_rev_old.replace(r"\label{sec:results:reverse}", r"\label{sec:results:reverse_old}").replace(r"\label{tab:reverse}", r"\label{tab:reverse_old}").replace(r"Table~\ref{tab:reverse}", r"Table~\ref{tab:reverse_old}")
S_rev_old = S_rev_old.replace(r"\subsection{The reverse transition}", r"\subsection{The reverse transition (earlier protocol)}")
S_rev_old = S_rev_old.replace(r"This direction is harder", r"The independent-noise re-run of four of these arms is Section~\ref{sec:results:reverse}. This direction is harder", 1)
S_nat = S_nat.replace(r"(Section~\ref{sec:results:reverse})", r"(Section~\ref{sec:results:reverse_old})")
assert "sec:results:reverse}" not in S_nat.replace("sec:results:reverse_old}", "")
# ---- rename appendix subsection titles ----
ren = [(r"\subsection{Freezing the input-dependent $\Delta$, $A$, $B$, $C$ projections}", r"\subsection{Freezing the input-dependent $\Delta$, $A$, $B$, $C$ projections (earlier protocol)}"),
       (r"\subsection{Freezing the remaining input-dependent scan terms}", r"\subsection{Freezing the remaining input-dependent scan terms (earlier protocol)}"),
       (r"\subsection{Removing the gate is the only ablation with a large, one-signed effect}", r"\subsection{Single-component removal on the frozen base (earlier protocol)}"),
       (r"\subsection{Adding a narrow gate branch to S4D}", r"\subsection{Adding a narrow gate branch to S4D (earlier protocol)}"),
       (r"\subsection{A width-matched graft and an additive control}", r"\subsection{A width-matched graft and an additive control (earlier protocol)}"),
       (r"\subsection{Replacing the gate in its native host}", r"\subsection{Replacing the gate in its native host (earlier protocol)}"),
       (r"\subsection{External validity: two settings on a second dataset}", r"\subsection{Paderborn: disjoint and shared bearings (earlier protocol)}"),
       (r"\subsection{Sensitivity to the reporting rule}", r"\subsection{Sensitivity to the reporting rule (earlier protocol)}"),
       (r"\subsection{The interaction effect concentrates on the hard class}", r"\subsection{Per-class recall (earlier protocol)}")]
body = S_fal + S_nat + S_rev_old + S_pu_on
for o, n in ren:
    assert body.count(o) == 1, o; body = body.replace(o, n)
# protocol label on every figure/table caption lacking it
EP = r" Earlier noise protocol (training and evaluation noise shared by index)."
def lab_caption(s, label):
    i = s.index(r"\label{" + label + "}"); j = s.rfind(r"\caption{", 0, i)
    seg = s[j:i]
    if "arlier noise protocol" in seg or "earlier protocol" in seg or "earlier-protocol" in seg: return s
    depth, p = 0, j + len(r"\caption{")
    while True:
        c = s[p]
        if c == "{": depth += 1
        elif c == "}":
            if depth == 0: break
            depth -= 1
        p += 1
    return s[:p] + EP + s[p:]
for lab in re.findall(r"\\label\{((?:tab|fig):[^}]+)\}", body): body = lab_caption(body, lab)
# Appendix D wrapper
APPD = r"""\section{Results under the earlier noise protocol}
\label{app:earlier}

This appendix collects the results obtained while the noise added to window $i$ of the training and the evaluation set was drawn from the same stream (Section~\ref{sec:protocol:confounds}), together with the final-epoch and best-epoch re-runs of the original grids. They are reported in full because they were the pre-specified adjudications of the original design, but they are of lower evidential weight than the independent-noise results of Section~\ref{sec:results}: no abstract-level claim rests on them alone, thresholds labelled ``primary'' or ``necessary'' below refer to the earlier-protocol grid only, and where an independent-noise counterpart exists (Table~\ref{tab:coverage}) the counterpart takes precedence. Section~\ref{sec:results:earlier} summarises what these results add.

""" + body.replace(r"\subsection{", r"\subsection{", 0)
wr('appendix/appendixD.tex', APPD)
print('appendix D written', len(APPD))
# ======================= new main Results =======================
Lf1, Lp1 = pc(A, "frozen_clti", "s4d"), pc(A, "bm3_frozen", "s4d")
G2, G2c = pc(A, "bm3_frozen", "frozen_nogate"), pc(AC, "frozen_clti", "frozen_clti_nogate")
CNs = pc(AC, "frozen_clti_nogate", "s4d")
ANn, ANc = pc(A, "bm3_frozen_add", "frozen_nogate"), pc(AC, "frozen_clti_add", "frozen_clti_nogate")
MAp, MAf = pc(A, "bm3_frozen", "bm3_frozen_add"), pc(AC, "frozen_clti", "frozen_clti_add")
Rl1f, Rl1p, Rl2, Rns = pc(D, "frozen_clti", "s4d"), pc(D, "bm3_frozen", "s4d"), pc(D, "bm3_frozen", "frozen_nogate"), pc(D, "frozen_nogate", "s4d")
P1, P2 = pc(Bp, "bm3_frozen", "s4d"), pc(Bp, "bm3_frozen", "frozen_nogate")
rho, rhoc = D8["L3b"], D8["L3c"]
dfp, dfc = drop_fracs(AC, "frozen_clti_add", "frozen_clti", "frozen_clti_nogate"), drop_fracs(A, "bm3_frozen_add", "bm3_frozen", "frozen_nogate")
f3 = lambda x: f"{x:.3f}"
dx = [x for k, vv in D8["indep_minus_shared_descriptive"].items() for x in vv]
dlt9 = [p - q for a_ in ["s4d", "bm3_frozen", "frozen_nogate", "frozen_clti"] for p, q in zip(m(D, a_), m(L('cells_stage3_exp2.jsonl'), a_))]
def mrow(nm, cs, arm): return nm + " & " + " & ".join(f"{x:.1f}" for x in m(cs, arm)) + r" \\"
arms_rows = [("BM3-frozen", A, "bm3_frozen"), ("frozen$-$all", A, "frozen_clti"), ("S4D", A, "s4d"), ("frozen$-$gate", A, "frozen_nogate"), ("frozen$+$add", A, "bm3_frozen_add"), ("frozen$-$all$-$gate", C, "frozen_clti_nogate"), ("frozen$-$all$+$add", C, "frozen_clti_add")]
mrows = "\n".join(mrow(nm, cs, a_) for nm, cs, a_ in arms_rows)
labs = lambda k: D8[k]["label"].lower()
crows = "\n".join(nm + " & " + " & ".join(cell(t) for t in tt) + f" & {lab}" + r" \\[3pt]" for nm, tt, lab in [
    (r"\shortstack[l]{frozen$-$all\\ $-$ S4D}", Lf1, labs("L1f")), (r"\shortstack[l]{BM3-frozen\\ $-$ S4D}", Lp1, labs("L1p")),
    (r"\shortstack[l]{BM3-frozen\\ $-$ frozen$-$gate}", G2, labs("L2")), (r"\shortstack[l]{frozen$-$all\\ $-$ frozen$-$all$-$gate}", G2c, labs("L2c")),
    (r"\shortstack[l]{frozen$-$all$-$gate\\ $-$ S4D}", CNs, "descriptive"),
    (r"\shortstack[l]{BM3-frozen\\ $-$ frozen$+$add}", MAp, "descriptive$^{\\ddagger}$"), (r"\shortstack[l]{frozen$-$all\\ $-$ frozen$-$all$+$add}", MAf, "descriptive$^{\\ddagger}$"),
    (r"\shortstack[l]{frozen$+$add\\ $-$ frozen$-$gate}", ANn, "descriptive"), (r"\shortstack[l]{frozen$-$all$+$add\\ $-$ frozen$-$all$-$gate}", ANc, "descriptive")])
rmrows = "\n".join(nm + " & " + " & ".join(f"{x:.1f} / {y:.1f}" for x, y in zip(m(D, a_, "oracle"), m(D, a_))) + r" \\" for nm, a_ in [("frozen$-$all", "frozen_clti"), ("BM3-frozen", "bm3_frozen"), ("frozen$-$gate", "frozen_nogate"), ("S4D", "s4d")])
rcrows = "\n".join(nm + " & " + " & ".join(cell(t) for t in tt) + f" & {lab}" + r" \\[3pt]" for nm, tt, lab in [
    (r"\shortstack[l]{frozen$-$all\\ $-$ S4D}", Rl1f, "holds"), (r"\shortstack[l]{BM3-frozen\\ $-$ S4D}", Rl1p, "holds"),
    (r"\shortstack[l]{BM3-frozen\\ $-$ frozen$-$gate}", Rl2, "holds"), (r"\shortstack[l]{frozen$-$gate\\ $-$ S4D}", Rns, "descriptive")])
prows = "\n".join(nm + " & " + " & ".join(cell(t) for t in tt) + f" & {lab}" + r" \\[3pt]" for nm, tt, lab in [
    (r"\shortstack[l]{BM3-frozen\\ $-$ S4D}", P1, labs("L1pp")), (r"\shortstack[l]{BM3-frozen\\ $-$ frozen$-$gate}", P2, labs("L2pp"))])
scenes = [("Forward, fully frozen", AC, "frozen_clti", "frozen_clti_nogate"), ("Forward, partially frozen", A, "bm3_frozen", "frozen_nogate"),
          ("Reverse, partially frozen", D, "bm3_frozen", "frozen_nogate"), ("PU shared, partially frozen", Bp, "bm3_frozen", "frozen_nogate")]
dd = {nm: (diff_of_diff(cs, a_, b2, BAND[1], BAND[0]), diff_of_diff(cs, a_, b2, BAND[2], BAND[0])) for nm, cs, a_, b2 in scenes}
ddrows = "\n".join(nm + " & " + cell(dd[nm][0]) + " & " + cell(dd[nm][1]) + r" \\[3pt]" for nm in dd)
def hl(x): return f"${x:.1f}$"
RES = r"""\section{Results}
\label{sec:results}

Throughout, ``deep noise'' denotes the pre-specified adjudication band $\{0, -2, -6\,\mathrm{dB}\}$; $-10\,\mathrm{dB}$ is reported only in the earlier-protocol appendix and excluded from adjudication (the time-invariant baseline is near chance there). All values are macro-F1 (\%), mean over 5 seeds, at the final epoch of the 50-epoch budget (Section~\ref{sec:protocol:report}). \textbf{The main text reports only results obtained with independent training and evaluation noise} (Sections~\ref{sec:results:indep}--\ref{sec:results:depth}). Results obtained while training and evaluation noise were shared by index (Section~\ref{sec:protocol:confounds}) are collected in Appendix~\ref{app:earlier} and summarised, with their evidence level, in Section~\ref{sec:results:earlier}. Table~\ref{tab:coverage} shows which operation was run under which protocol on which host and transition. Figure~\ref{fig:chain} summarises the chain.

\begin{table}[htbp]
\centering
\caption{Coverage of the study by protocol. I: run with independent training and evaluation noise (main text). E: run under the earlier protocol (Appendix~\ref{app:earlier}). $-$: not run. P: partially frozen host (BM3-frozen); F: fully frozen host. ``Freeze'' is the contrast with S4D (BM3-frozen $-$ S4D for P, frozen$-$all $-$ S4D for F); ``gate removal'' and ``additive substitution'' are applied inside the host. Graft, stem/$A$ removal and per-class recall have no fully frozen counterpart.}
\label{tab:coverage}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & \multicolumn{2}{c}{XJTU forward} & \multicolumn{2}{c}{XJTU reverse} & \multicolumn{2}{c}{PU shared} & \multicolumn{2}{c}{PU disjoint} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Operation & P & F & P & F & P & F & P & F \\
\midrule
Freeze (vs S4D) & I & I & I & I & I & $-$ & E & E \\
Gate removal & I & I & I & $-$ & I & $-$ & E & $-$ \\
Additive substitution & I & I & E & $-$ & $-$ & $-$ & E & $-$ \\
Graft onto S4D (with additive control) & E & $-$ & E & $-$ & $-$ & $-$ & $-$ & $-$ \\
Stem and $A$ removal & E & $-$ & $-$ & $-$ & $-$ & $-$ & $-$ & $-$ \\
Per-class recall & E & $-$ & $-$ & $-$ & $-$ & $-$ & $-$ & $-$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/fig1_chain}
\caption{The three-step chain. Steps 1 and 2 are tested under independent training and evaluation noise (Sections~\ref{sec:results:indep}--\ref{sec:results:depth}). The additive substitution of step 3 is tested under independent noise in Section~\ref{sec:results:indep}; the graft onto S4D (right box) was run only under the earlier noise protocol (Appendix~\ref{app:earlier}) and is descriptive. Numbers in the boxes are final-epoch differences in percentage points.}
\label{fig:chain}
\end{figure}

\subsection{Original transition under independent noise}
\label{sec:results:indep}

A pre-specified grid of 150 cells (thresholds, labels and writing consequences committed to the public repository before the first cell) re-ran the central arms with independent training and evaluation noise (Section~\ref{sec:protocol:confounds}); everything else is unchanged. Each contrast is labelled \emph{holds} if at least two of the three levels have a positive mean with an interval excluding zero and none is reversed, \emph{weakened} if exactly one, \emph{does not hold} if none. Relative to the earlier protocol the arm means moved by """ + f"${min(dx):.1f}$ to $+{max(dx):.1f}$" + r"""\,pp in the eight re-run arms; no contrast changed direction.

\begin{table}[htbp]
\centering
\caption{Independent training and evaluation noise, original XJTU-SY transition (37.5\,Hz/11\,kN $\rightarrow$ 40\,Hz/10\,kN). Upper block: macro-F1 (\%) at the final epoch, mean over 5 seeds. Lower block: paired differences with 95\% $t$-intervals and the number of seeds (of 5) with a positive difference, and the pre-specified label. $^{\ddagger}$Added during review, not pre-specified.}
\label{tab:indep}
\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccl@{}}
\toprule
Arm / contrast & 0\,dB & $-2$\,dB & $-6$\,dB & label \\
\midrule
""" + mrows + r"""
\midrule
""" + crows + r"""
\bottomrule
\end{tabular}
\end{table}

\emph{Freezing.} Both freeze contrasts hold. The fully frozen block exceeds S4D by """ + pp3(Lf1) + r"""\,pp, in every seed at every level; the partially frozen block exceeds it by """ + pp3(Lp1) + r"""\,pp.

\emph{Gate removal.} Removing the gate from the partially frozen block costs """ + pp3(G2) + r"""\,pp; the contrast is \emph{weakened} because only the $-6$\,dB interval excludes zero. In the fully frozen block it costs """ + pp3(G2c) + r"""\,pp, again resolved only at $-6$\,dB. Without its gate the fully frozen block still exceeds S4D by """ + sg(CNs[0][0]) + r""" and """ + sg(CNs[1][0]) + r"""\,pp at $0$ and $-2$\,dB (intervals excluding zero) and falls """ + f"{abs(r1(CNs[2][0])):.1f}" + r"""\,pp below it at $-6$\,dB (interval $""" + f"[{CNs[2][1]:.1f}, {CNs[2][2]:.1f}]" + r"""$, all seeds). The gate is therefore not required for a positive advantage over S4D at $0$ and $-2$\,dB, but it is required at $-6$\,dB. At the two milder levels the removal cost is """ + sg(G2c[0][0]) + r""" and """ + sg(G2c[1][0]) + r"""\,pp with intervals that include zero, so a contribution of the gate there is neither demonstrated nor excluded. The pre-specified condition for a combined attribution of the fully frozen block's advantage to the gate (gate removal holds and the additive substitution is not rescued) is not met, and we make no such claim.

\emph{Multiplicative versus additive combination.} Replacing the gate by an additive branch of identical size and initial weights cannot be distinguished from removing it in either host (""" + pp3(ANn) + r"""\,pp and """ + pp3(ANc) + r"""\,pp; all intervals include zero), and the rescue ratios are small ($\rho = """ + f3(rho['point']) + r"""$, paired-seed interval $[""" + f3(rho['paired_ci'][0]) + ", " + f3(rho['paired_ci'][1]) + r"""]$; fully frozen host $\rho = """ + f3(rhoc['point']) + r"""$, $[""" + f3(rhoc['paired_ci'][0]) + ", " + f3(rhoc['paired_ci'][1]) + r"""]$). Because gate removal is only weakened, the pre-specified rule labels the substitution test \emph{not assessable}. This is a statement that no clear rescue was detected, not that the two arms are equivalent: in the fully frozen host the additive arm is """ + sg(ANc[2][0]) + r"""\,pp above the gateless arm at $-6$\,dB (interval $[""" + f"{ANc[2][1]:.1f}, {ANc[2][2]:.1f}" + r"""]$, 5/5 seeds). The direct comparison is more informative: the multiplicative gate exceeds the additive replacement by """ + pp3(MAf) + r"""\,pp in the fully frozen host and by """ + pp3(MAp) + r"""\,pp in the partially frozen host, with intervals excluding zero at $-2$ and $-6$\,dB in the former and at $-6$\,dB in the latter (Table~\ref{tab:indep}, descriptive). The ratio $\rho$ is a conditional summary: in """ + f"{100*dfc[0]:.1f}" + r"""\% of the 3125 paired seed resamples for the fully frozen host (""" + f"{100*dfp[0]:.1f}" + r"""\% for the partially frozen host) at least one noise level was excluded because its denominator fell below $2\pp$, as pre-specified; none excluded all three.

"""
RES += r"""\subsection{Reverse transition under independent noise}
\label{sec:results:reverse}

A second pre-specified grid (60 cells, four arms; thresholds committed before the first cell) repeated the freeze and gate-removal contrasts on the reverse transition, 40\,Hz/10\,kN $\rightarrow$ 37.5\,Hz/11\,kN, with the same bearings in swapped roles, under independent noise. Relative to the earlier protocol the arm means moved by """ + f"${min(dlt9):.1f}$ to $+{max(dlt9):.1f}$" + r"""\,pp. This direction is harder for the time-invariant model: on its evaluation set a majority-class predictor scores 46.8\% macro-F1 and a uniformly random one 41.5\%, and S4D sits close to those levels.

\begin{table}[htbp]
\centering
\caption{Reverse transition (40\,Hz/10\,kN $\rightarrow$ 37.5\,Hz/11\,kN), independent training and evaluation noise. Macro-F1 (\%), mean over 5 seeds, best-epoch / final-epoch; paired differences under final-epoch reporting with 95\% $t$-intervals, seeds positive and label. Fully frozen gate removal and additive substitution were not run in this direction under independent noise (Table~\ref{tab:coverage}).}
\label{tab:reverse}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccl@{}}
\toprule
Arm (best / final) & 0\,dB & $-2$\,dB & $-6$\,dB & label \\
\midrule
""" + rmrows + r"""
\midrule
""" + rcrows + r"""
\bottomrule
\end{tabular}
\end{table}

The fully frozen block exceeds S4D by """ + pp3(Rl1f) + r"""\,pp and the partially frozen block by """ + pp3(Rl1p) + r"""\,pp; removing the gate from the partially frozen block costs """ + pl(Rl2) + r"""\,pp, with intervals excluding zero and 5/5 seeds at every level. All three contrasts hold and none is reversed. In this direction gate removal is thus resolved at every level for the partially frozen host, whereas on the original transition it is resolved at $-6$\,dB only; the two directions are reported side by side and not merged. The gateless arm cannot be distinguished from S4D (""" + pp3(Rns) + r"""\,pp). Every contrast rests on an S4D baseline close to the trivial floor. The reverse direction shares the eight bearings of the forward one with swapped roles and has a single inner-race bearing in its evaluation set, so it tests direction rather than new bearings.

\subsection{Paderborn with shared bearings}
\label{sec:results:pu_shared}

On the Paderborn real-damage bearings (1500 $\rightarrow$ 900\,rpm), a pre-specified 45-cell grid trained and evaluated on the same four bearings (KA04, KA16, KI04, KI14; different recordings) under independent noise (Table~\ref{tab:indep_pu}). Gate removal costs """ + pp3(P2) + r"""\,pp with intervals excluding zero at all three levels (\emph{holds}); the partially frozen block exceeds S4D by """ + pp3(P1) + r"""\,pp and is resolved at $-6$\,dB only (\emph{weakened}). The fully frozen block, the additive arm and the graft were not run in this setting. The companion setting with bearings disjoint from the training set was run only under the earlier protocol, where per-index label agreement between training and evaluation windows is 0.999; it is reported in Appendix~\ref{app:earlier} and is not an independent-noise test. With two bearings per class a bearing's identity determines its label in the shared setting, so a model can succeed partly by recognising bearing-specific signatures; XJTU-SY, by contrast, evaluates on bearings disjoint from the training ones in each direction.

\begin{table}[htbp]
\centering
\caption{Independent training and evaluation noise, Paderborn shared bearings (KA04, KA16, KI04, KI14; 1500 $\rightarrow$ 900\,rpm). Paired differences under final-epoch reporting with 95\% $t$-intervals, seeds positive and pre-specified label. Means: BM3-frozen """ + " / ".join(f"{x:.1f}" for x in m(Bp, "bm3_frozen")) + r""", S4D """ + " / ".join(f"{x:.1f}" for x in m(Bp, "s4d")) + r""", frozen$-$gate """ + " / ".join(f"{x:.1f}" for x in m(Bp, "frozen_nogate")) + r"""\,\%.}
\label{tab:indep_pu}
\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccl@{}}
\toprule
Contrast & 0\,dB & $-2$\,dB & $-6$\,dB & label \\
\midrule
""" + prows + r"""
\bottomrule
\end{tabular}
\end{table}

\subsection{Does the gate's contribution grow with noise depth?}
\label{sec:results:depth}

A contrast that is resolved at one level and not at another does not by itself show that the effect differs between levels. We therefore computed, with seeds paired, the difference between the gate-removal cost at a deeper level and at $0$\,dB (Table~\ref{tab:depth}). This analysis was added during review and is descriptive.

\begin{table}[htbp]
\centering
\caption{Change of the gate-removal cost (frozen host $-$ its gateless counterpart, percentage points) relative to $0$\,dB: mean, 95\% paired $t$-interval and seeds (of 5) with a positive change. Independent noise, final epoch. Descriptive; not pre-specified.}
\label{tab:depth}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcc@{}}
\toprule
Host and transition & $-2$ vs.\ $0$\,dB & $-6$ vs.\ $0$\,dB \\
\midrule
""" + ddrows + r"""
\bottomrule
\end{tabular}
\end{table}

Only the fully frozen host on the forward transition shows a change whose interval excludes zero at $-6$\,dB ($""" + f"{dd['Forward, fully frozen'][1][0]:+.1f}" + r"""$ pp, 5/5 seeds). The partially frozen host on the forward transition has a positive point estimate with an interval that includes zero; on the reverse transition the change is near zero although the cost itself is large at every level; for Paderborn shared bearings the point estimate is negative. The statement that the gate's contribution grows with noise depth is therefore supported only by the fully frozen block on the forward transition, and we restrict it to that case throughout; for the other settings the data are compatible with a depth-independent cost.

\subsection{Evidence from the earlier noise protocol}
\label{sec:results:earlier}

Several results were obtained only under the earlier protocol (Table~\ref{tab:coverage}, Appendix~\ref{app:earlier}). We list them with the weight we give them; none carries an abstract-level claim by itself.
\begin{itemize}
\item \emph{Graft onto S4D} (Appendices~\ref{sec:results:graft}--\ref{sec:results:graftwm}). A gate branch of the original width recovers $R=0.196$ of the gap between S4D and BM3-frozen; one rebuilt in the donor's width recovers $0.296$ (paired-seed interval $""" + f"{PB['R_gate_wm_fwd']['paired_ci'][0]:.2f}, {PB['R_gate_wm_fwd']['paired_ci'][1]:.2f}" + r"""$), against $0.408$ for an additive branch with identical parameters. On the original transition the gate did not recover more than the additive branch; on the reverse transition it recovered somewhat more ($\Delta R = 0.153$, paired-seed interval $""" + f"{PB['dR2_rev']['paired_ci'][0]:.2f}, {PB['dR2_rev']['paired_ci'][1]:.2f}" + r"""$). Descriptive.
\item \emph{Additive substitution on the reverse transition} (Appendix~\ref{sec:results:nativeadd}): not rescued by the pre-specified rule; not re-run under independent noise.
\item \emph{Stem and $A$ removal} (Appendix~\ref{sec:results:localise}): no resolved effect of removing the heavy-tailed $A$ parameterisation; the stem is beneficial to remove at mild noise and costly at $-6$\,dB, with intervals that mostly include zero.
\item \emph{Per-class recall} (Appendix~\ref{sec:results:perclass}, 20-epoch grid): gate-related differences concentrate on the minority inner-race class.
\item \emph{Paderborn with disjoint bearings} (Appendix~\ref{sec:results:pu}): none of the pre-specified contrasts replicates and no architecture transfers well (47.7--56.9\%).
\item \emph{Reporting rule} (Appendix~\ref{sec:results:sensitivity}): best-epoch values from the original grids are optimistic by 0.6--10.6\,pp relative to the final epoch; orderings are unchanged.
\end{itemize}
"""
wr(RS, RES)
print('main results written', len(RES))
