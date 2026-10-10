import json, re, itertools, numpy as np
from scipy import stats
R = '../p9_recheck/results/'
L = lambda f: [json.loads(l) for l in open(R + f)]
A, C, Bp, D = L('cells_stage8_A_xjtu.jsonl'), L('cells_stage8_C_xjtu_clti.jsonl'), L('cells_stage8_B_pu_shared.jsonl'), L('cells_stage9_D_xjtu_rev.jsonl')
AC = A + C
BAND = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
D8 = json.load(open(R + 'stage8_decisions.json')); PB = json.load(open(R + 'paired_bootstrap_check.json'))
def v(cs, a, k, r="final"):
    x = {q["seed"]: q[r] * 100 for q in cs if q["arm"] == a and q["condition"] == k}; assert sorted(x) == list(range(5)), (a, k); return np.array([x[s] for s in range(5)])
def m(cs, a, r="final"): return [v(cs, a, k, r).mean() for k in BAND]
def ci_(d):
    h = stats.t.ppf(.975, 4) * d.std(ddof=1) / 5 ** .5; return (d.mean(), d.mean() - h, d.mean() + h, int((d > 0).sum()))
def pc(cs, a, b, r="final"): return [ci_(v(cs, a, k, r) - v(cs, b, k, r)) for k in BAND]
def r1(x): x = round(float(x), 1); return 0.0 if x == 0 else x
def sg(x): x = r1(x); return "$0.0$" if x == 0 else (f"$+{x:.1f}$" if x > 0 else f"$-{abs(x):.1f}$")
def b_(x): x = r1(x); return f"{x:.1f}" if x >= 0 else f"$-${abs(x):.1f}"
def cell(t): return r"\shortstack{" + sg(t[0]) + r"\\ {[}" + b_(t[1]) + ", " + b_(t[2]) + r"{]} (" + str(t[3]) + "/5)}"
def pp3(t): return ", ".join(sg(x[0]) for x in t)
def pl(t): return ", ".join(f"{r1(x[0]):.1f}" for x in t)
def ivl(t): return f"${sg(t[0])[1:-1]}$ $[{b_(t[1]).replace('$-$','-')}, {b_(t[2]).replace('$-$','-')}]$"
def rd(f): return open(f, encoding='utf8').read()
def wr(f, s): open(f, 'w', encoding='utf8').write(s)
def rep(f, o, n, cnt=1):
    s = rd(f); assert s.count(o) == cnt, (f, o[:90], s.count(o)); wr(f, s.replace(o, n))
def diff_of_diff(cs, a, b, k1, k0):
    g = lambda k: v(cs, a, k) - v(cs, b, k); return ci_(g(k1) - g(k0))
def drop_fracs(cs, num, den, base):
    T = [np.array(t) for t in itertools.product(range(5), repeat=5)]
    d = {k: v(cs, den, k) for k in BAND}; b = {k: v(cs, base, k) for k in BAND}
    anyd = alld = 0
    for t in T:
        nd = sum((d[k][t].mean() - b[k][t].mean()) < 2 for k in BAND); anyd += nd > 0; alld += nd == len(BAND)
    return anyd / len(T), alld / len(T)
