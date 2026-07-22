#!/usr/bin/env python3
"""apply_secondary_vault_prereg.py — mechanically apply the ACTUAL preregistered
component-ablation decision rule (embedded in the vault task card
`bm3-secondary-ablation-decide.md`, created 2026-07-03, verified in
prereg_provenance.md to predate the `results/secondary_20260703-1034` grid)
to already-computed, frozen aggregate deltas.

This script does NOT recompute any aggregate statistic, does NOT touch
`results/secondary_20260703-1034/cells.jsonl`, and does NOT launch any grid.
It only reads two already-frozen (chmod 444) JSON files produced by an earlier,
untouched aggregation pass and classifies them per the card's own rule text:

    深噪声区（0/-2/-6dB）Delta(frozen - X):
      PRIMARY_DRIVER : all three deep-noise bins >= +8pp AND X collapses toward
                        s4d (all three deep-noise bins of Delta(X - s4d) <= +5pp)
      CONTRIBUTOR    : mean of the three deep-noise bins in [+3, +8)pp
      NOT_DRIVER     : ALL THREE deep-noise bins individually satisfy |Delta| < +3pp
      overall DISTRIBUTED_ACROSS_COMPONENTS: no single PRIMARY_DRIVER arm but
                        >=2 CONTRIBUTOR arms

Card text (verbatim, prereg_provenance.md §1): "无关（NOT_DRIVER）：深噪三档
|Δ| < +3pp。" — this parallels the PRIMARY_DRIVER phrasing ("深噪三档均 >= +8pp",
i.e. all three bins), and unlike the CONTRIBUTOR clause it does NOT say "均值"
(mean). So NOT_DRIVER is a per-bin condition on all three bins individually,
not a condition on their mean. An arm whose three bins do not all individually
satisfy PRIMARY_DRIVER, CONTRIBUTOR (mean-based, per the card), or NOT_DRIVER
(per-bin, per the card) falls outside the three preregistered categories and
is reported as UNCLASSIFIED_BY_PREREG rather than being defaulted to
NOT_DRIVER by an unregistered mean fallback.

2026-07-05 16:19 fix (per .orchestrate/0705-160159/r1_review.json, critical
issues #1-#3): the previous version of this script (see git history /
results/secondary_adjudication_20260705-1610/) used `NOT_DRIVER : |mean of
the three deep-noise bins| < +3pp` as an else-branch fallback for anything
that wasn't PRIMARY_DRIVER or CONTRIBUTOR. That let cancelling per-bin values
(e.g. frozen_noconv's {-6.56, -2.66, +7.91}, mean -0.44) or a lone
out-of-band bin (frozen_As4d's {-2.41, -2.21, -4.15}, one bin over 3pp in
magnitude) be mislabeled NOT_DRIVER even though the card's own text does not
use a mean for that branch. Fixed in `classify_arm()` below: NOT_DRIVER now
requires all three bins individually satisfy |Delta| < +3pp; anything meeting
none of the three named conditions is UNCLASSIFIED_BY_PREREG.

2026-07-05 16:29 fix (per .orchestrate/0705-160159/r2_review.json, critical
issue #1): r2 reviewed the 16:19 artifact (results/secondary_adjudication_
20260705-1619/) and found that, while the per-bin NOT_DRIVER fix above was
numerically correct (frozen_noconv and frozen_As4d indeed meet none of the
three named per-arm conditions), the enum used to report that outcome —
`UNCLASSIFIED_BY_PREREG` — is not rubric-legal; the rubric's expected
mechanical outcome for arms meeting none of PRIMARY_DRIVER/CONTRIBUTOR/
NOT_DRIVER is `OUT_OF_PREREG_BINS`. Fixed below: `classify_arm()`'s fallback
branch, `ALLOWED_ARM_VERDICTS`, and the overall-reason text now use
`OUT_OF_PREREG_BINS` instead of `UNCLASSIFIED_BY_PREREG`. The Delta evidence,
per-bin reasoning, and `overall_verdict` gate (SINGLE_PRIMARY_DRIVER,
frozen_nogate sole primary) are unchanged by this rename.

Source of the rule: vault card
`/mnt/c/Users/ThinkPad/Obsidian Vault/故障诊断Wiki/_tasks/bm3-secondary-ablation-decide.md`,
lines 22-28, quoted verbatim in prereg_provenance.md at the workdir root.
"""
import json
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent
SRC_DELTAS = WORKDIR / "results/secondary_deltapp_analysis_20260704-2224/secondary_component_deltas.json"
SRC_Q = WORKDIR / "results/secondary_deltapp_analysis_20260704-2224/q_secondary.json"
# New timestamped output dir per run (guardrail: never overwrite results/**); this
# supersedes results/secondary_adjudication_20260705-1619 (see supersedes_note below).
OUT_DIR_NAME = "secondary_adjudication_20260705-1629"

DEEP_NOISE_BINS = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]  # card: "深噪声区（0/-2/-6dB）"
OUT_OF_PREREG_BINS = ["clean", "awgn@+10dB", "awgn@+6dB", "awgn@-10dB"]

PRIMARY_DELTA_THRESHOLD = 8.0
PRIMARY_COLLAPSE_THRESHOLD = 5.0
CONTRIBUTOR_LOW = 3.0
CONTRIBUTOR_HIGH = 8.0
NOT_DRIVER_ABS_THRESHOLD = 3.0  # per-bin, per card text "深噪三档 |Δ| < +3pp" (no "均值"/mean)

ALLOWED_ARM_VERDICTS = {"PRIMARY_DRIVER", "CONTRIBUTOR", "NOT_DRIVER", "OUT_OF_PREREG_BINS"}
ALLOWED_OVERALL_VERDICTS = {
    "SINGLE_PRIMARY_DRIVER",
    "MULTIPLE_PRIMARY_DRIVERS",
    "DISTRIBUTED_ACROSS_COMPONENTS",
    "SINGLE_CONTRIBUTOR_NO_PRIMARY",
    "NO_DRIVER_IDENTIFIED",
}


def classify_arm(deep_frozen_minus_x, deep_x_minus_s4d):
    all_primary = all(v >= PRIMARY_DELTA_THRESHOLD for v in deep_frozen_minus_x)
    all_collapse = all(v <= PRIMARY_COLLAPSE_THRESHOLD for v in deep_x_minus_s4d)
    if all_primary and all_collapse:
        return "PRIMARY_DRIVER", {
            "rule_applied": "all three deep-noise bins of delta_frozen_minus_X_pp >= "
                            f"{PRIMARY_DELTA_THRESHOLD} AND all three deep-noise bins of "
                            f"delta_X_minus_s4d_pp <= {PRIMARY_COLLAPSE_THRESHOLD}",
        }
    mean_deep = sum(deep_frozen_minus_x) / len(deep_frozen_minus_x)
    if CONTRIBUTOR_LOW <= mean_deep < CONTRIBUTOR_HIGH:
        return "CONTRIBUTOR", {
            "rule_applied": f"mean of three deep-noise bins ({mean_deep:.4f}pp) in "
                            f"[{CONTRIBUTOR_LOW}, {CONTRIBUTOR_HIGH})pp",
            "mean_deep_noise_delta_pp": round(mean_deep, 4),
        }
    all_not_driver = all(abs(v) < NOT_DRIVER_ABS_THRESHOLD for v in deep_frozen_minus_x)
    if all_not_driver:
        return "NOT_DRIVER", {
            "rule_applied": f"all three deep-noise bins individually satisfy "
                            f"|Delta| < {NOT_DRIVER_ABS_THRESHOLD}pp",
            "mean_deep_noise_delta_pp": round(mean_deep, 4),
        }
    return "OUT_OF_PREREG_BINS", {
        "rule_applied": (
            "meets none of the three preregistered per-bin/mean conditions: not all "
            f"three bins >= {PRIMARY_DELTA_THRESHOLD}pp (PRIMARY_DRIVER), mean "
            f"{mean_deep:.4f}pp not in [{CONTRIBUTOR_LOW}, {CONTRIBUTOR_HIGH})pp "
            "(CONTRIBUTOR), and not all three bins individually satisfy "
            f"|Delta| < {NOT_DRIVER_ABS_THRESHOLD}pp (NOT_DRIVER); reported "
            "transparently as OUT_OF_PREREG_BINS rather than defaulted to NOT_DRIVER "
            "by an unregistered mean fallback"
        ),
        "mean_deep_noise_delta_pp": round(mean_deep, 4),
    }


def shape_note(cond_values):
    """Purely descriptive (non-decisive) note on the out-of-prereg-bin pattern."""
    signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in cond_values]
    if len(set(signs)) > 1:
        return "sign changes across out-of-prereg bins (not monotonic in one direction)"
    if all(s >= 0 for s in signs):
        return "consistently non-negative (frozen >= X) across out-of-prereg bins"
    return "consistently non-positive (frozen <= X) across out-of-prereg bins"


def main():
    deltas = json.loads(SRC_DELTAS.read_text(encoding="utf-8"))
    q = json.loads(SRC_Q.read_text(encoding="utf-8"))

    per_component_verdicts = {}
    per_component_evidence = {}
    for arm, by_cond in deltas["deltas"].items():
        deep_frozen_minus_x = [by_cond[c]["delta_frozen_minus_X_pp"] for c in DEEP_NOISE_BINS]
        deep_x_minus_s4d = [by_cond[c]["delta_X_minus_s4d_pp"] for c in DEEP_NOISE_BINS]
        verdict, detail = classify_arm(deep_frozen_minus_x, deep_x_minus_s4d)
        assert verdict in ALLOWED_ARM_VERDICTS, f"invalid arm verdict enum: {verdict!r}"

        per_component_verdicts[arm] = {
            "verdict": verdict,
            **detail,
            "deep_noise_bins_used": DEEP_NOISE_BINS,
            "deep_noise_delta_frozen_minus_X_pp": dict(zip(DEEP_NOISE_BINS, deep_frozen_minus_x)),
            "deep_noise_delta_X_minus_s4d_pp": dict(zip(DEEP_NOISE_BINS, deep_x_minus_s4d)),
        }

        out_frozen_minus_x = {c: by_cond[c]["delta_frozen_minus_X_pp"] for c in OUT_OF_PREREG_BINS}
        out_x_minus_s4d = {c: by_cond[c]["delta_X_minus_s4d_pp"] for c in OUT_OF_PREREG_BINS}
        per_component_evidence[arm] = {
            "all_conditions_delta_frozen_minus_X_pp": {c: by_cond[c]["delta_frozen_minus_X_pp"] for c in deltas["conditions_order"]},
            "all_conditions_delta_X_minus_s4d_pp": {c: by_cond[c]["delta_X_minus_s4d_pp"] for c in deltas["conditions_order"]},
            "OUT_OF_PREREG_BINS": {
                "bins": OUT_OF_PREREG_BINS,
                "note": "reported for transparency per card's '逐档证据在场、不掩盖逐档'; NOT used in classification",
                "shape_annotation_delta_frozen_minus_X": shape_note(list(out_frozen_minus_x.values())),
                "delta_frozen_minus_X_pp": out_frozen_minus_x,
                "delta_X_minus_s4d_pp": out_x_minus_s4d,
            },
            "raw_mean_std_macro_f1_by_condition": {
                cond: {
                    "bm3_frozen": q["conditions"][cond]["bm3_frozen"],
                    arm: q["conditions"][cond][arm],
                    "s4d": q["conditions"][cond]["s4d"],
                }
                for cond in deltas["conditions_order"]
            },
        }

    primary_arms = [a for a, v in per_component_verdicts.items() if v["verdict"] == "PRIMARY_DRIVER"]
    contributor_arms = [a for a, v in per_component_verdicts.items() if v["verdict"] == "CONTRIBUTOR"]
    out_of_prereg_arms = [a for a, v in per_component_verdicts.items() if v["verdict"] == "OUT_OF_PREREG_BINS"]

    out_of_prereg_suffix = f" ({len(out_of_prereg_arms)} arm(s) OUT_OF_PREREG_BINS: {out_of_prereg_arms}, not counted toward PRIMARY_DRIVER/CONTRIBUTOR/NOT_DRIVER)" if out_of_prereg_arms else ""

    if len(primary_arms) == 1:
        overall_verdict = "SINGLE_PRIMARY_DRIVER"
        overall_reason = f"exactly one component arm ({primary_arms[0]}) meets PRIMARY_DRIVER; no other arm is PRIMARY_DRIVER or CONTRIBUTOR.{out_of_prereg_suffix}"
    elif len(primary_arms) > 1:
        overall_verdict = "MULTIPLE_PRIMARY_DRIVERS"
        overall_reason = f"more than one component arm meets PRIMARY_DRIVER: {primary_arms}. The card's rule text does not define this case; reported transparently rather than silently picking one.{out_of_prereg_suffix}"
    elif len(contributor_arms) >= 2:
        overall_verdict = "DISTRIBUTED_ACROSS_COMPONENTS"
        overall_reason = f"no single PRIMARY_DRIVER arm; {len(contributor_arms)} CONTRIBUTOR arms ({contributor_arms}) per card's '组合情形' rule.{out_of_prereg_suffix}"
    elif len(contributor_arms) == 1:
        overall_verdict = "SINGLE_CONTRIBUTOR_NO_PRIMARY"
        overall_reason = f"no PRIMARY_DRIVER arm; exactly one CONTRIBUTOR arm ({contributor_arms[0]}). The card's rule text does not name this case explicitly; reported transparently.{out_of_prereg_suffix}"
    else:
        overall_verdict = "NO_DRIVER_IDENTIFIED"
        overall_reason = f"no component arm meets PRIMARY_DRIVER or CONTRIBUTOR; all arms are NOT_DRIVER or OUT_OF_PREREG_BINS.{out_of_prereg_suffix}"

    assert overall_verdict in ALLOWED_OVERALL_VERDICTS, f"invalid overall verdict enum: {overall_verdict!r}"

    decision = {
        "decision_kind": "mechanical_vault_prereg_application",
        "preregistered": True,
        "adjudicates_E3": True,
        "prereg_source": "vault card bm3-secondary-ablation-decide (created 2026-07-03) + log.md 2026-07-03, materialized in prereg_provenance.md",
        "prereg_card_path": "/mnt/c/Users/ThinkPad/Obsidian Vault/故障诊断Wiki/_tasks/bm3-secondary-ablation-decide.md",
        "prereg_provenance_doc": "prereg_provenance.md",
        "run_dir": "results/secondary_20260703-1034",
        "grid_start": "2026-07-03T10:34",
        "prereg_predates_grid": True,
        "void_preregs": {
            "results/secondary_prereg_20260704-0708/secondary_prereg.json": "post-dates grid start and grid completion; mean+/-std band heuristic; not the card's rule; void for E3.",
            "results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json": "post-dates grid; status=DRAFT_PENDING_REVIEWER_APPROVAL; self-declares it MUST NOT be applied retroactively to this run_dir; void for E3.",
        },
        "decisive_rule": (
            "Vault card decisive rule (verbatim, see prereg_provenance.md): for each component "
            "arm X in {frozen_noconv, frozen_nogate, frozen_As4d}, over deep-noise conditions "
            "{awgn@+0dB, awgn@-2dB, awgn@-6dB}: PRIMARY_DRIVER iff Delta(frozen-X) >= +8pp in "
            "ALL three deep-noise bins AND X collapses toward s4d (Delta(X-s4d) <= +5pp in all "
            "three deep-noise bins). CONTRIBUTOR iff mean of the three deep-noise "
            "Delta(frozen-X) values is in [+3, +8)pp. NOT_DRIVER iff ALL THREE deep-noise "
            "Delta(frozen-X) values individually satisfy |Delta| < +3pp (per-bin, per the "
            "card's '深噪三档 |Δ| < +3pp' phrasing, which — unlike the CONTRIBUTOR clause's "
            "'均值' — does not say mean). An arm meeting none of the three named conditions "
            "is OUT_OF_PREREG_BINS (not silently defaulted to NOT_DRIVER). No single "
            "PRIMARY_DRIVER but >=2 CONTRIBUTOR arms => overall DISTRIBUTED_ACROSS_COMPONENTS. "
            "Significance-vote or any non-preregistered criterion is explicitly forbidden by "
            "the card."
        ),
        "last_effective_reviewer_verdict_file": ".orchestrate/0705-160159/r1_review.json",
        "last_effective_reviewer_verdict": "REVISE",
        "last_effective_reviewer_verdict_note": (
            "The prior artifact at results/secondary_adjudication_20260705-1619/"
            "secondary_decision.json was generated to address r1_review.json's 3 critical "
            "issues and cited r1 as its 'current_task_reviewer_verdict_file'. r1 is now the "
            "last-effective (superseded-by-being-addressed) verdict, since "
            ".orchestrate/0705-160159/r2_review.json — the review of that 16:19 artifact — is "
            "the current task's review (see 'current_task_reviewer_verdict_file' below)."
        ),
        "current_task_reviewer_verdict_file": ".orchestrate/0705-160159/r2_review.json",
        "current_task_reviewer_verdict": "REVISE",
        "current_task_reviewer_verdict_note": (
            "This is the review whose 1 critical issue (§G2/§G3 enum mismatch: the 16:19 "
            "artifact emitted UNCLASSIFIED_BY_PREREG for frozen_noconv/frozen_As4d instead of "
            "the rubric-legal OUT_OF_PREREG_BINS, even though the underlying Delta reasoning "
            "and overall_verdict gate were already correct) this artifact and its sibling files "
            "fix. It is listed separately from last_effective_reviewer_verdict_file for the "
            "same reason the 16:19 artifact separated its own two reviewer-verdict fields: the "
            "current task's review is the one being remediated right now, not necessarily the "
            "most recently completed review in the whole tree."
        ),
        "reviewer_revise_reason_addressed": (
            "r1/r2 review at .orchestrate/0704-070402 correctly found that the executor's "
            "decision used a non-preregistered mean+/-std band heuristic instead of a "
            "preregistered Delta-pp rule, and explicitly invited: 'Replace the adjudication "
            "artifact with one that mechanically applies an already-registered Delta-pp "
            "threshold, if such a valid prior registration exists' (.orchestrate/0704-070402/"
            "r1_review.json). This decision does exactly that, using the rule already present "
            "in the vault card since 2026-07-03 (predating the grid) rather than inventing a "
            "new one. See e3_human_ruling.json for why the automated reviewer could not see "
            "the card itself (structural scope limitation, not an error). Separately, "
            ".orchestrate/0705-160159/r1_review.json found that the first mechanical-rule "
            "attempt (results/secondary_adjudication_20260705-1610/secondary_decision.json) "
            "misapplied NOT_DRIVER via an unregistered mean fallback and cited a stale "
            "reviewer verdict; both were fixed in results/secondary_adjudication_20260705-1619/"
            "secondary_decision.json. Then .orchestrate/0705-160159/r2_review.json found that "
            "the 16:19 fix's fallback enum (UNCLASSIFIED_BY_PREREG) was not the rubric-legal "
            "value (OUT_OF_PREREG_BINS) for arms meeting none of the three named per-arm "
            "conditions; fixed in this artifact and in classify_arm()'s fallback branch and "
            "ALLOWED_ARM_VERDICTS in apply_secondary_vault_prereg.py."
        ),
        "per_component_verdicts": per_component_verdicts,
        "per_component_evidence": per_component_evidence,
        "overall_verdict": overall_verdict,
        "overall_reason": overall_reason,
        "supersedes": [
            "results/secondary_analysis_20260704-0708/secondary_decision.json",
            "results/secondary_e3adjudication_20260704-0719/e3_adjudication.json",
            "results/secondary_e3ruling_20260704-2206/secondary_decision_corrected.json",
            "results/secondary_analysis_20260704-2223_bandprereg_nodecision/secondary_no_decision.json",
            "results/secondary_deltapp_analysis_20260704-2224/secondary_decision.json",
            "results/secondary_adjudication_20260705-1610/secondary_decision.json",
            "results/secondary_adjudication_20260705-1619/secondary_decision.json",
        ],
        "supersedes_note": (
            "The seven artifacts above remain in place, unmodified, as historical record "
            "(guardrail: never overwrite/delete results/**). Each is superseded for the E3 "
            "question by this artifact; see superseded_decisions_manifest.json in this "
            "directory for the reason each one is superseded, including why the immediately "
            "prior 2026-07-05 16:19 artifact itself needed superseding "
            "(.orchestrate/0705-160159/r2_review.json)."
        ),
    }

    out_dir = WORKDIR / "results" / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "secondary_decision.json"
    out_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    out_path.chmod(0o444)
    print(f"wrote {out_path}")
    print(f"overall_verdict={overall_verdict}  reason={overall_reason}")
    for arm, v in per_component_verdicts.items():
        print(f"  {arm}: {v['verdict']}")


if __name__ == "__main__":
    main()
