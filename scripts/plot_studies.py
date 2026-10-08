"""Plots and a text summary of several Optuna studies, from the JSON written by analyze_studies.py.

Usage:
  python scripts/plot_studies.py optimization/seeds/summary.json -o optimization/seeds/report [--determinism FILE]
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DMS = [0, 1, 2, 10, 11]
DM_LABELS = ["h± (0)", "h±π0 (1)", "h±π0π0 (2)", "3h± (10)", "3h±π0 (11)"]
COLORS = plt.cm.tab10(np.arange(10))


def norm_position(value, dist):
    """Position of a value within the search range, 0..1 (on a log scale for log-distributed parameters)."""
    lo, hi = dist["low"], dist["high"]
    if dist["log"]:
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (value - lo) / (hi - lo)


def best_trial(study):
    """The trial with the best balanced accuracy on the optimization half (the Pareto list is sorted by it)."""
    return study["pareto"][0]


def fmt_metrics(m):
    return "%.4f  %.4f  %.4f  %s  %.4f  %.4f" % (
        m["balanced_accuracy"], m["accuracy"], m["efficiency"],
        "  ".join("%.3f" % m["accuracy_dm%i" % d] for d in DMS), m["p_resolution"], m["p_scale"])  # fmt: skip


METRIC_HEADER = "balAcc  accuracy  eff     DM0    DM1    DM2    DM10   DM11   p_res   p_scale"


# ---------------------------------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------------------------------


def plot_convergence(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, st in enumerate(summary["studies"]):
        label = st["name"].replace("tpe_", "")
        x = np.arange(1, st["n_trials"] + 1)
        axes[0].plot(x, st["best_balanced_accuracy_vs_trial"], color=COLORS[i], label=label, lw=1.4)
        axes[1].plot(x, st["best_p_resolution_vs_trial"], color=COLORS[i], label=label, lw=1.4)
    axes[0].set_ylabel("best balanced DM accuracy so far (optimization half)")
    axes[1].set_ylabel("best momentum resolution so far (optimization half)")
    axes[0].set_ylim(0.6, None)
    axes[1].set_ylim(None, 0.06)
    for ax in axes:
        ax.set_xlabel("trial")
        ax.grid(alpha=0.3)
        ax.axvline(200, color="grey", ls=":", lw=1)
    axes[0].legend(ncol=2, fontsize=8, title="TPE seed")
    axes[0].set_title("Convergence: decay-mode accuracy")
    axes[1].set_title("Convergence: momentum resolution")
    fig.text(0.5, 0.005, "dotted line: end of the 200 random start-up trials", ha="center", fontsize=8, color="grey")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "convergence.png"), dpi=130)
    plt.close(fig)


def plot_pareto(summary, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, st in enumerate(summary["studies"]):
        pts = np.array([[p["heldout"]["p_resolution"], p["heldout"]["balanced_accuracy"]] for p in st["pareto"]])
        pts = pts[np.argsort(pts[:, 0])]
        ax.plot(pts[:, 0], pts[:, 1], "o-", ms=3, lw=1, color=COLORS[i], label=st["name"].replace("tpe_", ""), alpha=0.85)
        b = best_trial(st)["heldout"]
        ax.plot(b["p_resolution"], b["balanced_accuracy"], "*", ms=13, color=COLORS[i], mec="k", mew=0.6)
    for key, marker, label in [("default", "s", "HPyS default (CMS 2018)"), ("stored_tau_branches", "D", "stored tau_* (ml-tau-model)")]:
        m = summary["baselines"][key]
        ax.plot(m["p_resolution"], m["balanced_accuracy"], marker, ms=9, color="k", mfc="white", mew=1.5, label=label)
    ax.set_xlabel("momentum resolution  ½(q84 − q16)/median of p_reco/p_vis,gen")
    ax.set_ylabel("balanced decay-mode accuracy")
    ax.set_title("Pareto fronts on the held-out half (★ = best accuracy on the optimization half)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pareto_heldout.png"), dpi=130)
    # zoom on the high-accuracy region
    ax.set_xlim(0.035, 0.065)
    ax.set_ylim(0.62, 0.69)
    ax.set_title("Pareto fronts on the held-out half, zoom")
    fig.savefig(os.path.join(out, "pareto_heldout_zoom.png"), dpi=130)
    plt.close(fig)


def plot_dm_accuracy(summary, out):
    rows = [("stored tau_*", summary["baselines"]["stored_tau_branches"]), ("HPyS default", summary["baselines"]["default"])]
    best = [best_trial(st)["heldout"] for st in summary["studies"]]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.27
    x = np.arange(len(DMS) + 1)
    names = DM_LABELS + ["balanced"]
    for j, (label, m) in enumerate(rows):
        vals = [m["accuracy_dm%i" % d] for d in DMS] + [m["balanced_accuracy"]]
        ax.bar(x + (j - 1) * width, vals, width, label=label, color=["#9a9a9a", "#4d4d4d"][j])
    vals = np.array([[m["accuracy_dm%i" % d] for d in DMS] + [m["balanced_accuracy"]] for m in best])
    ax.bar(x + width, vals.mean(axis=0), width, yerr=[vals.mean(0) - vals.min(0), vals.max(0) - vals.mean(0)],
           capsize=3, color="#2a7ab0", label="tuned (mean of 8 seeds, bars = min–max)")  # fmt: skip
    ax.set_xticks(x, names)
    ax.set_ylabel("fraction reconstructed in the correct decay mode")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Decay-mode accuracy on the held-out half")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "dm_accuracy.png"), dpi=130)
    plt.close(fig)


def float_params(summary):
    dists = summary["distributions"]
    return [k for k in sorted(dists) if dists[k]["type"] == "float"]


def plot_parameter_stability(summary, out):
    dists = summary["distributions"]
    params = float_params(summary)
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(params) + 1.5))
    y = np.arange(len(params))[::-1]
    for yi, name in zip(y, params):
        ax.axhline(yi, color="grey", lw=0.3, alpha=0.4)
        if name in summary["default_params"]:
            try:
                ax.plot(norm_position(float(summary["default_params"][name]), dists[name]), yi, "|", ms=12, color="k", mew=2)
            except ValueError:
                pass
        for i, st in enumerate(summary["studies"]):
            top = [t["params"][name] for t in st["top_trials"] if name in t["params"]]
            if top:
                lo, hi = norm_position(min(top), dists[name]), norm_position(max(top), dists[name])
                ax.plot([lo, hi], [yi + (i - 3.5) * 0.08] * 2, color=COLORS[i], lw=1.2, alpha=0.6)
            b = best_trial(st)["params"].get(name)
            if b is not None:
                ax.plot(norm_position(b, dists[name]), yi + (i - 3.5) * 0.08, "o", ms=4, color=COLORS[i])
    labels = []
    for name in params:
        d = dists[name]
        labels.append("%s  [%g, %g%s]" % (name, d["low"], d["high"], ", log" if d["log"] else ""))
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("position in the search range (0 = lower edge, 1 = upper edge)")
    ax.set_title("Best trial per seed (●), range of its top 1% trials (—), default (|)")
    handles = [plt.Line2D([], [], color=COLORS[i], marker="o", ls="-", label=st["name"].replace("tpe_", ""))
               for i, st in enumerate(summary["studies"])]  # fmt: skip
    ax.legend(handles=handles, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.04 - 1.0 / len(params)))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "parameter_stability.png"), dpi=130)
    plt.close(fig)


def mean_importances(summary, metric):
    names = sorted({k for st in summary["studies"] for k in st["importances"][metric]})
    mat = np.array([[st["importances"][metric].get(k, 0.0) for k in names] for st in summary["studies"]])
    order = np.argsort(-mat.mean(axis=0))
    return [names[i] for i in order], mat[:, order]


def plot_importances(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, metric in zip(axes, ["balanced_accuracy", "p_resolution"]):
        names, mat = mean_importances(summary, metric)
        names, mat = names[:20], mat[:, :20]
        im = ax.imshow(mat.T, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_xticks(range(len(summary["studies"])), [st["name"].replace("tpe_seed", "") for st in summary["studies"]])
        ax.set_xlabel("TPE seed")
        ax.set_title("fANOVA importance: %s (top 20)" % metric, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "importances.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------------------------------------------------


def stability_rows(summary):
    """Per float parameter: default, best value per seed, and the spread of the best values relative to the range."""
    dists = summary["distributions"]
    rows = []
    for name in float_params(summary):
        best = [best_trial(st)["params"].get(name) for st in summary["studies"]]
        present = [b for b in best if b is not None]
        if len(present) < 2:
            spread = float("nan")
        else:
            pos = [norm_position(b, dists[name]) for b in present]
            spread = max(pos) - min(pos)
        rows.append((name, summary["default_params"].get(name), best, spread))
    return rows


def categorical_rows(summary):
    dists = summary["distributions"]
    rows = []
    for name in sorted(k for k in dists if dists[k]["type"] == "categorical"):
        best = [best_trial(st)["params"].get(name) for st in summary["studies"]]
        top_counts = {}
        for st in summary["studies"]:
            for t in st["top_trials"]:
                if name in t["params"]:
                    top_counts[str(t["params"][name])] = top_counts.get(str(t["params"][name]), 0) + 1
        rows.append((name, summary["default_params"].get(name), best, top_counts))
    return rows


def write_summary(summary, out, determinism):
    studies = summary["studies"]
    L = []
    w = L.append
    w("HPS hyperparameter optimization for e+e- : summary of %i independent Optuna studies" % len(studies))
    w("=" * 100)
    w("")
    w("SETUP")
    w("-----")
    w("Data:        z_test.parquet, %i jets. First half (%i jets) used for the optimization, second half (%i jets) held"
      % (2 * summary["n_heldout"], summary["n_heldout"], summary["n_heldout"]))
    w("             out; all numbers marked 'held-out' are from jets never seen during the optimization.")
    w("Truth:       generator-level hadronic taus matched to the jet (gen_jet_tau_decaymode in 0, 1, 2, 10, 11);")
    w("             visible momentum from gen_jet_tau_p4. Jets without a hadronic tau are not used (no fake rate).")
    w("Objectives:  1) balanced decay-mode accuracy (maximize): mean over DM 0/1/2/10/11 of the fraction reconstructed")
    w("                in the correct DM; taus that are not reconstructed count as wrong.")
    w("             2) momentum resolution (minimize): ½(q84 − q16)/median of p_reco / p_vis,gen, reconstructed taus.")
    w("Algorithm:   HPyS numba implementation (hpys/hps.py), validated field by field against the reference")
    w("             implementation (hpys/reference.py) on 3000 data jets x 4 configurations and on random jets.")
    w("Optimizer:   Optuna 5.0, multi-objective TPE sampler (multivariate=True, group=True), 200 random start-up")
    w("             trials, 5000 trials per study, first trial = default configuration. 8 studies that differ only in")
    w("             the sampler seed (1..8), run in parallel with 2 numba threads each (median %.1f s per trial,"
      % np.median([st["median_trial_seconds"] for st in studies]))
    w("             %.1f h per study)." % np.mean([st["wall_hours"] for st in studies]))
    w("Software:    python 3.11, numpy 2.2.6, numba 0.61.2, awkward 2.8.7, vector 1.6.3, optuna 5.0")
    w("             (container pytorch.simg:2025-09-01; optuna from HPyS/.deps).")
    w("Command:     python -m hpys.tuning z_test.parquet --n-trials 5000 --n-startup-trials 200 --seed N \\")
    w("                    --output-dir optimization/seeds --study-name tpe_seedN")
    w("")
    w("Determinism: %s" % (determinism or "not checked"))
    w("")
    w("SEARCH SPACE (%i parameters)" % len(summary["distributions"]))
    w("------------")
    for name, d in sorted(summary["distributions"].items()):
        if d["type"] == "float":
            w("  %-42s float  [%g, %g]%s   default %s" % (name, d["low"], d["high"], " log" if d["log"] else "",
                                                        summary["default_params"].get(name, "-")))  # fmt: skip
        else:
            w("  %-42s categorical %s   default %s" % (name, d["choices"], summary["default_params"].get(name, "-")))
    w("  Notes: 'X_maxMinusMin' parametrizes the upper limit as min + offset so that max >= min always holds;")
    w("         '*_massScaling_maxLimitMinusMax' is the cap of the pT-scaled upper mass limit above maxTauMass;")
    w("         strip parameters only exist for the chosen strip size (dynamic: a, b, min, max; static: theta, phi),")
    w("         massScaling/refValue/maxLimit and the DM2 strip-mass window only when switched on.")
    w("  Not tuned: isolation cone and isolation candidates (do not affect DM or momentum), maxStripOuterFraction")
    w("         (only a flag for the tau ID), maxStripBuildIterations, 1Prong0Pi0 mass window, maxChargedCands/maxStrips.")
    w("")
    w("CHANGES WITH RESPECT TO THE ml-tau-model HPS (the stored tau_* branches)")
    w("------------------------------------------------------------------------")
    for line in [
        "Fixes:",
        " - strips keep their visible mass (sum of the photon four-momenta). ml-tau-model set the strip mass to m(π0) when",
        "   building the strip, so the DM 2 strip-mass window (0.05-0.20) was always passed;",
        " - event-level isolation candidates restricted to the isolation cone around the tau (were all event candidates);",
        " - dz keeps its sign;",
        " - maxStripBuildIterations and the strip iteration counter now take effect (previously the counters were unused);",
        " - minLeadChargedCandPt read from the configuration (signalCands) instead of being hard-coded;",
        " - ties in the candidate ranking resolved deterministically (relative tolerance 1e-9).",
        "New features (all hyperparameters, defaults = CMS JINST 13 (2018) P10005):",
        " - dynamic strip size f = a*x^-b in [min, max] (default; static window is an option);",
        " - pT-scaled upper tau-mass limit (DM 1, 2) and the Δm strip-mass correction of the mass windows (Eq. 5);",
        " - signal cone scale/x in [min, max]; configurable candidate ranking; kinematic variable x = pT or p;",
        " - outer-strip pT sum and the 'pT_strip,outer < 0.1 pT_tau' flag written out (not applied: belongs to the tau ID);",
        " - numba implementation, ~1e4 x faster than the object-oriented one (2 s instead of ~10 CPU-hours for 274k jets).",
        "The ml-tau-model configuration differs from the HPyS default: static strips (0.05 x 0.20), no mass scaling, no Δm.",
    ]:
        w("  " + line)
    w("")
    w("BASELINES (held-out)")
    w("--------------------")
    w("  %-34s %s" % ("", METRIC_HEADER))
    w("  %-34s %s" % ("stored tau_* (ml-tau-model)", fmt_metrics(summary["baselines"]["stored_tau_branches"])))
    w("  %-34s %s" % ("HPyS default (CMS 2018 values)", fmt_metrics(summary["baselines"]["default"])))
    w("")
    w("RESULTS PER STUDY: trial with the best balanced accuracy on the optimization half")
    w("-------------------------------------------------------------------------------")
    w("  %-10s %6s %6s %7s %8s   %s" % ("study", "trial", "Pareto", "opt bA", "opt pRes", "held-out: " + METRIC_HEADER))
    for st in studies:
        b = best_trial(st)
        w("  %-10s %6i %6i %7.4f %8.4f   %s" % (st["name"].replace("tpe_", ""), b["number"], len(st["pareto"]),
                                              b["values"][0], b["values"][1], fmt_metrics(b["heldout"])))  # fmt: skip
    hb = np.array([best_trial(st)["heldout"]["balanced_accuracy"] for st in studies])
    hp = np.array([best_trial(st)["heldout"]["p_resolution"] for st in studies])
    ob = np.array([best_trial(st)["values"][0] for st in studies])
    w("")
    w("  held-out balanced accuracy of the best trials: mean %.4f, std %.4f, min %.4f, max %.4f" % (hb.mean(), hb.std(ddof=1), hb.min(), hb.max()))
    w("  held-out momentum resolution of these trials:  mean %.4f, std %.4f, min %.4f, max %.4f" % (hp.mean(), hp.std(ddof=1), hp.min(), hp.max()))
    w("  optimization-half minus held-out balanced accuracy (overfitting to the optimization half): mean %+.4f" % (ob - hb).mean())
    w("  statistical uncertainty of a balanced accuracy on the held-out half: ~0.002 (binomial, per-DM counts)")
    first_reach = []
    for st in studies:
        curve = np.array(st["best_balanced_accuracy_vs_trial"])
        final = curve[-1]
        first_reach.append((int(np.argmax(curve >= final - 0.005)) + 1, int(np.argmax(curve >= final - 0.001)) + 1))
    w("  trials needed to get within 0.005 / 0.001 of the final best (optimization half): %s"
      % ", ".join("%i/%i" % fr for fr in first_reach))
    w("")
    w("  Best held-out trial among the top-accuracy region (ranked on the held-out half, for information only):")
    for st in studies:
        p = max(st["pareto"], key=lambda t: t["heldout"]["balanced_accuracy"])
        w("    %-8s trial %5i  held-out balAcc %.4f, p_res %.4f" % (st["name"].replace("tpe_", ""), p["number"],
                                                                     p["heldout"]["balanced_accuracy"], p["heldout"]["p_resolution"]))  # fmt: skip
    w("")
    w("STABILITY OF THE BEST PARAMETERS ACROSS SEEDS")
    w("---------------------------------------------")
    w("  Float parameters: best value per seed (seeds 1..8; '-' = parameter inactive in that trial), and the spread of the")
    w("  best values as a fraction of the search range (log scale for log parameters). spread < 0.15: stable; > 0.4:")
    w("  not determined (flat direction, the objective does not care).")
    w("  %-42s %8s  %s  %6s" % ("parameter", "default", " ".join("%7s" % ("s%i" % (i + 1)) for i in range(len(studies))), "spread"))
    for name, default, best, spread in sorted(stability_rows(summary), key=lambda r: (np.nan_to_num(r[3], nan=9), r[0])):
        tag = "stable" if spread < 0.15 else ("FLAT" if spread > 0.4 else "")
        dstr = "%8.3g" % float(default) if isinstance(default, (int, float)) else "%8s" % "-"
        w("  %-42s %s  %s  %6.2f %s" % (name, dstr, " ".join("%7.3g" % b if b is not None else "%7s" % "-" for b in best), spread, tag))
    w("")
    w("  Categorical parameters: best value per seed, and how often each value appears in the top 1% trials of all seeds")
    for name, default, best, counts in categorical_rows(summary):
        w("  %-36s default %-20s best: %s" % (name, default, ", ".join(str(b) for b in best)))
        w("  %-36s top-1%% counts: %s" % ("", ", ".join("%s: %i" % kv for kv in sorted(counts.items(), key=lambda kv: -kv[1]))))
    w("")
    w("PARAMETER IMPORTANCE (fANOVA, mean over the 8 studies; spread = std over studies)")
    w("--------------------------------------------------------------------------------")
    for metric in ["balanced_accuracy", "p_resolution"]:
        names, mat = mean_importances(summary, metric)
        w("  %s:" % metric)
        for k, name in enumerate(names[:12]):
            w("    %2i. %-42s %.3f ± %.3f" % (k + 1, name, mat[:, k].mean(), mat[:, k].std(ddof=1)))
    w("")
    best_study = max(studies, key=lambda st: best_trial(st)["values"][0])
    b = best_trial(best_study)
    w("RECOMMENDED CONFIGURATION")
    w("-------------------------")
    w("  Highest balanced accuracy on the optimization half over all studies: %s trial %i" % (best_study["name"], b["number"]))
    w("  -> optimization/seeds/%s/trial_%05i.yaml" % (best_study["name"], b["number"]))
    w("  held-out: %s" % METRIC_HEADER)
    w("            %s" % fmt_metrics(b["heldout"]))
    w("")
    w("Plots (in this directory): convergence.png, pareto_heldout.png, pareto_heldout_zoom.png, dm_accuracy.png,")
    w("parameter_stability.png, importances.png")
    text = "\n".join(L) + "\n"
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", nargs="+")
    parser.add_argument("-o", "--output-dir", default="report")
    parser.add_argument("--determinism", default=None, help="Text file with the result of the determinism check")
    args = parser.parse_args()
    summary = None
    for path in args.summary:  # several summaries (e.g. one per study) are merged
        with open(path) as f:
            part = json.load(f)
        if summary is None:
            summary = part
        else:
            summary["studies"] += part["studies"]
            summary["distributions"].update(part["distributions"])
    os.makedirs(args.output_dir, exist_ok=True)
    determinism = open(args.determinism).read().strip() if args.determinism and os.path.exists(args.determinism) else None
    plot_convergence(summary, args.output_dir)
    plot_pareto(summary, args.output_dir)
    plot_dm_accuracy(summary, args.output_dir)
    plot_parameter_stability(summary, args.output_dir)
    plot_importances(summary, args.output_dir)
    print(write_summary(summary, args.output_dir, determinism))


if __name__ == "__main__":
    main()
