"""Compares several Optuna studies of the same search space (e.g. different sampler seeds) and evaluates their best
configurations on the held-out sample.

Writes a JSON summary used for the tuning report:
  - per study: number of trials, convergence of the best balanced accuracy, Pareto front (optimization and held-out
    metrics), fANOVA parameter importances, the parameters of the best trials;
  - the baselines (default configuration and, if given, the stored tau_* branches of the input file);
  - the search space (distributions) of the studies.

Usage:
  python scripts/analyze_studies.py z_test.parquet optimization/seeds/tpe_seed{1..8}.db -o summary.json
"""

import argparse
import json
import os
import time

import awkward as ak
import numpy as np
import optuna
from omegaconf import OmegaConf

from hpys import kinematics as g
from hpys import tuning

TOP_FRACTION = 0.01  # trials used for the parameter-spread of the "best" region of each study


def stored_metrics(path, n_opt):
    """Metrics of the tau_* branches stored in the input file, on the held-out part."""
    data = ak.from_parquet(path, columns=["gen_jet_tau_decaymode", "gen_jet_tau_p4", "tau_decaymode", "tau_p4s"])[n_opt:]
    gen_dm = np.asarray(ak.fill_none(data["gen_jet_tau_decaymode"], -1), dtype=np.int64)
    gen_p = np.asarray(g.reinitialize_p4(ak.fill_none(data["gen_jet_tau_p4"], 0.0)).p, dtype=float)
    reco_dm = np.asarray(ak.fill_none(data["tau_decaymode"], -1), dtype=np.int64)
    reco_p = np.asarray(g.reinitialize_p4(ak.fill_none(data["tau_p4s"], 0.0)).p, dtype=float)
    return tuning.metrics_from_arrays(gen_dm, gen_p, reco_dm >= 0, reco_dm, reco_p)


def distribution_dict(dist):
    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        return {"type": "categorical", "choices": [str(c) for c in dist.choices]}
    return {"type": "float", "low": dist.low, "high": dist.high, "log": bool(dist.log)}


def trial_dict(trial):
    return {
        "number": trial.number,
        "values": list(trial.values),
        "params": {k: (v if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)) for k, v in trial.params.items()},
        "metrics": dict(trial.user_attrs),
    }


def analyze_study(db_path, base_cfg, heldout):
    name = os.path.splitext(os.path.basename(db_path))[0]
    study = optuna.load_study(study_name=name, storage="sqlite:///%s" % os.path.abspath(db_path))
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = np.array([t.values for t in trials])
    numbers = np.array([t.number for t in trials])
    order = np.argsort(numbers)
    values, trials = values[order], [trials[i] for i in order]

    out = {"name": name, "n_trials": len(trials)}
    out["best_balanced_accuracy_vs_trial"] = np.maximum.accumulate(values[:, 0]).tolist()
    out["best_p_resolution_vs_trial"] = np.minimum.accumulate(values[:, 1]).tolist()
    durations = [(t.datetime_complete - t.datetime_start).total_seconds() for t in trials]
    out["median_trial_seconds"] = float(np.median(durations))
    out["wall_hours"] = (trials[-1].datetime_complete - trials[0].datetime_start).total_seconds() / 3600.0

    # Pareto front, evaluated on the held-out sample
    pareto = []
    for t in sorted(study.best_trials, key=lambda t: -t.values[0]):
        d = trial_dict(t)
        d["heldout"] = tuning.evaluate(tuning.config_from_trial(t, base_cfg), heldout)
        pareto.append(d)
    out["pareto"] = pareto

    # Best region: the top TOP_FRACTION of the trials in balanced accuracy
    n_top = max(10, int(TOP_FRACTION * len(trials)))
    top = np.argsort(-values[:, 0], kind="stable")[:n_top]
    out["top_trials"] = [trial_dict(trials[i]) for i in top]
    out["top_balanced_accuracy_range"] = [float(values[top, 0].min()), float(values[top, 0].max())]

    out["importances"] = {}
    for i, metric in enumerate(["balanced_accuracy", "p_resolution"]):
        t0 = time.time()
        evaluator = optuna.importance.FanovaImportanceEvaluator(n_trees=32, max_depth=16, seed=0)
        imp = optuna.importance.get_param_importances(study, evaluator=evaluator, target=lambda t, i=i: t.values[i])
        out["importances"][metric] = {k: float(v) for k, v in imp.items()}
        print("  fANOVA %s: %.0fs" % (metric, time.time() - t0), flush=True)
    distributions = {}
    for t in trials:
        for k, dist in t.distributions.items():
            distributions.setdefault(k, distribution_dict(dist))
    return out, distributions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("studies", nargs="+", help="SQLite files; the study name is the file name without .db")
    parser.add_argument("--config", default=tuning.DEFAULT_CONFIG)
    parser.add_argument("--split-fraction", type=float, default=0.5)
    parser.add_argument("-o", "--output", default="summary.json")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base_cfg = OmegaConf.load(args.config)
    _, heldout = tuning.load_samples(args.data, args.split_fraction)
    n_total = len(ak.from_parquet(args.data, columns=["gen_jet_tau_decaymode"]))
    n_opt = int(n_total * args.split_fraction)  # as in tuning.load_samples

    summary = {
        "n_heldout": len(heldout),
        "baselines": {
            "default": tuning.evaluate(base_cfg, heldout),
            "stored_tau_branches": stored_metrics(args.data, n_opt),
        },
        "default_params": {k: (v if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))
                           for k, v in tuning.default_params(base_cfg).items()},
        "studies": [],
        "distributions": {},
    }  # fmt: skip
    for path in args.studies:
        t0 = time.time()
        print("Analyzing %s" % path, flush=True)
        study_summary, distributions = analyze_study(path, base_cfg, heldout)
        summary["studies"].append(study_summary)
        summary["distributions"].update(distributions)
        print("  done in %.0fs" % (time.time() - t0))
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
