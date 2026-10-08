"""One-at-a-time sensitivity scan around a configuration: every tuned parameter is varied over its full search range
while the others stay fixed, and the change of the held-out metrics is recorded.

Usage:
  python scripts/sensitivity_scan.py z_test.parquet SUMMARY.json STUDY TRIAL -o scan.json
(SUMMARY.json from analyze_studies.py, for the search space and the trial parameters.)
"""

import argparse
import json
import math

import numpy as np
import optuna
from omegaconf import OmegaConf

from hpys import tuning

N_POINTS = 9


def scan_values(dist):
    if dist["type"] == "categorical":
        return [{"True": True, "False": False}.get(c, c) for c in dist["choices"]]
    lo, hi = dist["low"], dist["high"]
    if dist["log"]:
        return [float(v) for v in np.exp(np.linspace(math.log(lo), math.log(hi), N_POINTS))]
    return [float(v) for v in np.linspace(lo, hi, N_POINTS)]


def gate(name):
    """The categorical setting that makes a conditional parameter active."""
    if "_massScaling_" in name:
        return {name.split("_massScaling_")[0] + "_massScaling": True}
    if name.startswith("strip_static_"):
        return {"strip_stripSize": "static"}
    if name.startswith(("strip_theta_", "strip_phi_")):
        return {"strip_stripSize": "dynamic"}
    if name in ("1Prong2Pi0_minStripMass", "1Prong2Pi0_maxStripMass"):
        return {"1Prong2Pi0_stripMassWindow": True}
    if name.startswith("strip_minElectronPt"):
        return {"strip_useElectrons": True}
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("summary")
    parser.add_argument("study")
    parser.add_argument("trial", type=int)
    parser.add_argument("--config", default=tuning.DEFAULT_CONFIG)
    parser.add_argument("-o", "--output", default="scan.json")
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    dists = summary["distributions"]
    study = optuna.load_study(study_name=args.study, storage="sqlite:///optimization/seeds/%s.db" % args.study)
    base_params = dict(study.trials[args.trial].params)
    base_cfg = OmegaConf.load(args.config)
    defaults = tuning.default_params(base_cfg)
    # Values for parameters that are inactive in the base trial, used when a scan switches them on
    fill = {}
    for name, dist in dists.items():
        if name in defaults:
            fill[name] = defaults[name]
        elif dist["type"] == "float":
            fill[name] = math.sqrt(dist["low"] * dist["high"]) if dist["log"] else 0.5 * (dist["low"] + dist["high"])
    _, heldout = tuning.load_samples(args.data)

    def run(params):
        full = dict(fill)
        full.update(params)
        return tuning.evaluate(tuning.suggest_config(optuna.trial.FixedTrial(full), base_cfg), heldout)

    base = run(base_params)
    out = {"study": args.study, "trial": args.trial, "base_params": base_params, "base": base, "scans": {}}
    for name in sorted(dists):
        points = []
        for value in scan_values(dists[name]):
            params = dict(base_params)
            if name not in base_params:
                params.update(gate(name))  # scan the parameter with the option that uses it switched on
            params[name] = value
            m = run(params)
            points.append({"value": value if not isinstance(value, (np.floating,)) else float(value),
                           "balanced_accuracy": m["balanced_accuracy"], "p_resolution": m["p_resolution"]})
        out["scans"][name] = {"active": name in base_params, "points": points}
        bA = [p["balanced_accuracy"] for p in points]
        pr = [p["p_resolution"] for p in points]
        print("%-42s %s  bA %.4f..%.4f  pRes %.4f..%.4f" % (name, "active  " if name in base_params else "gated on",
                                                            min(bA), max(bA), min(pr), max(pr)), flush=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
