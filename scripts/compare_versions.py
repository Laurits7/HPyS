"""Compares HPS versions with the ml-tau-model evaluation tools (decay-mode F1 comparison, pT resolution / response
and median ΔR versus generator-level visible pT):
  - HPS_mltau:   the tau_* branches stored in the input file (ml-tau-model HPS, before the fixes in this repository);
  - HPS_CMS2018: HPyS with the default configuration (CMS 2018 parameters);
  - HPS_tuned:   HPyS with a tuned configuration.
Only the held-out part of the file (not used in the optimization) is evaluated.

Needs ml-tau-model on PYTHONPATH (for mltau.tools.evaluation). Usage:
  python scripts/compare_versions.py z_test.parquet TUNED.yaml -o comparison
"""

import argparse
import json
import os

import awkward as ak
import numpy as np
from omegaconf import OmegaConf

from hpys import hps, tuning
from mltau.tools import features as f
from mltau.tools.evaluation import decay_mode as d
from mltau.tools.evaluation import kinematics as k
from mltau.tools.general import reinitialize_p4

ML_TAU_CONFIG = os.path.join(os.path.dirname(os.path.dirname(d.__file__)), "..", "config", "metrics")
STYLES = {
    "HPS_mltau": {"name": "HPS ml-tau-model (tau_* branches)", "marker": "s", "color": "tab:gray", "ls": "dashed", "lw": 3},
    "HPS_CMS2018": {"name": "HPyS, CMS 2018 parameters", "marker": "o", "color": "tab:orange", "ls": "solid", "lw": 3},
    "HPS_tuned": {"name": "HPyS, tuned parameters", "marker": "^", "color": "tab:blue", "ls": "solid", "lw": 3},
    "PerfectCounting": {"name": "Perfect counting of reco particles", "marker": "*", "color": "black", "ls": "dotted", "lw": 2},
}


def load_cfg():
    cfg = OmegaConf.create({"metrics": OmegaConf.load(os.path.join(ML_TAU_CONFIG, "kinematics.yaml"))})
    cfg.metrics.ALGORITHM_PLOT_STYLES = STYLES
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("tuned_config")
    parser.add_argument("--default-config", default=tuning.DEFAULT_CONFIG)
    parser.add_argument("--split-fraction", type=float, default=0.5)
    parser.add_argument("--oracle", default=None, help="Decay modes from scripts/oracle.py (oracle_dm.npy), shown in the F1 plot")
    parser.add_argument("-o", "--output-dir", default="comparison")
    args = parser.parse_args()

    data = ak.from_parquet(args.data, columns=tuning.INPUT_COLUMNS + ["tau_decaymode", "tau_p4s"])
    data = data[int(len(data) * args.split_fraction):]  # held-out part, as in hpys.tuning.load_samples
    taus = {"HPS_mltau": {"tau_decaymode": data["tau_decaymode"], "tau_p4s": data["tau_p4s"]}}
    for name, path in [("HPS_CMS2018", args.default_config), ("HPS_tuned", args.tuned_config)]:
        taus[name] = hps.HPSTauBuilder(OmegaConf.load(path)).process_jets(data)

    cfg = load_cfg()
    sample = "z"
    gen_dm = np.asarray(ak.fill_none(data["gen_jet_tau_decaymode"], -1), dtype=int)
    gen_p4 = reinitialize_p4(data["gen_jet_tau_p4"])
    os.makedirs(args.output_dir, exist_ok=True)

    # Decay mode: all jets with a generator-level tau; jets without a reconstructed tau are predicted as "Rare"
    dm_dir = os.path.join(args.output_dir, "decay_mode")
    dm_evaluators = []
    has_gen_tau = gen_dm >= 0
    for name, t in taus.items():
        pred = np.asarray(ak.fill_none(t["tau_decaymode"], -1), dtype=int)
        dm_evaluators.append(
            d.HardLabelDecayModeEvaluator(predicted=pred[has_gen_tau], truth=gen_dm[has_gen_tau], output_dir=dm_dir,
                                          sample=sample, algorithm=name)  # fmt: skip
        )
    if args.oracle is not None:
        oracle = np.load(args.oracle)
        dm_evaluators.append(
            d.HardLabelDecayModeEvaluator(predicted=oracle[has_gen_tau], truth=gen_dm[has_gen_tau], output_dir=dm_dir,
                                          sample=sample, algorithm="PerfectCounting")  # fmt: skip
        )
    dme = d.DecayModeMultiEvaluator(dm_dir, cfg, sample=sample)
    dme.combine_results(dm_evaluators)
    dme.save()

    # Kinematics: jets with a generator-level tau and a reconstructed tau
    kin_dir = os.path.join(args.output_dir, "kinematics")
    os.makedirs(kin_dir, exist_ok=True)
    kcfg = cfg.metrics.kinematics
    bins = list(kcfg.pt.bin_edges[sample])
    plots = {
        "pt_resolution": k.LinePlot(cfg, kcfg.pt.resolution_plot.xlabel, kcfg.pt.resolution_plot.ylabel,
                                    ymin=kcfg.pt.resolution_plot.ylim[0], ymax=kcfg.pt.resolution_plot.ylim[1]),
        "pt_response": k.LinePlot(cfg, kcfg.pt.response_plot.xlabel, kcfg.pt.response_plot.ylabel,
                                  ymin=kcfg.pt.response_plot.ylim[0], ymax=kcfg.pt.response_plot.ylim[1],
                                  nticks=kcfg.pt.response_plot.nticks, axhline_loc=1.0),
        "deltaR_median": k.LinePlot(cfg, kcfg.deltaR.median_plot.xlabel, kcfg.deltaR.median_plot.ylabel,
                                    ymin=kcfg.deltaR.median_plot.ylim[0], ymax=kcfg.deltaR.median_plot.ylim[1]),
    }  # fmt: skip
    summary = {}
    for name, t in taus.items():
        found = has_gen_tau & (np.asarray(ak.fill_none(t["tau_decaymode"], -1)) >= 0)
        reco = reinitialize_p4(t["tau_p4s"][found])
        gen = gen_p4[found]
        pt_eval = k.RegressionEvaluator(prediction=reco.pt, truth=gen.pt, bin_edges=bins, algorithm=name,
                                        sample_name=sample, variable="pt", mode="ratio")  # fmt: skip
        deltaR = f.deltaR_thetaPhi(theta1=reco.theta, phi1=reco.phi, theta2=gen.theta, phi2=gen.phi)
        dr_eval = k.DeltaREvaluator(deltaR=deltaR, pt_truth=np.asarray(gen.pt), bin_edges=bins, algorithm=name)
        label = STYLES[name]["name"]
        plots["pt_resolution"].add_line(pt_eval.bin_centers, pt_eval.resolutions, name, label=label)
        plots["pt_response"].add_line(pt_eval.bin_centers, pt_eval.responses, name, label=label)
        plots["deltaR_median"].add_line(dr_eval.bin_centers, dr_eval.medians, name, label=label)
        dm_eval = next(e for e in dm_evaluators if e.algorithm == name)
        summary[name] = {
            "n_jets": int(has_gen_tau.sum()),
            "n_reconstructed": int(found.sum()),
            "decay_mode_F1": [float(x) for x in dm_eval.class_performances["F1"]],
            "pt_resolution_IQR_over_median": float(pt_eval.resolution),
            "pt_response_median": float(pt_eval.response),
            "pt_resolution_per_bin": [float(x) for x in pt_eval.resolutions],
            "deltaR_median": float(dr_eval.median),
            "deltaR_median_per_bin": [float(x) for x in dr_eval.medians],
        }
    for key, plot in plots.items():
        plot.save(os.path.join(kin_dir, "%s.pdf" % key))
    with open(os.path.join(args.output_dir, "summary.json"), "w") as out:
        json.dump(summary, out, indent=2)
    print(json.dumps({n: {k_: v for k_, v in s.items() if "per_bin" not in k_} for n, s in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
