"""Hyperparameter optimization of HPS with Optuna.

Objectives, evaluated on generator-level hadronic taus (decay modes 0, 1, 2, 10, 11):
  1. balanced decay-mode accuracy (maximized): mean over the decay modes of the fraction of taus reconstructed in the
     correct decay mode; taus that are not reconstructed count as wrong;
  2. momentum resolution (minimized): half the 16-84% quantile range of p_reco / p_vis_gen, relative to its median,
     for the reconstructed taus.
The data are split into an optimization half and a held-out half for reporting.
"""

import argparse
import json
import os
import time

import awkward as ak
import numpy as np
import optuna
from omegaconf import OmegaConf

from hpys import hps
from hpys import kinematics as g

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config", "hps.yaml")
TARGET_DECAY_MODES = [0, 1, 2, 10, 11]
RANKINGS = {
    "nCh-pt-nStrips-iso": ["numChargedCands", "pt", "numStrips", "isolation"],  # CMSSW
    "pt-nCh-nStrips-iso": ["pt", "numChargedCands", "numStrips", "isolation"],  # "highest pT" (CMS paper)
    "nCh-nStrips-pt-iso": ["numChargedCands", "numStrips", "pt", "isolation"],
    "nStrips-nCh-pt-iso": ["numStrips", "numChargedCands", "pt", "isolation"],
    "nCh-iso-pt": ["numChargedCands", "isolation", "pt"],
}
INPUT_COLUMNS = [
    "reco_jet_p4", "reco_cand_p4s", "reco_cand_pdgs", "reco_cand_charges",
    "reco_cand_dxy", "reco_cand_dxy_error", "reco_cand_dz", "reco_cand_dz_error",
    "event_reco_cand_p4s", "event_reco_cand_pdgs", "event_reco_cand_charges",
    "event_reco_cand_dxy", "event_reco_cand_dxy_error", "event_reco_cand_dz", "event_reco_cand_dz_error",
    "gen_jet_tau_decaymode", "gen_jet_tau_p4",
]  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------


class Sample:
    """Prepared HPS inputs together with the generator-level truth."""

    def __init__(self, data):
        self.inputs = hps.prepare_inputs(data)
        self.gen_dm = np.asarray(ak.fill_none(data["gen_jet_tau_decaymode"], -1), dtype=np.int64)
        gen_p4 = g.reinitialize_p4(ak.fill_none(data["gen_jet_tau_p4"], 0.0))
        self.gen_p = np.asarray(gen_p4.p, dtype=np.float64)
        self.is_target = np.isin(self.gen_dm, TARGET_DECAY_MODES) & (self.gen_p > 0.0)

    def __len__(self):
        return len(self.gen_dm)


def load_samples(path, split_fraction=0.5):
    """Splits the file into an optimization part (first split_fraction of the jets) and a held-out part."""
    data = ak.from_parquet(path, columns=INPUT_COLUMNS)
    n_opt = int(len(data) * split_fraction)
    return Sample(data[:n_opt]), Sample(data[n_opt:])


# ---------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------


def evaluate(cfg, sample):
    """Runs HPS with the given configuration and returns the metrics."""
    P, DM, RANK = hps.build_params(cfg.builder)
    found, taus, *_ = hps.run(sample.inputs, P, DM, RANK)
    reco_dm = np.where(found, taus[:, hps.T_DM], -1).astype(np.int64)
    reco_p = np.sqrt(taus[:, hps.T_PX] ** 2 + taus[:, hps.T_PY] ** 2 + taus[:, hps.T_PZ] ** 2)
    return metrics_from_arrays(sample.gen_dm, sample.gen_p, found, reco_dm, reco_p)


def metrics_from_arrays(gen_dm, gen_p, found, reco_dm, reco_p):
    """Metrics of reconstructed taus (decay mode reco_dm, momentum reco_p) against the generator-level truth."""
    found = np.asarray(found, dtype=bool)
    target = np.isin(gen_dm, TARGET_DECAY_MODES) & (gen_p > 0.0)
    metrics = {}
    per_dm = []
    for dm in TARGET_DECAY_MODES:
        mask = target & (gen_dm == dm)
        accuracy = float(np.mean(reco_dm[mask] == dm)) if mask.any() else float("nan")
        metrics["accuracy_dm%i" % dm] = accuracy
        per_dm.append(accuracy)
    metrics["balanced_accuracy"] = float(np.nanmean(per_dm))
    metrics["accuracy"] = float(np.mean(reco_dm[target] == gen_dm[target]))
    metrics["efficiency"] = float(np.mean(found[target]))

    reco = target & found
    ratio = reco_p[reco] / gen_p[reco]
    q16, q50, q84 = np.quantile(ratio, [0.16, 0.50, 0.84]) if reco.any() else (np.nan, np.nan, np.nan)
    metrics["p_scale"] = float(q50)
    metrics["p_resolution"] = float(0.5 * (q84 - q16) / q50)
    return metrics


# ---------------------------------------------------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------------------------------------------------


def suggest_config(trial, base_cfg):
    """Samples a configuration. Parameters that are not tuned keep their values from base_cfg."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    b = cfg.builder
    b.kinematicVariable = trial.suggest_categorical("kinematicVariable", ["pt", "p"])
    b.candidateRanking = RANKINGS[trial.suggest_categorical("candidateRanking", list(RANKINGS))]
    b.matchingConeSize = trial.suggest_float("matchingConeSize", 0.02, 0.4)
    b.signalCone.scale = trial.suggest_float("signalCone_scale", 0.5, 20.0)
    b.signalCone["min"] = trial.suggest_float("signalCone_min", 0.005, 0.15)
    b.signalCone["max"] = b.signalCone["min"] + trial.suggest_float("signalCone_maxMinusMin", 0.0, 0.5)

    b.signalCands.minChargedHadronPt = trial.suggest_float("signal_minChargedHadronPt", 0.0, 2.0)
    b.signalCands.minElectronPt = trial.suggest_float("signal_minElectronPt", 0.0, 2.0)
    b.signalCands.minLeadChargedCandPt = trial.suggest_float("signal_minLeadChargedCandPt", 0.0, 5.0)

    s = b.StripAlgo
    s.minGammaPtSeed = trial.suggest_float("strip_minGammaPtSeed", 0.0, 3.0)
    s.minGammaPtAdd = trial.suggest_float("strip_minGammaPtAdd", 0.0, 3.0)
    s.useElectrons = trial.suggest_categorical("strip_useElectrons", [True, False])
    if s.useElectrons:
        s.minElectronPtSeed = trial.suggest_float("strip_minElectronPtSeed", 0.0, 3.0)
        s.minElectronPtAdd = trial.suggest_float("strip_minElectronPtAdd", 0.0, 3.0)
    s.minStripPt = trial.suggest_float("strip_minStripPt", 0.0, 5.0)
    s.updateStripAfterEachCand = trial.suggest_categorical("strip_updateStripAfterEachCand", [False, True])
    s.stripSize = trial.suggest_categorical("strip_stripSize", ["dynamic", "static"])
    if s.stripSize == "dynamic":
        for axis, a_max in [("theta", 0.5), ("phi", 0.7)]:
            d = s.dynamic[axis]
            d.a = trial.suggest_float("strip_%s_a" % axis, 0.002, a_max, log=True)
            d.b = trial.suggest_float("strip_%s_b" % axis, 0.0, 1.5)
            d["min"] = trial.suggest_float("strip_%s_min" % axis, 0.0005, 0.10, log=True)
            d["max"] = d["min"] + trial.suggest_float("strip_%s_maxMinusMin" % axis, 0.0, 0.8)
    else:
        s.static.theta = trial.suggest_float("strip_static_theta", 0.002, 0.20, log=True)
        s.static.phi = trial.suggest_float("strip_static_phi", 0.002, 0.40, log=True)

    modes = b.decayModes
    for name, min_range, max_range in [
        ("1Prong1Pi0", (0.0, 0.6), (0.7, 4.0)),
        ("1Prong2Pi0", (0.0, 0.8), (0.8, 4.0)),
        ("3Prong0Pi0", (0.0, 1.0), (1.0, 4.0)),
        ("3Prong1Pi0", (0.0, 1.2), (1.2, 4.0)),
    ]:
        mode = modes[name]
        mode.minTauMass = trial.suggest_float("%s_minTauMass" % name, *min_range)
        mode.maxTauMass = trial.suggest_float("%s_maxTauMass" % name, *max_range)
        if mode.numStrips > 0:
            if trial.suggest_categorical("%s_massScaling" % name, [False, True]):
                ref = trial.suggest_float("%s_massScaling_refValue" % name, 0.5, 200.0, log=True)
                headroom = trial.suggest_float("%s_massScaling_maxLimitMinusMax" % name, 0.0, 4.0)
                mode.maxTauMassScaling = {"refValue": ref, "maxLimit": mode.maxTauMass + headroom}
            else:
                mode.maxTauMassScaling = None
            mode.stripMassCorrection = trial.suggest_categorical("%s_stripMassCorrection" % name, [False, True])
    if trial.suggest_categorical("1Prong2Pi0_stripMassWindow", [False, True]):
        modes["1Prong2Pi0"].minStripMass = trial.suggest_float("1Prong2Pi0_minStripMass", 0.0, 0.12)
        modes["1Prong2Pi0"].maxStripMass = trial.suggest_float("1Prong2Pi0_maxStripMass", 0.14, 1.0)
    else:
        modes["1Prong2Pi0"].minStripMass = -1.0e3
        modes["1Prong2Pi0"].maxStripMass = 1.0e3
    return cfg


def default_params(base_cfg):
    """Parameters of the search space that reproduce base_cfg, to start the study from the default configuration."""
    b = base_cfg.builder
    s = b.StripAlgo
    params = {
        "kinematicVariable": b.kinematicVariable,
        "candidateRanking": next(k for k, v in RANKINGS.items() if v == list(b.candidateRanking)),
        "matchingConeSize": b.matchingConeSize,
        "signalCone_scale": b.signalCone.scale,
        "signalCone_min": b.signalCone["min"],
        "signalCone_maxMinusMin": b.signalCone["max"] - b.signalCone["min"],
        "signal_minChargedHadronPt": b.signalCands.minChargedHadronPt,
        "signal_minElectronPt": b.signalCands.minElectronPt,
        "signal_minLeadChargedCandPt": b.signalCands.minLeadChargedCandPt,
        "strip_minGammaPtSeed": s.minGammaPtSeed,
        "strip_minGammaPtAdd": s.minGammaPtAdd,
        "strip_useElectrons": s.useElectrons,
        "strip_minElectronPtSeed": s.minElectronPtSeed,
        "strip_minElectronPtAdd": s.minElectronPtAdd,
        "strip_minStripPt": s.minStripPt,
        "strip_updateStripAfterEachCand": s.updateStripAfterEachCand,
        "strip_stripSize": s.stripSize,
        "strip_static_theta": s.static.theta,
        "strip_static_phi": s.static.phi,
    }
    for axis in ["theta", "phi"]:
        d = s.dynamic[axis]
        params.update(
            {
                "strip_%s_a" % axis: d.a,
                "strip_%s_b" % axis: d.b,
                "strip_%s_min" % axis: d["min"],
                "strip_%s_maxMinusMin" % axis: d["max"] - d["min"],
            }
        )
    for name, mode in b.decayModes.items():
        if name == "1Prong0Pi0":
            continue
        params["%s_minTauMass" % name] = mode.minTauMass
        params["%s_maxTauMass" % name] = mode.maxTauMass
        if mode.numStrips > 0:
            scaling = mode.get("maxTauMassScaling", None)
            params["%s_massScaling" % name] = scaling is not None
            if scaling is not None:
                params["%s_massScaling_refValue" % name] = scaling.refValue
                params["%s_massScaling_maxLimitMinusMax" % name] = scaling.maxLimit - mode.maxTauMass
            params["%s_stripMassCorrection" % name] = bool(mode.get("stripMassCorrection", False))
    dm2 = b.decayModes["1Prong2Pi0"]
    params["1Prong2Pi0_stripMassWindow"] = dm2.minStripMass > -1.0e2
    params["1Prong2Pi0_minStripMass"] = dm2.minStripMass
    params["1Prong2Pi0_maxStripMass"] = dm2.maxStripMass
    return params


# ---------------------------------------------------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------------------------------------------------


def make_objective(base_cfg, sample):
    def objective(trial):
        cfg = suggest_config(trial, base_cfg)
        metrics = evaluate(cfg, sample)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        return metrics["balanced_accuracy"], metrics["p_resolution"]

    return objective


def config_from_trial(trial, base_cfg):
    return suggest_config(optuna.trial.FixedTrial(trial.params), base_cfg)


def report(study, base_cfg, heldout, output_dir):
    """Evaluates the default configuration and the Pareto-optimal trials on the held-out sample and writes their configs."""
    os.makedirs(output_dir, exist_ok=True)
    rows = [("default", evaluate(base_cfg, heldout))]
    for trial in sorted(study.best_trials, key=lambda t: -t.values[0]):
        cfg = config_from_trial(trial, base_cfg)
        OmegaConf.save(cfg, os.path.join(output_dir, "trial_%05i.yaml" % trial.number))
        rows.append(("trial %i" % trial.number, evaluate(cfg, heldout)))
    header = "%-12s %8s %8s %8s %6s %6s %6s %6s %6s %8s %8s" % (
        "config", "balAcc", "acc", "eff", "DM0", "DM1", "DM2", "DM10", "DM11", "p_res", "p_scale")
    lines = [header]
    for name, m in rows:
        lines.append(
            "%-12s %8.4f %8.4f %8.4f %6.3f %6.3f %6.3f %6.3f %6.3f %8.4f %8.4f"
            % (name, m["balanced_accuracy"], m["accuracy"], m["efficiency"], *[m["accuracy_dm%i" % d] for d in TARGET_DECAY_MODES],
               m["p_resolution"], m["p_scale"])
        )  # fmt: skip
    text = "\n".join(lines)
    with open(os.path.join(output_dir, "pareto_heldout.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(output_dir, "pareto_heldout.json"), "w") as f:
        json.dump({name: m for name, m in rows}, f, indent=2)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="Parquet file with HPS inputs and generator-level truth")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Base configuration")
    parser.add_argument("--output-dir", default="optimization", help="Directory for the study database and results")
    parser.add_argument("--study-name", default="hps")
    parser.add_argument("--n-trials", type=int, default=2000)
    parser.add_argument("--sampler", choices=["tpe", "nsga2"], default="tpe")
    parser.add_argument("--split-fraction", type=float, default=0.5, help="Fraction of jets used for the optimization")
    parser.add_argument("--max-jets", type=int, default=None, help="Use only the first N jets of the file (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-startup-trials", type=int, default=100, help="Random trials before TPE takes over")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.config)
    t0 = time.time()
    data_path = args.data
    if args.max_jets is not None:
        data = ak.from_parquet(args.data, columns=INPUT_COLUMNS)[: args.max_jets]
        n_opt = int(len(data) * args.split_fraction)
        sample, heldout = Sample(data[:n_opt]), Sample(data[n_opt:])
    else:
        sample, heldout = load_samples(data_path, args.split_fraction)
    print("Loaded %i jets for optimization, %i held out (%.1fs)" % (len(sample), len(heldout), time.time() - t0))

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True, n_startup_trials=args.n_startup_trials)
    else:
        sampler = optuna.samplers.NSGAIISampler(seed=args.seed, population_size=100)
    os.makedirs(args.output_dir, exist_ok=True)
    storage = "sqlite:///%s" % os.path.abspath(os.path.join(args.output_dir, "%s.db" % args.study_name))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["maximize", "minimize"],
        sampler=sampler,
        load_if_exists=True,
    )
    study.set_metric_names(["balanced_accuracy", "p_resolution"])
    if len(study.trials) == 0:
        study.enqueue_trial(default_params(base_cfg))
    study.optimize(make_objective(base_cfg, sample), n_trials=args.n_trials)
    print("Pareto front: %i trials" % len(study.best_trials))
    print(report(study, base_cfg, heldout, os.path.join(args.output_dir, args.study_name)))


if __name__ == "__main__":
    main()
