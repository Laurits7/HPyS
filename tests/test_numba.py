"""Checks that the numba implementation (hpys.hps) reproduces the reference implementation (hpys.reference)."""

import itertools
import math
import os

import awkward as ak
import numpy as np
import pytest
from omegaconf import OmegaConf

from hpys import hps, reference

CONFIG_PATH = os.path.join(os.path.dirname(hps.__file__), "config", "hps.yaml")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "z_test.parquet")

CONFIG_VARIANTS = [
    {},
    {"builder.StripAlgo.stripSize": "static"},
    {"builder.kinematicVariable": "p", "builder.candidateRanking": ["pt", "isolation"]},
    {
        "builder.StripAlgo.updateStripAfterEachCand": True,
        "builder.StripAlgo.maxStripBuildIterations": 1,
        "builder.isolation.maxStripOuterFraction": None,
        "builder.decayModes.1Prong2Pi0.minStripMass": -1.0e3,
    },
]

PDGS = [211, -211, 22, 22, 22, 11, -11, 130, 13]


def load_cfg(overrides):
    cfg = OmegaConf.load(CONFIG_PATH)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    return cfg


def random_cands(rng, theta0, phi0, n, spread):
    pdgs = rng.choice(PDGS, size=n)
    charges = np.where(np.isin(pdgs, [22, 130]), 0.0, np.where(pdgs > 0, 1.0, -1.0))
    charges = np.where(np.abs(pdgs) == 211, np.sign(pdgs), charges)
    theta = np.clip(theta0 + rng.normal(0.0, spread, n), 0.05, math.pi - 0.05)
    phi = phi0 + rng.normal(0.0, spread, n)
    pt = rng.exponential(4.0, n) + 0.2
    eta = -np.log(np.tan(theta / 2.0))
    mass = np.where(np.abs(pdgs) == 211, 0.13957, np.where(np.abs(pdgs) == 130, 0.4976, 0.0))
    energy = np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2)
    return {"pt": pt, "eta": eta, "phi": phi, "energy": energy, "pdg": pdgs.astype(np.int64), "q": charges}


def random_data(seed, n_jets=60):
    rng = np.random.default_rng(seed)
    jets, cands, events = [], [], []
    for _ in range(n_jets):
        theta0, phi0 = rng.uniform(0.3, math.pi - 0.3), rng.uniform(-math.pi, math.pi)
        c = random_cands(rng, theta0, phi0, int(rng.integers(1, 12)), rng.choice([0.01, 0.03, 0.1]))
        extra = random_cands(rng, theta0, phi0, int(rng.integers(0, 10)), 0.4)
        e = {k: np.concatenate([c[k], extra[k]]) for k in c}
        px = np.sum(c["pt"] * np.cos(c["phi"]))
        py = np.sum(c["pt"] * np.sin(c["phi"]))
        pz = np.sum(c["pt"] * np.sinh(c["eta"]))
        E = np.sum(c["energy"])
        jet_pt = math.hypot(px, py)
        jets.append({"pt": jet_pt, "eta": math.asinh(pz / jet_pt), "phi": math.atan2(py, px), "energy": E})
        cands.append(c)
        events.append(e)

    def p4s(collection):
        return ak.zip({k: [list(x[k]) for x in collection] for k in ["pt", "eta", "phi", "energy"]})

    def attr(collection, key):
        return [list(x[key]) for x in collection]

    def ip(collection, scale):
        return [list(rng.normal(0.0, scale, len(x["pt"]))) for x in collection]

    data = {
        "reco_jet_p4": ak.zip({k: [j[k] for j in jets] for k in ["pt", "eta", "phi", "energy"]}),
        "reco_cand_p4s": p4s(cands),
        "reco_cand_pdgs": attr(cands, "pdg"),
        "reco_cand_charges": attr(cands, "q"),
        "event_reco_cand_p4s": p4s(events),
        "event_reco_cand_pdgs": attr(events, "pdg"),
        "event_reco_cand_charges": attr(events, "q"),
    }
    for prefix, collection in [("reco_cand", cands), ("event_reco_cand", events)]:
        for name, scale in [("dxy", 0.01), ("dxy_error", 0.001), ("dz", 0.01), ("dz_error", 0.001)]:
            data["%s_%s" % (prefix, name)] = ip(collection, scale)
    return ak.Array(data)


def as_sorted_lists(array):
    """Per-jet lists of candidate values (or (px, py, pz) of four-vectors), sorted, as the reference stores signal
    candidates in a set without a defined order."""
    fields = ak.fields(array)
    if fields:
        names = ["px", "py", "pz"] if "px" in fields else ["x", "y", "z"]
        columns = [ak.to_list(array[name]) for name in names]
        return [sorted(zip(*[np.round(c, 6).tolist() for c in jet])) for jet in zip(*columns)]
    return [sorted(np.round(np.asarray(jet, dtype=float), 6).tolist()) for jet in ak.to_list(array)]


def assert_same_output(ref, new):
    assert set(ref) == set(new)
    for field in ref:
        r, n = ref[field], new[field]
        if field == "tau_p4s":
            for name in ["px", "py", "pz", "mass"]:
                np.testing.assert_allclose(np.asarray(getattr(n, name)), np.asarray(getattr(r, name)), atol=1e-6, err_msg=field)
        elif field.startswith(("tauSigCand_", "tauIsoCand_", "tauStrip_")):
            assert as_sorted_lists(r) == as_sorted_lists(n), field
        else:
            np.testing.assert_allclose(np.asarray(n, dtype=float), np.asarray(r, dtype=float), atol=1e-6, err_msg=field)


@pytest.mark.parametrize("overrides", CONFIG_VARIANTS)
@pytest.mark.parametrize("seed", [1, 2])
def test_numba_matches_reference_on_random_jets(seed, overrides):
    data = random_data(seed)
    cfg = load_cfg(overrides)
    new = hps.HPSTauBuilder(cfg).process_jets(data)
    ref = reference.HPSTauBuilder(cfg).process_jets(data)
    assert np.sum(np.asarray(new["tau_decaymode"]) >= 0) > 5, "test sample should contain reconstructed taus"
    assert_same_output(ref, new)


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="z_test.parquet not available")
@pytest.mark.parametrize("overrides", CONFIG_VARIANTS[:2])
def test_numba_matches_reference_on_data(overrides):
    data = ak.from_parquet(DATA_PATH)[:300]
    cfg = load_cfg(overrides)
    assert_same_output(reference.HPSTauBuilder(cfg).process_jets(data), hps.HPSTauBuilder(cfg).process_jets(data))


@pytest.mark.parametrize("n", range(1, 8))
def test_next_combination(n):
    for k in range(1, n + 1):
        combo = np.arange(k)
        combos = [list(combo)]
        while hps.next_combination(combo, k, n):
            combos.append(list(combo))
        assert combos == [list(c) for c in itertools.combinations(range(n), k)]
