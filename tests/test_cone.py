import math

import awkward as ak
import numpy as np
import pytest
from omegaconf import OmegaConf

from hpys import cone

CFG = OmegaConf.load(cone.DEFAULT_CONFIG)

M_PI = 0.13957


def p4(p, theta, phi, mass=0.0):
    px, py, pz = p * math.sin(theta) * math.cos(phi), p * math.sin(theta) * math.sin(phi), p * math.cos(theta)
    return {"px": px, "py": py, "pz": pz, "mass": mass}


def make_data(events, jets):
    """events: per row a list of (p4, pdg, charge); jets: per row the jet p4."""
    cands = ak.zip({k: [[c[0][k] for c in ev] for ev in events] for k in ["px", "py", "pz", "mass"]})
    return {
        "reco_jet_p4": ak.zip({k: [j[k] for j in jets] for k in ["px", "py", "pz", "mass"]}),
        "event_reco_cand_p4s": cands,
        "event_reco_cand_pdgs": ak.Array([[c[1] for c in ev] for ev in events]),
        "event_reco_cand_charges": ak.Array([[float(c[2]) for c in ev] for ev in events]),
    }


def process(events, jets, cfg=CFG):
    return cone.ConeTauBuilder(cfg).process_jets(make_data(events, jets))


@pytest.mark.parametrize(
    "counts, tau_id, dm",
    [((1, 0, 0), 0, 0), ((1, 1, 0), 1, 1), ((1, 2, 0), 2, 1), ((1, 3, 0), 3, 2), ((1, 12, 0), 9, 2),
     ((3, 0, 0), 10, 10), ((3, 1, 0), 11, 11), ((3, 4, 0), 14, 11), ((1, 0, 1), -20, -1), ((3, 2, 1), -21, -1),
     ((2, 0, 0), -1, -1), ((5, 0, 0), -1, -1)],
)  # fmt: skip
def test_tau_id_and_decay_mode(counts, tau_id, dm):
    assert cone.assign_tau_id(*counts) == tau_id
    assert cone.tau_id_to_dm(tau_id) == dm


def test_one_prong_one_pi0():
    pion = (p4(10.0, 1.0, 0.0, M_PI), 211, -1)
    g1 = (p4(3.0, 1.02, 0.01), 22, 0)
    g2 = (p4(2.0, 0.99, -0.02), 22, 0)
    soft = (p4(0.2, 1.0, 0.03), 22, 0)  # below minPhotonP
    far = (p4(5.0, 2.0, 1.0), 22, 0)  # outside the cone
    out = process([[pion, g1, g2, soft, far]], [p4(15.0, 1.0, 0.0)])
    assert out["tau_decaymode"][0] == 1
    assert out["tau_coneID"][0] == 2
    assert out["tau_charge"][0] == -1
    assert out["tau_p4s"][0].p == pytest.approx(np.linalg.norm(
        [sum(c[0][k] for c in [pion, g1, g2]) for k in ["px", "py", "pz"]]))  # fmt: skip


def test_three_prongs_seeded_once():
    pions = [(p4(8.0, 1.0, 0.0, M_PI), 211, 1), (p4(5.0, 1.05, 0.05, M_PI), 211, -1), (p4(4.0, 0.97, -0.04, M_PI), 211, 1)]
    out = process([pions], [p4(17.0, 1.0, 0.0)])
    assert out["tau_decaymode"][0] == 10
    assert out["tau_nCharged"][0] == 3


def test_neutral_hadron_and_charge():
    pion = (p4(10.0, 1.0, 0.0, M_PI), 211, 1)
    nh = (p4(5.0, 1.02, 0.0, 0.5), 130, 0)
    out = process([[pion, nh]], [p4(15.0, 1.0, 0.0)])
    assert out["tau_coneID"][0] == -20
    assert out["tau_decaymode"][0] == -1
    # two charged hadrons with the same sign: |q| = 2, rejected and not matched to the jet
    out = process([[pion, (p4(5.0, 1.02, 0.0, M_PI), 211, 1)]], [p4(15.0, 1.0, 0.0)])
    assert out["tau_coneID"][0] == -1000
    assert out["tau_p4s"][0].p == 0.0


def test_jet_matching():
    pion = (p4(10.0, 1.0, 0.0, M_PI), 211, 1)
    other = (p4(10.0, 2.0, 2.5, M_PI), 211, -1)
    out = process([[pion, other], [pion, other]], [p4(10.0, 2.0, 2.5), p4(10.0, 1.0, 1.0)])
    assert out["tau_p4s"][0].theta == pytest.approx(2.0)
    assert out["tau_coneID"][1] == -1000  # no tau within matchingConeSize


@pytest.mark.parametrize("fsr", [True, False])
def test_fsr_photon_removal(fsr):
    cfg = OmegaConf.merge(CFG, {"builder": {"pionPhotonFSR": CFG.builder.pionPhotonFSR if fsr else None}})
    pion = (p4(10.0, 1.0, 0.0, M_PI), 211, 1)
    shower = (p4(0.4, 1.01, 0.01), 22, 0)  # p_gamma / p_pion < softFrac
    hard = (p4(5.0, 1.2, 0.1), 22, 0)  # m(pi + gamma) > massMin
    out = process([[pion, shower, hard]], [p4(15.0, 1.0, 0.0)], cfg)
    assert out["tau_decaymode"][0] == (0 if fsr else 1)
