import itertools
import math
import os

import pytest
import vector
from omegaconf import OmegaConf

from hpys import hps, reference

CONFIG_PATH = os.path.join(os.path.dirname(hps.__file__), "config", "hps.yaml")
CFG = OmegaConf.load(CONFIG_PATH)


def make_cand(px, py, pz, mass, pdgId, q, barcode):
    p4 = vector.obj(px=px, py=py, pz=pz, E=math.sqrt(px**2 + py**2 + pz**2 + mass**2))
    return reference.Cand(p4, pdgId, q, 0.0, 1.0e-3, 0.0, 1.0e-3, barcode=barcode)


@pytest.mark.parametrize("n", range(0, 8))
def test_combinatorics(n):
    gen = reference.CombinatoricsGenerator(verbosity=0)
    for k in range(1, n + 2):
        assert gen.generate(k, n) == [list(c) for c in itertools.combinations(range(n), k)]
    assert gen.generate(0, n) == []


def test_strip_mass_is_visible_mass_during_building():
    # two photons with a small opening angle: the visible strip mass must be their invariant mass,
    # not the nominal pi0 mass, so that the strip-mass window in buildTau has an effect
    g1 = make_cand(10.0, 0.0, 0.0, 0.0, 22, 0.0, 0)
    g2 = make_cand(5.0, 0.15, 0.0, 0.0, 22, 0.0, 1)
    strips = reference.StripAlgo(CFG.builder.StripAlgo).buildStrips([g1, g2])
    assert len(strips) == 1
    expected = (g1.p4 + g2.p4).mass
    assert strips[0].mass == pytest.approx(expected, rel=1e-6)
    assert abs(strips[0].mass - reference.m_pi0) > 1e-3
    # the final signal strips (after cleaning) carry the pi0 mass constraint
    assert reference.Strip(list(strips[0].cands)).mass == pytest.approx(reference.m_pi0, rel=1e-6)


def test_event_iso_cands_restricted_to_isolation_cone():
    pion = make_cand(10.0, 0.0, 0.0, 0.13957, 211, 1.0, 0)
    jet = reference.Jet(pion.p4, [pion.p4], [211], [1.0], [0.0], [1.0e-3], [0.0], [1.0e-3])
    # charged hadron outside the isolation cone of the tau (angle ~0.55), but within isolationConeSize + matchingConeSize
    phi = 0.55
    far = make_cand(5.0 * math.cos(phi), 5.0 * math.sin(phi), 0.0, 0.13957, 211, 1.0, 0)
    near = make_cand(5.0 * math.cos(0.3), 5.0 * math.sin(0.3), 0.0, 0.13957, 211, -1.0, 1)
    tau = reference.HPSAlgo(CFG.builder).buildTau(jet, [far, near])
    assert tau is not None
    assert tau.decayMode == "1Prong0Pi0"
    assert tau.chargedIso_dR0p5 == pytest.approx(near.pt)


def test_process_jets_end_to_end():
    import awkward as ak

    def p4s(rows):
        return ak.zip({k: [[r[i] for r in jet] for jet in rows] for i, k in enumerate(["px", "py", "pz", "mass"])})

    # jet 0: a pi+ and two photons (pi0 -> gamma gamma); jet 1: a lone neutral hadron (no tau)
    cands = [
        [(10.0, 0.0, 0.0, 0.13957), (5.0, 0.1, 0.0, 0.0), (3.0, 0.08, 0.02, 0.0)],
        [(8.0, 1.0, 0.0, 0.4977)],
    ]
    pdgs = [[211, 22, 22], [130]]
    charges = [[1.0, 0.0, 0.0], [0.0]]
    zeros = [[0.0] * len(c) for c in cands]
    errs = [[1.0e-3] * len(c) for c in cands]
    jets = ak.zip(
        {k: [sum(c[i] for c in jet) if k != "mass" else 0.0 for jet in cands] for i, k in enumerate(["px", "py", "pz", "mass"])}
    )
    data = ak.Array(
        {
            "reco_jet_p4": jets,
            "reco_cand_p4s": p4s(cands),
            "reco_cand_pdgs": pdgs,
            "reco_cand_charges": charges,
            "reco_cand_dxy": zeros,
            "reco_cand_dxy_error": errs,
            "reco_cand_dz": zeros,
            "reco_cand_dz_error": errs,
            "event_reco_cand_p4s": p4s(cands),
            "event_reco_cand_pdgs": pdgs,
            "event_reco_cand_charges": charges,
            "event_reco_cand_dxy": zeros,
            "event_reco_cand_dxy_error": errs,
            "event_reco_cand_dz": zeros,
            "event_reco_cand_dz_error": errs,
        }
    )
    out = hps.HPSTauBuilder(CFG).process_jets(data)
    assert out["tau_decaymode"].tolist() == [1, -1]
    assert out["tau_charge"].tolist() == [1.0, 0.0]
    assert out["tau_p4s"][1].pt == 0.0
    assert len(out["tauSigCand_p4s"][0]) == 3


def cfg_with(overrides):
    cfg = OmegaConf.create(OmegaConf.to_container(CFG, resolve=True))
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    return cfg


def photon_at(pt, theta, phi, barcode):
    return make_cand(
        pt * math.cos(phi), pt * math.sin(phi), pt / math.tan(theta), 0.0, 22, 0.0, barcode
    )


def test_dz_keeps_sign():
    assert make_cand(1.0, 0.0, 0.0, 0.0, 211, 1.0, 0).dz == 0.0
    p4 = vector.obj(px=1.0, py=0.0, pz=0.0, E=1.0)
    assert reference.Cand(p4, 211, 1.0, -0.1, 0.01, -0.2, 0.01).dz == -0.2


def test_dynamic_vs_static_strip_size():
    # a soft photon at dPhi = 0.25 from a hard one: outside the static window (0.20),
    # inside the dynamic one (g(1) + g(20) = 0.35 + 0.042 -> clamped to 0.30)
    seed = photon_at(20.0, math.pi / 2, 0.0, 0)
    soft = photon_at(1.5, math.pi / 2, 0.25, 1)
    dynamic = reference.StripAlgo(CFG.builder.StripAlgo).buildStrips([seed, soft])
    static = reference.StripAlgo(cfg_with({"builder.StripAlgo.stripSize": "static"}).builder.StripAlgo).buildStrips([seed, soft])
    assert [len(s.cands) for s in dynamic] == [2]
    assert [len(s.cands) for s in static] == [1, 1]
    # two hard photons at dPhi = 0.1 are merged by the static window, but not by the dynamic one (0.35*(20^-0.71+20^-0.71) = 0.084)
    seed2 = photon_at(20.0, math.pi / 2, 0.1, 1)
    assert len(reference.StripAlgo(CFG.builder.StripAlgo).buildStrips([seed, seed2])) == 2
    assert len(static[0].cands) == 1


def test_max_strip_build_iterations():
    # chain of photons, each reachable only after the strip position has moved towards the previous one
    # window 0.08: iteration 1 adds the photon at 0.05 (strip moves to ~0.025), iteration 2 the one at 0.10
    cands = [photon_at(10.0, math.pi / 2, 0.05 * i, i) for i in range(3)]
    cfg = cfg_with({"builder.StripAlgo.stripSize": "static", "builder.StripAlgo.static.phi": 0.08})
    unlimited = reference.StripAlgo(cfg.builder.StripAlgo).buildStrips(cands)
    cfg = cfg_with({"builder.StripAlgo.stripSize": "static", "builder.StripAlgo.static.phi": 0.08, "builder.StripAlgo.maxStripBuildIterations": 1})
    limited = reference.StripAlgo(cfg.builder.StripAlgo).buildStrips(cands)
    assert len(unlimited) == 1 and len(unlimited[0].cands) == 3
    assert len(limited[0].cands) == 2
    assert [s.barcode for s in limited] == list(range(len(limited)))


def test_signal_cone_size():
    algo = reference.HPSAlgo(CFG.builder)
    for pt, expected in [(10.0, 0.10), (40.0, 0.075), (100.0, 0.05)]:
        assert algo.compSignalConeSize(reference.hpsParticleBase(vector.obj(px=pt, py=0.0, pz=0.0, E=pt))) == pytest.approx(expected)
    algo = reference.HPSAlgo(cfg_with({"builder.kinematicVariable": "p"}).builder)
    # pT = 30, p = 50 -> 3 / 50
    tau = reference.hpsParticleBase(vector.obj(px=30.0, py=0.0, pz=40.0, E=50.0))
    assert algo.compSignalConeSize(tau) == pytest.approx(0.06)


def test_strip_mass_correction_matches_finite_difference():
    pion = make_cand(10.0, 0.0, 2.0, 0.13957, 211, 1.0, 0)
    strip = reference.Strip([photon_at(5.0, 1.3, 0.2, 1)])
    tau = reference.Tau([pion], [strip])
    stripAlgo = reference.StripAlgo(CFG.builder.StripAlgo)
    dTheta, dPhi = stripAlgo.getStripMassCorrectionSize(strip)
    eps = 1.0e-6

    def tau_mass(theta, phi):
        p = strip.p
        p4 = vector.obj(
            px=p * math.sin(theta) * math.cos(phi), py=p * math.sin(theta) * math.sin(phi), pz=p * math.cos(theta), E=strip.energy
        )
        return (pion.p4 + p4).mass

    dm_dTheta = (tau_mass(strip.theta + eps, strip.phi) - tau_mass(strip.theta - eps, strip.phi)) / (2 * eps)
    dm_dPhi = (tau_mass(strip.theta, strip.phi + eps) - tau_mass(strip.theta, strip.phi - eps)) / (2 * eps)
    expected = math.sqrt((dm_dTheta * dTheta) ** 2 + (dm_dPhi * dPhi) ** 2)
    assert reference.comp_stripMassCorrection(tau, stripAlgo) == pytest.approx(expected, rel=1e-4)


def test_mass_window_pt_scaling():
    algo = reference.HPSAlgo(cfg_with({"builder.decayModes.1Prong1Pi0.stripMassCorrection": False}).builder)
    cfgDecayMode = algo.cfg.decayModes["1Prong1Pi0"]

    def tau_with(mass, pt):
        return reference.hpsParticleBase(vector.obj(px=pt, py=0.0, pz=0.0, E=math.sqrt(pt**2 + mass**2)))

    # pT = 400 GeV: upper limit 1.3 * sqrt(4) = 2.6 GeV
    assert algo.passesMassWindow(tau_with(2.5, 400.0), cfgDecayMode)
    assert not algo.passesMassWindow(tau_with(2.7, 400.0), cfgDecayMode)
    # pT = 40 GeV: sqrt(0.4) < 1, upper limit clamped to 1.3 GeV
    assert algo.passesMassWindow(tau_with(1.25, 40.0), cfgDecayMode)
    assert not algo.passesMassWindow(tau_with(1.35, 40.0), cfgDecayMode)


def test_ranking():
    def fake(nCharged, pt, nStrips, iso):
        tau = reference.Tau()
        tau.num_signal_chargedCands, tau.pt, tau.num_signal_strips, tau.combinedIso_dR0p5 = nCharged, pt, nStrips, iso
        return tau

    a, b, c = fake(1, 30.0, 1, 0.0), fake(3, 20.0, 0, 0.0), fake(1, 30.0, 1, 2.0)
    assert reference.rank_tau_candidates([a, b, c], ["numChargedCands", "pt", "numStrips", "isolation"]) == [b, a, c]
    assert reference.rank_tau_candidates([c, b, a], ["pt", "isolation"]) == [a, c, b]
    with pytest.raises(ValueError):
        reference.rank_tau_candidates([a], ["foo"])
