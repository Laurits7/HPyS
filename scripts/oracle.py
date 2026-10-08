"""Reconstruction limits for HPS decay-mode classification, from the generator-level tau daughters matched to the
reconstructed particles of the jet.

1. Track loss: how many of the charged tau daughters (h±) have a matching reconstructed charged candidate
   (one-to-one, closest opening angle first, angle < --charged-angle).
2. Perfect-counting reference ("oracle"): the decay mode HPS would get if it counted the reconstructed particles
   perfectly:
     n_charged = number of charged daughters matched to a reconstructed charged candidate,
     n_pi0     = number of generator-level π0 whose reconstructed photons carry at least a fraction --pi0-fraction of
                 the π0 energy (each reconstructed photon is assigned to the closest visible tau daughter within
                 --photon-angle),
   mapped to the HPS decay modes: 1 charged + 0 / 1 / >=2 π0 -> DM 0 / 1 / 2, 3 charged + 0 / >=1 π0 -> DM 10 / 11,
   anything else -> no tau (-1). With all daughters "reconstructed" (--sanity) the mapping must reproduce the
   generator-level decay mode up to modes outside HPS (kaons, rare decays).

Evaluated on the held-out half of the file, like the tuning.

Usage:
  python scripts/oracle.py z_test.parquet -o optimization/oracle
"""

import argparse
import json
import math
import os

import awkward as ak
import numba as nb
import numpy as np

from hpys import kinematics as g
from hpys import tuning

COLUMNS = [
    "gen_jet_tau_decaymode", "gen_jet_tau_p4", "gen_jet_tau_vis_daughter_p4s", "gen_jet_tau_vis_daughter_pdgs",
    "gen_jet_tau_vis_daughter_charges", "reco_cand_p4s", "reco_cand_pdgs", "reco_cand_charges",
]  # fmt: skip


def flat(p4s, pdgs, charges):
    counts = np.asarray(ak.num(pdgs), dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    p4s = g.reinitialize_p4(p4s)
    px, py, pz, energy = [np.asarray(ak.flatten(getattr(p4s, c)), dtype=np.float64) for c in ["px", "py", "pz", "energy"]]
    return (offsets, px, py, pz, energy, np.asarray(ak.flatten(pdgs), dtype=np.int64),
            np.asarray(ak.flatten(charges), dtype=np.float64))  # fmt: skip


@nb.njit(cache=True)
def angle(px1, py1, pz1, px2, py2, pz2):
    n1 = math.sqrt(px1 * px1 + py1 * py1 + pz1 * pz1)
    n2 = math.sqrt(px2 * px2 + py2 * py2 + pz2 * pz2)
    if n1 <= 0.0 or n2 <= 0.0:
        return 1.0e9
    c = (px1 * px2 + py1 * py2 + pz1 * pz2) / (n1 * n2)
    return math.acos(min(1.0, max(-1.0, c)))


@nb.njit(cache=True)
def hps_dm(n_ch, n_pi0):
    if n_ch == 1:
        return 0 if n_pi0 == 0 else (1 if n_pi0 == 1 else 2)
    if n_ch == 3:
        return 10 if n_pi0 == 0 else 11
    return -1


@nb.njit(parallel=True, cache=True)
def run(g_off, g_px, g_py, g_pz, g_e, g_pdg, g_q, r_off, r_px, r_py, r_pz, r_e, r_pdg, r_q,
        ch_angle, ph_angle, pi0_fraction, sanity):  # fmt: skip
    n_jets = len(g_off) - 1
    n_gen_ch = np.zeros(n_jets, np.int64)
    n_matched_ch = np.zeros(n_jets, np.int64)
    n_reco_ch = np.zeros(n_jets, np.int64)
    n_gen_pi0 = np.zeros(n_jets, np.int64)
    n_found_pi0 = np.zeros(n_jets, np.int64)
    oracle = np.full(n_jets, -1, np.int64)
    lost_p = np.full(n_jets, -1.0)  # momentum of the softest unmatched charged daughter
    for j in nb.prange(n_jets):
        g0, g1, r0, r1 = g_off[j], g_off[j + 1], r_off[j], r_off[j + 1]
        ng, nr = g1 - g0, r1 - r0
        for r in range(r0, r1):
            if r_q[r] != 0.0:
                n_reco_ch[j] += 1
        # charged daughters: one-to-one matching, closest pairs first
        g_used = np.zeros(ng, np.bool_)
        r_used = np.zeros(nr, np.bool_)
        for g in range(g0, g1):
            if g_q[g] != 0.0:
                n_gen_ch[j] += 1
        if sanity:
            n_matched_ch[j] = n_gen_ch[j]
        else:
            while True:
                best, bg, br = ch_angle, -1, -1
                for g in range(g0, g1):
                    if g_q[g] == 0.0 or g_used[g - g0]:
                        continue
                    for r in range(r0, r1):
                        if r_q[r] == 0.0 or r_used[r - r0]:
                            continue
                        a = angle(g_px[g], g_py[g], g_pz[g], r_px[r], r_py[r], r_pz[r])
                        if a < best:
                            best, bg, br = a, g, r
                if bg < 0:
                    break
                g_used[bg - g0] = True
                r_used[br - r0] = True
                n_matched_ch[j] += 1
            for g in range(g0, g1):
                if g_q[g] != 0.0 and not g_used[g - g0]:
                    p = math.sqrt(g_px[g] ** 2 + g_py[g] ** 2 + g_pz[g] ** 2)
                    if lost_p[j] < 0.0 or p < lost_p[j]:
                        lost_p[j] = p
        # π0: reconstructed photons assigned to the closest visible daughter
        e_assigned = np.zeros(ng)
        if not sanity:
            for r in range(r0, r1):
                if r_pdg[r] != 22:
                    continue
                best, bg = ph_angle, -1
                for g in range(g0, g1):
                    a = angle(g_px[g], g_py[g], g_pz[g], r_px[r], r_py[r], r_pz[r])
                    if a < best:
                        best, bg = a, g
                if bg >= 0:
                    e_assigned[bg - g0] += r_e[r]
        for g in range(g0, g1):
            if g_pdg[g] == 111:
                n_gen_pi0[j] += 1
                if sanity or e_assigned[g - g0] >= pi0_fraction * g_e[g]:
                    n_found_pi0[j] += 1
        oracle[j] = hps_dm(n_matched_ch[j], n_found_pi0[j])
    return n_gen_ch, n_matched_ch, n_reco_ch, n_gen_pi0, n_found_pi0, oracle, lost_p


def dm_metrics(gen_dm, gen_p, pred):
    m = tuning.metrics_from_arrays(gen_dm, gen_p, pred >= 0, pred, np.ones_like(gen_p))
    return {k: v for k, v in m.items() if not k.startswith("p_")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("--split-fraction", type=float, default=0.5)
    parser.add_argument("--charged-angle", type=float, default=0.01, help="[rad]")
    parser.add_argument("--photon-angle", type=float, default=0.2, help="[rad]")
    parser.add_argument("--pi0-fraction", type=float, default=0.25)
    parser.add_argument("-o", "--output-dir", default="oracle")
    args = parser.parse_args()

    data = ak.from_parquet(args.data, columns=COLUMNS)
    data = data[int(len(data) * args.split_fraction):]
    gen = flat(data["gen_jet_tau_vis_daughter_p4s"], data["gen_jet_tau_vis_daughter_pdgs"], data["gen_jet_tau_vis_daughter_charges"])
    reco = flat(data["reco_cand_p4s"], data["reco_cand_pdgs"], data["reco_cand_charges"])
    gen_dm = np.asarray(ak.fill_none(data["gen_jet_tau_decaymode"], -1), dtype=np.int64)
    gen_p = np.asarray(g.reinitialize_p4(ak.fill_none(data["gen_jet_tau_p4"], 0.0)).p, dtype=np.float64)
    os.makedirs(args.output_dir, exist_ok=True)
    result = {"settings": vars(args), "n_jets": len(data)}

    # sanity check of the DM mapping with perfect reconstruction
    *_, oracle_gen, _ = run(*gen, *reco, args.charged_angle, args.photon_angle, args.pi0_fraction, True)
    result["sanity_gen_level"] = dm_metrics(gen_dm, gen_p, oracle_gen)

    # track loss vs matching angle
    result["track_loss"] = {}
    for ch_angle in [0.002, 0.005, 0.01, 0.02, 0.05]:
        n_gen_ch, n_matched_ch, n_reco_ch, *_, lost_p = run(*gen, *reco, ch_angle, args.photon_angle, args.pi0_fraction, False)
        row = {}
        for dm in [0, 1, 2, 10, 11]:
            mask = gen_dm == dm
            n_exp = 1 if dm < 10 else 3
            row["dm%i" % dm] = {
                "n": int(mask.sum()),
                "all_matched": float(np.mean(n_matched_ch[mask] == n_exp)),
                "matched_fraction_by_count": {str(c): float(np.mean(n_matched_ch[mask] == c)) for c in range(0, 5)},
                "reco_charged_ge_expected": float(np.mean(n_reco_ch[mask] >= n_exp)),
            }
            if dm >= 10:
                lp = lost_p[mask & (n_matched_ch < 3)]
                row["dm%i" % dm]["lost_track_p_quantiles_10_50_90"] = [float(x) for x in np.quantile(lp, [0.1, 0.5, 0.9])] if len(lp) else []
        result["track_loss"][str(ch_angle)] = row

    # perfect-counting reference vs π0 energy fraction
    result["oracle"] = {}
    for fraction in [0.25, 0.5, 0.75]:
        *_, n_gen_pi0, n_found_pi0, oracle, _ = run(*gen, *reco, args.charged_angle, args.photon_angle, fraction, False)
        result["oracle"][str(fraction)] = dm_metrics(gen_dm, gen_p, oracle)
        if fraction == args.pi0_fraction:
            np.save(os.path.join(args.output_dir, "oracle_dm.npy"), oracle)
            result["pi0_found_fraction"] = {
                "dm%i" % dm: float(n_found_pi0[gen_dm == dm].sum() / max(1, n_gen_pi0[gen_dm == dm].sum())) for dm in [1, 2, 11]
            }
    with open(os.path.join(args.output_dir, "oracle.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
