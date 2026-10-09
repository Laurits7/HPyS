"""Numba implementation of the cone-based tau reconstruction of TausFCCee (https://github.com/Olmichu22/TausFCCee,
modules/tauReco.py: findAllTaus, buildTauFromPion, assignTauID and the 'pion_photon_fsr' extra correction), for the
comparison with HPS.

The algorithm works on all particles of the event:

1. Every charged hadron is a seed. Starting from the seed, the particles within dRMax = sqrt(dtheta^2 + dphi^2) of the
   cone axis are added in the order of the candidate collection: charged hadrons, photons above minPhotonP and neutral
   hadrons above minNeutralHadronP. Electrons and muons are ignored. With coneAxis 'running' the axis is the sum of the
   particles accepted so far, with 'lead' it is the seed.
2. Taus with |q| != 1 are rejected. The others get an ID from the constituent counts: 1 charged hadron + N photons -> N
   (at most 9), 3 charged hadrons + N photons -> 10 + N, with a neutral hadron -> -20 (1 prong) / -21 (3 prongs),
   anything else -> -1.
3. Optionally, photons from FSR or the hadronic shower are removed from 1-prong taus (pionPhotonFSR).
4. Taus closer than duplicateDR to a tau built before are dropped.

To compare with HPS, every jet gets the tau closest to the jet axis within matchingConeSize. Its ID is translated to
the HPS decay modes as in TausFCCee (the photons are counted in pairs): 1 prong + 0 / 1-2 / >=3 photons -> DM 0 / 1 / 2,
3 prongs + 0 / >=1 photons -> DM 10 / 11; the negative IDs give no tau (DM -1).
"""

import math
import os

import awkward as ak
import numba
import numpy as np

from hpys import hps
from hpys import kinematics as g

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config", "cone.yaml")
PDG_NEUTRAL_HADRON = 130
CONE_AXES = {"running": 1, "lead": 0}

# indices of the parameters
(
    P_DR_MAX,
    P_RUNNING_AXIS,
    P_MIN_P,
    P_MIN_PION_P,
    P_MIN_PHOTON_P,
    P_MIN_NH_P,
    P_DUPLICATE_DR,
    P_MATCHING_CONE,
    P_FSR,
    P_FSR_PI0_MASS,
    P_FSR_PI0_WINDOW,
    P_FSR_SOFT_FRAC,
    P_FSR_HARD_P_MIN,
    P_FSR_MASS_MIN,
    P_FSR_MAX_PHOTONS,
    N_PARAMS,
) = range(16)

# indices of the per-tau outputs
(
    T_PX,
    T_PY,
    T_PZ,
    T_E,
    T_CHARGE,
    T_ID,
    T_NUM_CHARGED,
    T_NUM_PHOTONS,
    N_TAU_VARS,
) = range(9)


def build_params(cfg):
    P = np.zeros(N_PARAMS, dtype=np.float64)
    P[P_DR_MAX] = cfg.dRMax
    P[P_RUNNING_AXIS] = CONE_AXES[cfg.coneAxis]
    P[P_MIN_P] = cfg.minP
    P[P_MIN_PION_P] = cfg.minPionP
    P[P_MIN_PHOTON_P] = cfg.minPhotonP
    P[P_MIN_NH_P] = cfg.minNeutralHadronP
    P[P_DUPLICATE_DR] = cfg.duplicateDR
    P[P_MATCHING_CONE] = cfg.matchingConeSize
    fsr = cfg.get("pionPhotonFSR")
    if fsr is not None:
        P[P_FSR] = 1.0
        P[P_FSR_PI0_MASS] = fsr.pi0Mass
        P[P_FSR_PI0_WINDOW] = fsr.pi0MassWindow
        P[P_FSR_SOFT_FRAC] = fsr.softFrac
        P[P_FSR_HARD_P_MIN] = fsr.hardPMin
        P[P_FSR_MASS_MIN] = fsr.massMin
        P[P_FSR_MAX_PHOTONS] = fsr.maxPhotons
    return P


@numba.njit(cache=True)
def theta_phi(px, py, pz):
    # as ROOT's TVector3::Theta / Phi, which are 0 for a null vector
    return math.atan2(math.sqrt(px * px + py * py), pz), math.atan2(py, px)


@numba.njit(cache=True)
def dr_angle(px1, py1, pz1, px2, py2, pz2):
    """TausFCCee's myutils.dRAngle: sqrt(dtheta^2 + dphi^2)."""
    theta1, phi1 = theta_phi(px1, py1, pz1)
    theta2, phi2 = theta_phi(px2, py2, pz2)
    return math.sqrt((theta1 - theta2) ** 2 + hps.deltaPhi(phi1, phi2) ** 2)


@numba.njit(cache=True)
def mass(px, py, pz, E):
    m2 = E * E - (px * px + py * py + pz * pz)
    return math.copysign(math.sqrt(abs(m2)), m2)


@numba.njit(cache=True)
def assign_tau_id(n_pions, n_photons, n_neutrons):
    if n_pions == 1 and n_neutrons == 0:
        return min(n_photons, 9)
    if n_pions == 3 and n_neutrons == 0:
        return n_photons + 10
    if n_pions == 3 and n_neutrons > 0:
        return -21
    if n_pions == 1 and n_neutrons > 0:
        return -20
    return -1


@numba.njit(cache=True)
def tau_id_to_dm(tau_id):
    if 0 <= tau_id < 10:
        n_pi0 = (tau_id + 1) // 2
        return 0 if n_pi0 == 0 else (1 if n_pi0 == 1 else 2)
    if tau_id >= 10:
        return 10 if tau_id == 10 else 11
    return -1


@numba.njit(cache=True)
def remove_fsr_photons(members, n_members, pdg, px, py, pz, E, P):
    """'pion_photon_fsr' correction of a 1-prong tau without neutral hadrons: photons that do not form a pi0 with
    another photon of the tau are removed if they are soft compared to the pion (p_gamma / p_pion < softFrac), or hard
    (p_gamma > hardPMin) with m(pion + gamma) > massMin. members[0] is the seed. Returns the new number of members."""
    lead = members[0]
    photons = np.empty(n_members, np.int64)
    n_photons = 0
    for k in range(1, n_members):
        if abs(pdg[members[k]]) == 22:
            photons[n_photons] = k
            n_photons += 1
    if n_photons == 0:
        return n_members
    paired = np.zeros(n_photons, np.bool_)
    for i in range(n_photons):
        a = members[photons[i]]
        for j in range(i + 1, n_photons):
            b = members[photons[j]]
            m = mass(px[a] + px[b], py[a] + py[b], pz[a] + pz[b], E[a] + E[b])
            if abs(m - P[P_FSR_PI0_MASS]) < P[P_FSR_PI0_WINDOW]:
                paired[i] = True
                paired[j] = True
    p_pion = math.sqrt(px[lead] ** 2 + py[lead] ** 2 + pz[lead] ** 2)
    drop = np.zeros(n_members, np.bool_)
    for i in range(n_photons):
        if paired[i]:
            continue
        c = members[photons[i]]
        p_gamma = math.sqrt(px[c] ** 2 + py[c] ** 2 + pz[c] ** 2)
        frac = p_gamma / p_pion if p_pion > 0.0 else 0.0
        if frac < P[P_FSR_SOFT_FRAC]:
            drop[photons[i]] = True
        elif p_gamma > P[P_FSR_HARD_P_MIN]:
            m = mass(px[lead] + px[c], py[lead] + py[c], pz[lead] + pz[c], E[lead] + E[c])
            if m > P[P_FSR_MASS_MIN]:
                drop[photons[i]] = True
    n_kept = 0
    for k in range(n_members):
        if not drop[k]:
            members[n_kept] = members[k]
            n_kept += 1
    return n_kept


@numba.njit(cache=True)
def build_tau(seed, e0, e1, pdg, q, px, py, pz, E, p, P, members, out):
    """buildTauFromPion: fills out[T_*] for the tau seeded by the charged hadron 'seed'."""
    tpx, tpy, tpz, tE = px[seed], py[seed], pz[seed], E[seed]
    charge = q[seed]
    n_pions, n_photons, n_neutrons = 1, 0, 0
    members[0] = seed
    n_members = 1
    running = P[P_RUNNING_AXIS] > 0.5
    for c in range(e0, e1):
        if c == seed:
            continue
        if running:
            dR = dr_angle(px[c], py[c], pz[c], tpx, tpy, tpz)
        else:
            dR = dr_angle(px[c], py[c], pz[c], px[seed], py[seed], pz[seed])
        if dR > P[P_DR_MAX]:
            continue
        if p[c] < P[P_MIN_P]:
            continue
        a = abs(pdg[c])
        if a == 11 or a == 13:
            continue
        elif a == PDG_NEUTRAL_HADRON and p[c] > P[P_MIN_NH_P]:
            n_neutrons += 1
        elif a == 211 and p[c] > P[P_MIN_PION_P]:
            n_pions += 1
        elif a == 22 and p[c] > P[P_MIN_PHOTON_P]:
            n_photons += 1
        else:
            continue
        charge += q[c]
        tpx += px[c]
        tpy += py[c]
        tpz += pz[c]
        tE += E[c]
        members[n_members] = c
        n_members += 1

    out[:] = 0.0
    if abs(charge) != 1.0:
        out[T_ID] = -1
        return
    tau_id = assign_tau_id(n_pions, n_photons, n_neutrons)
    if P[P_FSR] > 0.5 and n_pions == 1 and n_neutrons == 0 and n_photons >= 1:
        if P[P_FSR_MAX_PHOTONS] <= 0 or n_photons <= P[P_FSR_MAX_PHOTONS]:
            n_kept = remove_fsr_photons(members, n_members, pdg, px, py, pz, E, P)
            if n_kept < n_members:
                n_members = n_kept
                tpx, tpy, tpz, tE = 0.0, 0.0, 0.0, 0.0
                n_photons = 0
                for k in range(n_members):
                    c = members[k]
                    tpx += px[c]
                    tpy += py[c]
                    tpz += pz[c]
                    tE += E[c]
                    if abs(pdg[c]) == 22:
                        n_photons += 1
                tau_id = assign_tau_id(n_pions, n_photons, n_neutrons)
    out[T_PX], out[T_PY], out[T_PZ], out[T_E] = tpx, tpy, tpz, tE
    out[T_CHARGE] = charge
    out[T_ID] = tau_id
    out[T_NUM_CHARGED] = n_pions
    out[T_NUM_PHOTONS] = n_photons


@numba.njit(cache=True)
def find_all_taus(e0, e1, pdg, q, px, py, pz, E, p, P):
    """findAllTaus: the taus of one event, after removing duplicates."""
    n = e1 - e0
    taus = np.zeros((n, N_TAU_VARS))
    members = np.empty(n, np.int64)
    out = np.zeros(N_TAU_VARS)
    n_taus = 0
    for seed in range(e0, e1):
        if abs(pdg[seed]) != 211 or p[seed] < P[P_MIN_PION_P] or p[seed] < P[P_MIN_P]:
            continue
        build_tau(seed, e0, e1, pdg, q, px, py, pz, E, p, P, members, out)
        duplicate = False
        for i in range(n_taus):
            if dr_angle(out[T_PX], out[T_PY], out[T_PZ], taus[i, T_PX], taus[i, T_PY], taus[i, T_PZ]) < P[P_DUPLICATE_DR]:
                duplicate = True
                break
        if not duplicate:
            taus[n_taus] = out
            n_taus += 1
    return taus[:n_taus]


@numba.njit(parallel=True, cache=True)
def run_cone(jets, e_off, pdg, q, px, py, pz, E, p, P):
    n_jets = len(jets)
    found = np.zeros(n_jets, np.bool_)
    result = np.zeros((n_jets, N_TAU_VARS))
    result[:, T_ID] = -1
    dm = np.full(n_jets, -1, np.int64)
    matched = np.zeros(n_jets, np.bool_)
    for j in numba.prange(n_jets):
        taus = find_all_taus(e_off[j], e_off[j + 1], pdg, q, px, py, pz, E, p, P)
        best, best_angle = -1, P[P_MATCHING_CONE]
        for i in range(len(taus)):
            if taus[i, T_PX] == 0.0 and taus[i, T_PY] == 0.0 and taus[i, T_PZ] == 0.0:
                continue  # rejected tau (|q| != 1), no direction
            theta, phi = theta_phi(taus[i, T_PX], taus[i, T_PY], taus[i, T_PZ])
            angle = hps.angle3d(theta, phi, jets[j, 2], jets[j, 3])
            if angle < best_angle:
                best, best_angle = i, angle
        if best < 0:
            continue
        matched[j] = True
        result[j] = taus[best]
        dm[j] = tau_id_to_dm(int(taus[best, T_ID]))
        found[j] = dm[j] >= 0
    return found, matched, dm, result


def prepare_inputs(data):
    jet_p4 = g.reinitialize_p4(data["reco_jet_p4"])
    jets = np.ascontiguousarray(
        np.stack([ak.to_numpy(x) for x in [jet_p4.pt, jet_p4.p, jet_p4.theta, jet_p4.phi]], axis=1),
        dtype=np.float64,
    )
    e_off, e = hps._flatten_cands(data["event_reco_cand_p4s"], data["event_reco_cand_pdgs"], data["event_reco_cand_charges"])
    return jets, e_off, e


def run(jets, e_off, e, P):
    return run_cone(jets, e_off, e["pdg"], e["q"], e["px"], e["py"], e["pz"], e["E"], e["p"], P)


class ConeTauBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.P = build_params(cfg.builder)

    def process_jets(self, data):
        """One tau per jet. Jets without a matched tau, or whose tau has no HPS decay mode, get decay mode -1 and a
        null four-momentum."""
        found, matched, dm, taus = run(*prepare_inputs(data), self.P)

        def per_tau(i, default=0.0):
            return np.where(found, taus[:, i], default)

        return {
            "tau_p4s": hps._p4_record(*[per_tau(i) for i in [T_PX, T_PY, T_PZ, T_E]]),
            "tau_decaymode": ak.Array(dm),
            "tau_charge": ak.Array(per_tau(T_CHARGE)),
            "tau_nCharged": ak.Array(per_tau(T_NUM_CHARGED).astype(np.int64)),
            "tau_nGammas": ak.Array(per_tau(T_NUM_PHOTONS).astype(np.int64)),
            # ID of the matched tau as in TausFCCee (also for the negative IDs), -1000 if no tau was matched
            "tau_coneID": ak.Array(np.where(matched, taus[:, T_ID], -1000).astype(np.int64)),
        }
