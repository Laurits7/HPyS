"""Numba implementation of the hadrons-plus-strips (HPS) tau reconstruction algorithm.

The jets and particle-flow candidates are flattened into numpy arrays and processed jet-by-jet in parallel by a
compiled kernel. The algorithm is identical to the object-oriented reference implementation in hpys.reference.
"""

import json
import math

import awkward as ak
import numba
import numpy as np
import vector
from omegaconf import OmegaConf

from hpys import kinematics as g

m_pi0 = 0.135
# CV: relative tolerance below which two values of a ranking criterion are considered equal, so that candidates that differ
#     only by floating-point rounding (e.g. the same particles summed in a different order) are ranked by the next criterion
RANKING_TOLERANCE = 1.0e-9
# CV: distance below which two candidates of the same type and charge are considered to be the same particle
dRmatch = 1.0e-3

KINEMATIC_VARIABLES = {"pt": 0, "p": 1}
RANKING_CRITERIA = {"numChargedCands": 0, "pt": 1, "p": 2, "numStrips": 3, "isolation": 4}
DECAY_MODE_NAMES = {0: "1Prong0Pi0", 1: "1Prong1Pi0", 2: "1Prong2Pi0", 10: "3Prong0Pi0", 11: "3Prong1Pi0"}

# indices of the scalar parameters
(
    P_KIN,
    P_SIG_MIN_CH,
    P_SIG_MIN_E,
    P_SIG_MIN_MU,
    P_SIG_MIN_LEAD,
    P_ISO_MIN_CH,
    P_ISO_MIN_E,
    P_ISO_MIN_GAMMA,
    P_ISO_MIN_MU,
    P_ISO_MIN_NH,
    P_MATCHING_CONE,
    P_SIGNAL_CONE_SCALE,
    P_SIGNAL_CONE_MIN,
    P_SIGNAL_CONE_MAX,
    P_ISO_CONE,
    P_MAX_OUTER_FRAC,
    P_USE_GAMMAS,
    P_MIN_GAMMA_SEED,
    P_MIN_GAMMA_ADD,
    P_USE_ELECTRONS,
    P_MIN_ELECTRON_SEED,
    P_MIN_ELECTRON_ADD,
    P_MIN_STRIP_PT,
    P_UPDATE_EACH_CAND,
    P_MAX_ITERATIONS,
    P_DYNAMIC,
    P_DYN_THETA_A,
    P_DYN_THETA_B,
    P_DYN_THETA_MIN,
    P_DYN_THETA_MAX,
    P_DYN_PHI_A,
    P_DYN_PHI_B,
    P_DYN_PHI_MIN,
    P_DYN_PHI_MAX,
    P_STATIC_THETA,
    P_STATIC_PHI,
    N_PARAMS,
) = range(37)

# indices of the per-decay-mode parameters
(
    D_IDX,
    D_NUM_CHARGED,
    D_NUM_STRIPS,
    D_MIN_MASS,
    D_MAX_MASS,
    D_HAS_SCALING,
    D_SCALING_REF,
    D_SCALING_MAX,
    D_STRIP_MASS_CORR,
    D_MIN_STRIP_MASS,
    D_MAX_STRIP_MASS,
    D_MAX_CHARGED,
    D_MAX_STRIPS,
    N_DM_PARAMS,
) = range(14)

# indices of the per-tau scalar outputs
(
    T_PX,
    T_PY,
    T_PZ,
    T_E,
    T_CHARGE,
    T_LEAD_PT,
    T_SIGNAL_CONE,
    T_CH_ISO,
    T_GAMMA_ISO,
    T_NH_ISO,
    T_OUTER_SUM,
    T_PASSES_OUTER,
    T_STRIP_MASS_CORR,
    T_NUM_CHARGED,
    T_NUM_STRIPS,
    T_DM,
    N_TAU_VARS,
) = range(17)

# indices of the per-tau derived features
(
    F_CH_ISO_0P3,
    F_GAMMA_ISO_0P3,
    F_NH_ISO_0P3,
    F_N_GAMMAS,
    F_EM_FRAC,
    F_DTHETA_STRIP,
    F_DPHI_STRIP,
    F_DR_SIGNAL,
    F_DR_ISO,
    N_FEATURES,
) = range(10)


def build_params(cfg):
    """Converts the 'builder' configuration into the numeric arrays used by the numba kernel."""
    kinematicVariable = cfg.get("kinematicVariable", "pt")
    if kinematicVariable not in KINEMATIC_VARIABLES:
        raise ValueError("Invalid kinematicVariable = '%s' !!" % kinematicVariable)
    strip = cfg.StripAlgo
    if strip.stripSize not in ["dynamic", "static"]:
        raise ValueError("Invalid stripSize = '%s' !!" % strip.stripSize)
    maxStripOuterFraction = cfg.isolation.maxStripOuterFraction

    P = np.zeros(N_PARAMS, dtype=np.float64)
    P[P_KIN] = KINEMATIC_VARIABLES[kinematicVariable]
    P[P_SIG_MIN_CH] = cfg.signalCands.minChargedHadronPt
    P[P_SIG_MIN_E] = cfg.signalCands.minElectronPt
    P[P_SIG_MIN_MU] = cfg.signalCands.minMuonPt
    P[P_SIG_MIN_LEAD] = cfg.signalCands.minLeadChargedCandPt
    P[P_ISO_MIN_CH] = cfg.isolationCands.minChargedHadronPt
    P[P_ISO_MIN_E] = cfg.isolationCands.minElectronPt
    P[P_ISO_MIN_GAMMA] = cfg.isolationCands.minGammaPt
    P[P_ISO_MIN_MU] = cfg.isolationCands.minMuonPt
    P[P_ISO_MIN_NH] = cfg.isolationCands.minNeutralHadronPt
    P[P_MATCHING_CONE] = cfg.matchingConeSize
    P[P_SIGNAL_CONE_SCALE] = cfg.signalCone.scale
    P[P_SIGNAL_CONE_MIN] = cfg.signalCone["min"]
    P[P_SIGNAL_CONE_MAX] = cfg.signalCone["max"]
    P[P_ISO_CONE] = cfg.isolationConeSize
    P[P_MAX_OUTER_FRAC] = np.nan if maxStripOuterFraction is None else maxStripOuterFraction
    P[P_USE_GAMMAS] = strip.useGammas
    P[P_MIN_GAMMA_SEED] = strip.minGammaPtSeed
    P[P_MIN_GAMMA_ADD] = strip.minGammaPtAdd
    P[P_USE_ELECTRONS] = strip.useElectrons
    P[P_MIN_ELECTRON_SEED] = strip.minElectronPtSeed
    P[P_MIN_ELECTRON_ADD] = strip.minElectronPtAdd
    P[P_MIN_STRIP_PT] = strip.minStripPt
    P[P_UPDATE_EACH_CAND] = strip.updateStripAfterEachCand
    P[P_MAX_ITERATIONS] = strip.maxStripBuildIterations
    P[P_DYNAMIC] = strip.stripSize == "dynamic"
    for axis, (a, b, lo, hi) in {
        "theta": (P_DYN_THETA_A, P_DYN_THETA_B, P_DYN_THETA_MIN, P_DYN_THETA_MAX),
        "phi": (P_DYN_PHI_A, P_DYN_PHI_B, P_DYN_PHI_MIN, P_DYN_PHI_MAX),
    }.items():
        params = strip.dynamic[axis]
        P[a], P[b], P[lo], P[hi] = params["a"], params["b"], params["min"], params["max"]
    P[P_STATIC_THETA] = strip.static.theta
    P[P_STATIC_PHI] = strip.static.phi

    DM = np.zeros((len(cfg.decayModes), N_DM_PARAMS), dtype=np.float64)
    for i, cfgDecayMode in enumerate(cfg.decayModes.values()):
        DM[i, D_IDX] = cfgDecayMode["dm_idx"]
        DM[i, D_NUM_CHARGED] = cfgDecayMode["numChargedCands"]
        DM[i, D_NUM_STRIPS] = cfgDecayMode["numStrips"]
        DM[i, D_MIN_MASS] = cfgDecayMode["minTauMass"]
        DM[i, D_MAX_MASS] = cfgDecayMode["maxTauMass"]
        scaling = cfgDecayMode.get("maxTauMassScaling", None)
        if scaling is not None:
            DM[i, D_HAS_SCALING] = 1
            DM[i, D_SCALING_REF] = scaling["refValue"]
            DM[i, D_SCALING_MAX] = scaling["maxLimit"]
        DM[i, D_STRIP_MASS_CORR] = cfgDecayMode.get("stripMassCorrection", False)
        DM[i, D_MIN_STRIP_MASS] = cfgDecayMode.get("minStripMass", -np.inf)
        DM[i, D_MAX_STRIP_MASS] = cfgDecayMode.get("maxStripMass", np.inf)
        DM[i, D_MAX_CHARGED] = cfgDecayMode["maxChargedCands"]
        DM[i, D_MAX_STRIPS] = cfgDecayMode["maxStrips"]

    for criterion in cfg.candidateRanking:
        if criterion not in RANKING_CRITERIA:
            raise ValueError("Invalid candidateRanking criterion = '%s' !!" % criterion)
    RANK = np.array([RANKING_CRITERIA[c] for c in cfg.candidateRanking], dtype=np.int64)
    return P, DM, RANK


# ---------------------------------------------------------------------------------------------------------------------
# Kinematic helpers
# ---------------------------------------------------------------------------------------------------------------------


@numba.njit(cache=True)
def angle3d(theta1, phi1, theta2, phi2):
    cos_angle = math.sin(theta1) * math.sin(theta2) * math.cos(phi1 - phi2) + math.cos(theta1) * math.cos(theta2)
    return math.acos(min(max(cos_angle, -1.0), 1.0))


@numba.njit(cache=True)
def deltaPhi(phi1, phi2):
    diff = phi1 - phi2
    return abs(math.atan2(math.sin(diff), math.cos(diff)))


@numba.njit(cache=True)
def p4_props(px, py, pz, E):
    """Returns (pt, p, theta, phi, mass) of a four-vector, with the conventions of the vector package."""
    pt = math.sqrt(px * px + py * py)
    p = math.sqrt(px * px + py * py + pz * pz)
    theta = math.atan2(pt, pz)
    phi = math.atan2(py, px)
    m2 = E * E - p * p
    mass = math.copysign(math.sqrt(abs(m2)), m2)
    return pt, p, theta, phi, mass


@numba.njit(cache=True)
def kin(pt, p, kinVar):
    return pt if kinVar == 0 else p


@numba.njit(cache=True)
def stripSizeFunction(x, a, b, max_value):
    if x <= 0.0:
        return max_value
    return a * x ** (-b)


@numba.njit(cache=True)
def stable_argsort_desc(values):
    return np.argsort(-values, kind="mergesort")


@numba.njit(cache=True)
def next_combination(combo, k, n):
    """Advances combo (k indices out of n) to the next combination in lexicographic order; returns False at the end."""
    i = k - 1
    while i >= 0 and combo[i] == n - k + i:
        i -= 1
    if i < 0:
        return False
    combo[i] += 1
    for j in range(i + 1, k):
        combo[j] = combo[j - 1] + 1
    return True


# ---------------------------------------------------------------------------------------------------------------------
# Strips
# ---------------------------------------------------------------------------------------------------------------------


@numba.njit(cache=True)
def sum_members(members, flags, px, py, pz, E):
    sx = 0.0
    sy = 0.0
    sz = 0.0
    sE = 0.0
    for c in members:
        if flags[c]:
            sx += px[c]
            sy += py[c]
            sz += pz[c]
            sE += E[c]
    return sx, sy, sz, sE


@numba.njit(cache=True)
def add_cands_to_strip(cands, used, current, strip, pt, p, theta, phi, px, py, pz, E, P):
    """Adds the candidates within the strip window to the strip; strip = [px, py, pz, E] is updated in place."""
    isCandAdded = False
    kinVar = int(P[P_KIN])
    for c in cands:
        if used[c] or current[c]:
            continue
        s_pt, s_p, s_theta, s_phi, _ = p4_props(strip[0], strip[1], strip[2], strip[3])
        if P[P_DYNAMIC] > 0.5:
            x_cand = kin(pt[c], p[c], kinVar)
            x_strip = kin(s_pt, s_p, kinVar)
            sizeTheta = stripSizeFunction(x_cand, P[P_DYN_THETA_A], P[P_DYN_THETA_B], P[P_DYN_THETA_MAX])
            sizeTheta += stripSizeFunction(x_strip, P[P_DYN_THETA_A], P[P_DYN_THETA_B], P[P_DYN_THETA_MAX])
            sizeTheta = min(max(sizeTheta, P[P_DYN_THETA_MIN]), P[P_DYN_THETA_MAX])
            sizePhi = stripSizeFunction(x_cand, P[P_DYN_PHI_A], P[P_DYN_PHI_B], P[P_DYN_PHI_MAX])
            sizePhi += stripSizeFunction(x_strip, P[P_DYN_PHI_A], P[P_DYN_PHI_B], P[P_DYN_PHI_MAX])
            sizePhi = min(max(sizePhi, P[P_DYN_PHI_MIN]), P[P_DYN_PHI_MAX])
        else:
            sizeTheta = P[P_STATIC_THETA]
            sizePhi = P[P_STATIC_PHI]
        if abs(theta[c] - s_theta) < sizeTheta and deltaPhi(phi[c], s_phi) < sizePhi:
            current[c] = True
            isCandAdded = True
            if P[P_UPDATE_EACH_CAND] > 0.5:
                strip[0], strip[1], strip[2], strip[3] = sum_members(np.arange(len(px)), current, px, py, pz, E)
    return isCandAdded


@numba.njit(cache=True)
def build_strips(abs_pdg, pt, p, theta, phi, px, py, pz, E, P):
    """Clusters e/gamma candidates into strips. Returns the strip index of each candidate (-1 if in no strip)
    and the four-vectors (visible mass) of the strips, in order of decreasing pT."""
    n = len(pt)
    seeds = []
    adds = []
    for c in range(n):
        isGamma = abs_pdg[c] == 22 and P[P_USE_GAMMAS] > 0.5
        isElectron = abs_pdg[c] == 11 and P[P_USE_ELECTRONS] > 0.5
        if isGamma or isElectron:
            minPtSeed = P[P_MIN_GAMMA_SEED] if abs_pdg[c] == 22 else P[P_MIN_ELECTRON_SEED]
            minPtAdd = P[P_MIN_GAMMA_ADD] if abs_pdg[c] == 22 else P[P_MIN_ELECTRON_ADD]
            if pt[c] > minPtSeed:
                seeds.append(c)
            elif pt[c] > minPtAdd:
                adds.append(c)
    seedCands = np.array(seeds, dtype=np.int64)
    addCands = np.array(adds, dtype=np.int64)

    stripOf = np.full(n, -1, dtype=np.int64)
    strip_p4s = np.zeros((n, 4), dtype=np.float64)
    numStrips = 0
    used = np.zeros(n, dtype=np.bool_)
    current = np.zeros(n, dtype=np.bool_)
    all_cands = np.arange(n)
    maxIterations = P[P_MAX_ITERATIONS]
    for seed in seedCands:
        if used[seed]:
            continue
        current[:] = False
        current[seed] = True
        # CV: the strip is initialized with the pi0 mass and set to the visible mass of its constituents once updated
        strip = np.array(
            [px[seed], py[seed], pz[seed], math.sqrt(px[seed] ** 2 + py[seed] ** 2 + pz[seed] ** 2 + m_pi0**2)]
        )
        iteration = 0
        while iteration < maxIterations or maxIterations == -1:
            isCandAdded = add_cands_to_strip(seedCands, used, current, strip, pt, p, theta, phi, px, py, pz, E, P)
            isCandAdded |= add_cands_to_strip(addCands, used, current, strip, pt, p, theta, phi, px, py, pz, E, P)
            if P[P_UPDATE_EACH_CAND] < 0.5:
                strip[0], strip[1], strip[2], strip[3] = sum_members(all_cands, current, px, py, pz, E)
            if not isCandAdded:
                break
            iteration += 1
        s_pt = math.sqrt(strip[0] ** 2 + strip[1] ** 2)
        if s_pt > P[P_MIN_STRIP_PT]:
            for c in range(n):
                if current[c]:
                    used[c] = True
                    stripOf[c] = numStrips
            strip_p4s[numStrips] = strip
            numStrips += 1

    # CV: sort strips in order of decreasing pT
    strip_pts = np.sqrt(strip_p4s[:numStrips, 0] ** 2 + strip_p4s[:numStrips, 1] ** 2)
    order = stable_argsort_desc(strip_pts)
    rank = np.empty(numStrips, dtype=np.int64)
    for i in range(numStrips):
        rank[order[i]] = i
    for c in range(n):
        if stripOf[c] >= 0:
            stripOf[c] = rank[stripOf[c]]
    return stripOf, strip_p4s[:numStrips][order]


# ---------------------------------------------------------------------------------------------------------------------
# Tau candidates
# ---------------------------------------------------------------------------------------------------------------------


@numba.njit(cache=True)
def is_same_particle(c1, pdg1, q1, theta1, phi1, c2, pdg2, q2, theta2, phi2, same_collection):
    if same_collection and c1 == c2:
        return True
    return pdg1 == pdg2 and q1 == q2 and angle3d(theta1, phi1, theta2, phi2) < dRmatch


@numba.njit(cache=True)
def isolation_class(abs_pdg):
    """0 = charged (e, mu, h), 1 = photon, 2 = neutral hadron."""
    if abs_pdg == 11 or abs_pdg == 13 or abs_pdg == 211:
        return 0
    if abs_pdg == 22:
        return 1
    if abs_pdg == 130:
        return 2
    return -1


@numba.njit(cache=True)
def select_isolation(abs_pdg, pt, P):
    cls = isolation_class(abs_pdg)
    if abs_pdg == 11:
        return pt > P[P_ISO_MIN_E]
    if abs_pdg == 13:
        return pt > P[P_ISO_MIN_MU]
    if abs_pdg == 211:
        return pt > P[P_ISO_MIN_CH]
    if cls == 1:
        return pt > P[P_ISO_MIN_GAMMA]
    if cls == 2:
        return pt > P[P_ISO_MIN_NH]
    return False


@numba.njit(cache=True)
def is_better(tau, best, RANK):
    """Lexicographic comparison of the ranking criteria, equal within RANKING_TOLERANCE; ties keep the earlier candidate."""
    for criterion in RANK:
        if criterion == 0:
            a, b = tau[T_NUM_CHARGED], best[T_NUM_CHARGED]
        elif criterion == 1:
            a = math.sqrt(tau[T_PX] ** 2 + tau[T_PY] ** 2)
            b = math.sqrt(best[T_PX] ** 2 + best[T_PY] ** 2)
        elif criterion == 2:
            a = math.sqrt(tau[T_PX] ** 2 + tau[T_PY] ** 2 + tau[T_PZ] ** 2)
            b = math.sqrt(best[T_PX] ** 2 + best[T_PY] ** 2 + best[T_PZ] ** 2)
        elif criterion == 3:
            a, b = tau[T_NUM_STRIPS], best[T_NUM_STRIPS]
        else:
            a = -(tau[T_CH_ISO] + tau[T_GAMMA_ISO])
            b = -(best[T_CH_ISO] + best[T_GAMMA_ISO])
        if abs(a - b) <= RANKING_TOLERANCE * max(abs(a), abs(b), 1.0):
            continue
        return a > b
    return False


@numba.njit(cache=True)
def eval_candidate(
    iMode,
    chargedCands,
    stripCombo,
    selectedStrips,
    stripOf,
    strip_p4s,
    jet,
    c_pdg,
    c_abs_pdg,
    c_q,
    c_pt,
    c_p,
    c_theta,
    c_phi,
    c_px,
    c_py,
    c_pz,
    c_E,
    jetIsoCands,
    e_pdg,
    e_abs_pdg,
    e_q,
    e_pt,
    e_p,
    e_theta,
    e_phi,
    evtIsoCands,
    P,
    DM,
    tau,
    sigCands,
    stripP4s,
    isoJet,
    isoEvt,
):
    """Builds the tau candidate for the given charged candidates and strip combination and applies the preselection.
    The outputs are written to tau, sigCands, stripP4s, isoJet and isoEvt.
    Returns (passes, number of signal candidates, number of jet isolation candidates, number of event isolation candidates)."""
    kinVar = int(P[P_KIN])
    numCharged = len(chargedCands)
    numStripsRequired = int(DM[iMode, D_NUM_STRIPS])
    n = len(c_pt)

    # clean strips of the charged candidates; strips are assigned the pi0 mass
    numSig = 0
    for c in chargedCands:
        sigCands[numSig] = c
        numSig += 1
    numCleanedStrips = 0
    cleaned = np.zeros((len(stripCombo), 4), dtype=np.float64)
    cleanedMembers = np.full(n, -1, dtype=np.int64)  # position in stripCombo of the strip a member belongs to
    keptIndex = np.full(len(stripCombo), -1, dtype=np.int64)  # index in 'cleaned' of each strip in stripCombo
    for i_s in range(len(stripCombo)):
        s = selectedStrips[stripCombo[i_s]]
        sx = 0.0
        sy = 0.0
        sz = 0.0
        count = 0
        for m in range(n):
            if stripOf[m] != s:
                continue
            overlap = False
            for c in chargedCands:
                if is_same_particle(
                    m, c_pdg[m], c_q[m], c_theta[m], c_phi[m], c, c_pdg[c], c_q[c], c_theta[c], c_phi[c], True
                ):
                    overlap = True
                    break
            if not overlap:
                sx += c_px[m]
                sy += c_py[m]
                sz += c_pz[m]
                count += 1
                cleanedMembers[m] = i_s
        if count > 0 and math.sqrt(sx * sx + sy * sy) > P[P_MIN_STRIP_PT]:
            cleaned[numCleanedStrips] = (sx, sy, sz, math.sqrt(sx * sx + sy * sy + sz * sz + m_pi0 * m_pi0))
            keptIndex[i_s] = numCleanedStrips
            numCleanedStrips += 1
    if numCleanedStrips < numStripsRequired:
        return False, 0, 0, 0
    cleaned = cleaned[:numCleanedStrips]
    order = stable_argsort_desc(np.sqrt(cleaned[:, 0] ** 2 + cleaned[:, 1] ** 2))
    for i in range(numCleanedStrips):
        stripP4s[i] = cleaned[order[i]]
        for m in range(n):
            if cleanedMembers[m] >= 0 and keptIndex[cleanedMembers[m]] == order[i]:
                sigCands[numSig] = m
                numSig += 1

    # tau four-vector
    tx = 0.0
    ty = 0.0
    tz = 0.0
    tE = 0.0
    q = 0.0
    leadPt = 0.0
    for c in chargedCands:
        tx += c_px[c]
        ty += c_py[c]
        tz += c_pz[c]
        tE += c_E[c]
        q += 1.0 if c_q[c] > 0.0 else -1.0
        leadPt = max(leadPt, c_pt[c])
    for i in range(numCleanedStrips):
        tx += stripP4s[i, 0]
        ty += stripP4s[i, 1]
        tz += stripP4s[i, 2]
        tE += stripP4s[i, 3]
    t_pt, t_p, t_theta, t_phi, t_mass = p4_props(tx, ty, tz, tE)
    t_kin = kin(t_pt, t_p, kinVar)

    if t_kin > 0.0:
        signalCone = min(max(P[P_SIGNAL_CONE_SCALE] / t_kin, P[P_SIGNAL_CONE_MIN]), P[P_SIGNAL_CONE_MAX])
    else:
        signalCone = P[P_SIGNAL_CONE_MAX]

    # preselection
    if abs(round(q)) != 1 or not leadPt > P[P_SIG_MIN_LEAD]:
        return False, 0, 0, 0
    if not angle3d(t_theta, t_phi, jet[2], jet[3]) < P[P_MATCHING_CONE]:
        return False, 0, 0, 0
    for c in chargedCands:
        if angle3d(t_theta, t_phi, c_theta[c], c_phi[c]) > signalCone:
            return False, 0, 0, 0
    for i in range(numCleanedStrips):
        _, _, s_theta, s_phi, _ = p4_props(stripP4s[i, 0], stripP4s[i, 1], stripP4s[i, 2], stripP4s[i, 3])
        if angle3d(t_theta, t_phi, s_theta, s_phi) > signalCone:
            return False, 0, 0, 0

    # mass window
    minMass = DM[iMode, D_MIN_MASS]
    maxMass = DM[iMode, D_MAX_MASS]
    if DM[iMode, D_HAS_SCALING] > 0.5:
        maxMass = min(max(maxMass * math.sqrt(t_kin / DM[iMode, D_SCALING_REF]), maxMass), DM[iMode, D_SCALING_MAX])
    stripMassCorr = 0.0
    if DM[iMode, D_STRIP_MASS_CORR] > 0.5 and t_mass > 0.0:
        dm2 = 0.0
        for i in range(numCleanedStrips):
            s_pt, s_p, s_theta, s_phi, _ = p4_props(stripP4s[i, 0], stripP4s[i, 1], stripP4s[i, 2], stripP4s[i, 3])
            if P[P_DYNAMIC] > 0.5:
                x = kin(s_pt, s_p, kinVar)
                dTheta = stripSizeFunction(x, P[P_DYN_THETA_A], P[P_DYN_THETA_B], P[P_DYN_THETA_MAX])
                dPhi = stripSizeFunction(x, P[P_DYN_PHI_A], P[P_DYN_PHI_B], P[P_DYN_PHI_MAX])
            else:
                dTheta = P[P_STATIC_THETA]
                dPhi = P[P_STATIC_PHI]
            # dm/dX = -(P_tau . dp_strip/dX) / m_tau, varying the strip direction at fixed strip momentum
            dp_dTheta_x = s_p * math.cos(s_theta) * math.cos(s_phi)
            dp_dTheta_y = s_p * math.cos(s_theta) * math.sin(s_phi)
            dp_dTheta_z = -s_p * math.sin(s_theta)
            dm_dTheta = -(tx * dp_dTheta_x + ty * dp_dTheta_y + tz * dp_dTheta_z) / t_mass
            dm_dPhi = -(tx * (-stripP4s[i, 1]) + ty * stripP4s[i, 0]) / t_mass
            dm2 += (dm_dTheta * dTheta) ** 2 + (dm_dPhi * dPhi) ** 2
        stripMassCorr = math.sqrt(dm2)
    if not (minMass - stripMassCorr < t_mass < maxMass + stripMassCorr):
        return False, 0, 0, 0

    # isolation
    isoSums = np.zeros(3, dtype=np.float64)
    numIsoJet = 0
    for c in jetIsoCands:
        if not angle3d(c_theta[c], c_phi[c], t_theta, t_phi) < P[P_ISO_CONE]:
            continue
        overlap = False
        for i in range(numSig):
            s = sigCands[i]
            if is_same_particle(
                c, c_pdg[c], c_q[c], c_theta[c], c_phi[c], s, c_pdg[s], c_q[s], c_theta[s], c_phi[s], True
            ):
                overlap = True
                break
        if overlap:
            continue
        isoJet[numIsoJet] = c
        numIsoJet += 1
        cls = isolation_class(c_abs_pdg[c])
        if cls >= 0:
            isoSums[cls] += kin(c_pt[c], c_p[c], kinVar)
    numIsoEvt = 0
    for c in evtIsoCands:
        if not angle3d(e_theta[c], e_phi[c], t_theta, t_phi) < P[P_ISO_CONE]:
            continue
        isoEvt[numIsoEvt] = c
        numIsoEvt += 1
        cls = isolation_class(e_abs_pdg[c])
        if cls >= 0:
            isoSums[cls] += kin(e_pt[c], e_p[c], kinVar)

    # e/gamma in strips outside the signal cone
    outerSum = 0.0
    for i in range(numCharged, numSig):
        m = sigCands[i]
        if angle3d(t_theta, t_phi, c_theta[m], c_phi[m]) > signalCone:
            outerSum += kin(c_pt[m], c_p[m], kinVar)
    maxOuterFrac = P[P_MAX_OUTER_FRAC]
    passesOuter = math.isnan(maxOuterFrac) or outerSum < maxOuterFrac * t_kin

    tau[T_PX] = tx
    tau[T_PY] = ty
    tau[T_PZ] = tz
    tau[T_E] = tE
    tau[T_CHARGE] = q
    tau[T_LEAD_PT] = leadPt
    tau[T_SIGNAL_CONE] = signalCone
    tau[T_CH_ISO] = isoSums[0]
    tau[T_GAMMA_ISO] = isoSums[1]
    tau[T_NH_ISO] = isoSums[2]
    tau[T_OUTER_SUM] = outerSum
    tau[T_PASSES_OUTER] = passesOuter
    tau[T_STRIP_MASS_CORR] = stripMassCorr
    tau[T_NUM_CHARGED] = numCharged
    tau[T_NUM_STRIPS] = numCleanedStrips
    tau[T_DM] = DM[iMode, D_IDX]
    return True, numSig, numIsoJet, numIsoEvt


@numba.njit(cache=True)
def build_tau(
    jet,
    c_pdg,
    c_abs_pdg,
    c_q,
    c_pt,
    c_p,
    c_theta,
    c_phi,
    c_px,
    c_py,
    c_pz,
    c_E,
    e_pdg,
    e_abs_pdg,
    e_q,
    e_pt,
    e_p,
    e_theta,
    e_phi,
    P,
    DM,
    RANK,
    best_tau,
    best_sig,
    best_strips,
    best_isoJet,
    best_isoEvt,
):
    """Reconstructs the tau of one jet. The jet constituents and event candidates must be sorted by decreasing pT.
    Returns (found, number of signal candidates, number of jet isolation candidates, number of event isolation candidates)."""
    n = len(c_pt)
    n_evt = len(e_pt)

    signalChargedCands = []
    for c in range(n):
        a = c_abs_pdg[c]
        if (
            (a == 11 and c_pt[c] > P[P_SIG_MIN_E])
            or (a == 13 and c_pt[c] > P[P_SIG_MIN_MU])
            or (a == 211 and c_pt[c] > P[P_SIG_MIN_CH])
        ):
            signalChargedCands.append(c)
    numSignalCharged = len(signalChargedCands)

    stripOf, strip_p4s = build_strips(c_abs_pdg, c_pt, c_p, c_theta, c_phi, c_px, c_py, c_pz, c_E, P)
    numStrips = len(strip_p4s)
    strip_masses = np.empty(numStrips, dtype=np.float64)
    for s in range(numStrips):
        strip_masses[s] = p4_props(strip_p4s[s, 0], strip_p4s[s, 1], strip_p4s[s, 2], strip_p4s[s, 3])[4]

    jetIsoList = []
    for c in range(n):
        if select_isolation(c_abs_pdg[c], c_pt[c], P):
            jetIsoList.append(c)
    jetIsoCands = np.array(jetIsoList, dtype=np.int64)

    # event-level isolation candidates near the jet that are not jet constituents
    evtIsoList = []
    maxDistance = P[P_ISO_CONE] + P[P_MATCHING_CONE]
    for e in range(n_evt):
        if not select_isolation(e_abs_pdg[e], e_pt[e], P):
            continue
        if not angle3d(e_theta[e], e_phi[e], jet[2], jet[3]) < maxDistance:
            continue
        overlap = False
        for c in range(n):
            if is_same_particle(
                e, e_pdg[e], e_q[e], e_theta[e], e_phi[e], c, c_pdg[c], c_q[c], c_theta[c], c_phi[c], False
            ):
                overlap = True
                break
        if not overlap:
            evtIsoList.append(e)
    evtIsoCands = np.array(evtIsoList, dtype=np.int64)

    tau = np.zeros(N_TAU_VARS, dtype=np.float64)
    sig = np.empty(n, dtype=np.int64)
    strips = np.empty((n, 4), dtype=np.float64)
    isoJet = np.empty(n, dtype=np.int64)
    isoEvt = np.empty(n_evt, dtype=np.int64)
    found = False
    best_counts = (0, 0, 0)

    for iMode in range(DM.shape[0]):
        numCharged = int(DM[iMode, D_NUM_CHARGED])
        numStripsRequired = int(DM[iMode, D_NUM_STRIPS])
        if numSignalCharged < numCharged:
            continue
        selected = []
        if numStripsRequired > 0:
            for s in range(numStrips):
                if DM[iMode, D_MIN_STRIP_MASS] < strip_masses[s] < DM[iMode, D_MAX_STRIP_MASS]:
                    selected.append(s)
        selectedStrips = np.array(selected, dtype=np.int64)
        if len(selectedStrips) < numStripsRequired:
            continue
        nCharged = min(numSignalCharged, int(DM[iMode, D_MAX_CHARGED]))
        nStrips = min(len(selectedStrips), int(DM[iMode, D_MAX_STRIPS]))
        if numCharged <= 0 or numCharged > nCharged:
            continue
        if numStripsRequired > nStrips:
            continue

        chargedCombo = np.arange(numCharged)
        hasChargedCombo = True
        while hasChargedCombo:
            chargedCands = np.empty(numCharged, dtype=np.int64)
            for i in range(numCharged):
                chargedCands[i] = signalChargedCands[chargedCombo[i]]
            stripCombo = np.arange(numStripsRequired)
            hasStripCombo = True
            while hasStripCombo:
                passes, numSig, numIsoJet, numIsoEvt = eval_candidate(
                    iMode, chargedCands, stripCombo, selectedStrips, stripOf, strip_p4s, jet,
                    c_pdg, c_abs_pdg, c_q, c_pt, c_p, c_theta, c_phi, c_px, c_py, c_pz, c_E, jetIsoCands,
                    e_pdg, e_abs_pdg, e_q, e_pt, e_p, e_theta, e_phi, evtIsoCands,
                    P, DM, tau, sig, strips, isoJet, isoEvt,
                )  # fmt: skip
                if passes and (not found or is_better(tau, best_tau, RANK)):
                    found = True
                    best_tau[:] = tau
                    best_sig[:numSig] = sig[:numSig]
                    numBestStrips = int(tau[T_NUM_STRIPS])
                    best_strips[:numBestStrips] = strips[:numBestStrips]
                    best_isoJet[:numIsoJet] = isoJet[:numIsoJet]
                    best_isoEvt[:numIsoEvt] = isoEvt[:numIsoEvt]
                    best_counts = (numSig, numIsoJet, numIsoEvt)
                hasStripCombo = numStripsRequired > 0 and next_combination(stripCombo, numStripsRequired, nStrips)
            hasChargedCombo = next_combination(chargedCombo, numCharged, nCharged)
    return found, best_counts[0], best_counts[1], best_counts[2]


@numba.njit(cache=True)
def comp_features(tau, sig, numSig, isoJet, numIsoJet, isoEvt, numIsoEvt, c, e, kinVar, features):
    """Derived per-tau features. c and e are (abs_pdg, pt, p, theta, phi) arrays of jet constituents and event candidates."""
    t_pt, t_p, t_theta, t_phi, _ = p4_props(tau[T_PX], tau[T_PY], tau[T_PZ], tau[T_E])
    # isolation sums in a cone of 0.3
    iso0p3 = np.zeros(3, dtype=np.float64)
    dR_iso_sum = 0.0
    pt_iso_sum = 0.0
    for k in range(numIsoJet + numIsoEvt):
        arr = c if k < numIsoJet else e
        idx = isoJet[k] if k < numIsoJet else isoEvt[k - numIsoJet]
        cls = isolation_class(int(arr[0][idx]))
        angle = angle3d(arr[3][idx], arr[4][idx], t_theta, t_phi)
        if cls >= 0 and angle < 0.3:
            iso0p3[cls] += kin(arr[1][idx], arr[2][idx], kinVar)
        if cls == 1:
            dR_iso_sum += arr[1][idx] * angle
            pt_iso_sum += arr[1][idx]
    # signal photons
    nGammas = 0
    gamma_pt_sum = 0.0
    dTheta_sum = 0.0
    dPhi_sum = 0.0
    dR_sum = 0.0
    for k in range(numSig):
        idx = sig[k]
        if c[0][idx] != 22:
            continue
        pt = c[1][idx]
        nGammas += 1
        gamma_pt_sum += pt
        dTheta_sum += pt * abs(t_theta - c[3][idx])
        dPhi_sum += pt * deltaPhi(t_phi, c[4][idx])
        dR_sum += pt * angle3d(t_theta, t_phi, c[3][idx], c[4][idx])
    features[F_CH_ISO_0P3] = iso0p3[0]
    features[F_GAMMA_ISO_0P3] = iso0p3[1]
    features[F_NH_ISO_0P3] = iso0p3[2]
    features[F_N_GAMMAS] = nGammas
    features[F_EM_FRAC] = gamma_pt_sum / t_pt if t_pt > 0.0 else 0.0
    features[F_DTHETA_STRIP] = dTheta_sum / gamma_pt_sum if gamma_pt_sum > 0.0 else 0.0
    features[F_DPHI_STRIP] = dPhi_sum / gamma_pt_sum if gamma_pt_sum > 0.0 else 0.0
    features[F_DR_SIGNAL] = dR_sum / gamma_pt_sum if gamma_pt_sum > 0.0 else 0.0
    features[F_DR_ISO] = dR_iso_sum / pt_iso_sum if pt_iso_sum > 0.0 else 0.0


@numba.njit(cache=True, parallel=True)
def run_hps(jets, c_off, c_pdg, c_q, c_pt, c_p, c_theta, c_phi, c_px, c_py, c_pz, c_E,
            e_off, e_pdg, e_q, e_pt, e_p, e_theta, e_phi, P, DM, RANK):  # fmt: skip
    """Runs HPS on all jets. jets = (n_jets, 4) array of (pt, p, theta, phi); c_* and e_* are the flattened jet constituents
    and event candidates with offsets c_off and e_off. Candidate indices in the outputs refer to the flattened arrays."""
    n_jets = len(c_off) - 1
    taus = np.zeros((n_jets, N_TAU_VARS), dtype=np.float64)
    features = np.zeros((n_jets, N_FEATURES), dtype=np.float64)
    found = np.zeros(n_jets, dtype=np.bool_)
    sig = np.full(len(c_pt), -1, dtype=np.int64)
    strips = np.zeros((len(c_pt), 4), dtype=np.float64)
    isoJet = np.full(len(c_pt), -1, dtype=np.int64)
    isoEvt = np.full(len(e_pt), -1, dtype=np.int64)
    counts = np.zeros((n_jets, 4), dtype=np.int64)  # signal candidates, strips, jet and event isolation candidates
    kinVar = int(P[P_KIN])
    for j in numba.prange(n_jets):
        c0, c1 = c_off[j], c_off[j + 1]
        e0, e1 = e_off[j], e_off[j + 1]
        # CV: sort candidates in order of decreasing pT
        c_order = stable_argsort_desc(c_pt[c0:c1]) + c0
        e_order = stable_argsort_desc(e_pt[e0:e1]) + e0
        c_abs = np.abs(c_pdg[c_order])
        e_abs = np.abs(e_pdg[e_order])
        n = c1 - c0
        n_evt = e1 - e0
        best_tau = np.zeros(N_TAU_VARS, dtype=np.float64)
        best_sig = np.empty(n, dtype=np.int64)
        best_strips = np.empty((n, 4), dtype=np.float64)
        best_isoJet = np.empty(n, dtype=np.int64)
        best_isoEvt = np.empty(n_evt, dtype=np.int64)
        ok, numSig, numIsoJet, numIsoEvt = build_tau(
            jets[j], c_pdg[c_order], c_abs, c_q[c_order], c_pt[c_order], c_p[c_order], c_theta[c_order], c_phi[c_order],
            c_px[c_order], c_py[c_order], c_pz[c_order], c_E[c_order],
            e_pdg[e_order], e_abs, e_q[e_order], e_pt[e_order], e_p[e_order], e_theta[e_order], e_phi[e_order],
            P, DM, RANK, best_tau, best_sig, best_strips, best_isoJet, best_isoEvt,
        )  # fmt: skip
        if not ok:
            continue
        found[j] = True
        taus[j] = best_tau
        numStrips = int(best_tau[T_NUM_STRIPS])
        counts[j, 0] = numSig
        counts[j, 1] = numStrips
        counts[j, 2] = numIsoJet
        counts[j, 3] = numIsoEvt
        c_props = (c_abs.astype(np.float64), c_pt[c_order], c_p[c_order], c_theta[c_order], c_phi[c_order])
        e_props = (e_abs.astype(np.float64), e_pt[e_order], e_p[e_order], e_theta[e_order], e_phi[e_order])
        comp_features(best_tau, best_sig, numSig, best_isoJet, numIsoJet, best_isoEvt, numIsoEvt, c_props, e_props,
                      kinVar, features[j])  # fmt: skip
        # translate to indices in the flattened arrays; each jet owns the slots [c0, c1) and [e0, e1)
        for k in range(numSig):
            sig[c0 + k] = c_order[best_sig[k]]
        strips[c0 : c0 + numStrips] = best_strips[:numStrips]
        for k in range(numIsoJet):
            isoJet[c0 + k] = c_order[best_isoJet[k]]
        for k in range(numIsoEvt):
            isoEvt[e0 + k] = e_order[best_isoEvt[k]]
    return found, taus, features, counts, sig, strips, isoJet, isoEvt


# ---------------------------------------------------------------------------------------------------------------------
# Python interface
# ---------------------------------------------------------------------------------------------------------------------


def _flatten_cands(p4s, pdgs, charges):
    p4 = g.reinitialize_p4(p4s)
    offsets = np.concatenate([[0], np.cumsum(ak.to_numpy(ak.num(pdgs)))]).astype(np.int64)

    def flat(x, dtype=np.float64):
        return np.ascontiguousarray(ak.to_numpy(ak.flatten(x)), dtype=dtype)

    cands = {
        "pdg": flat(pdgs, np.int64),
        "q": flat(charges),
        "pt": flat(p4.pt),
        "p": flat(p4.p),
        "theta": flat(p4.theta),
        "phi": flat(p4.phi),
        "px": flat(p4.px),
        "py": flat(p4.py),
        "pz": flat(p4.pz),
        "E": flat(p4.energy),
        "mass": flat(p4.mass),
    }
    return offsets, cands


def _p4_record(px, py, pz, E, counts=None):
    px, py, pz, E = (np.ascontiguousarray(x, dtype=np.float64) for x in (px, py, pz, E))
    m2 = E * E - (px * px + py * py + pz * pz)
    mass = np.copysign(np.sqrt(np.abs(m2)), m2)
    record = ak.zip({"px": px, "py": py, "pz": pz, "mass": mass})
    if counts is not None:
        record = ak.unflatten(record, counts)
    return vector.awk(record)


def _gather(values, indices, counts):
    return ak.unflatten(values[indices], counts)


class HPSInputs:
    """Flattened jets, jet constituents and event candidates, as used by the numba kernel."""

    def __init__(self, jets, c_off, c, c_extra, e_off, e, e_extra):
        self.jets = jets
        self.c_off = c_off
        self.c = c
        self.c_extra = c_extra
        self.e_off = e_off
        self.e = e
        self.e_extra = e_extra

    def __len__(self):
        return len(self.jets)


def prepare_inputs(data):
    """Flattens the jets and candidates of an awkward array into numpy arrays. The result can be reused for several
    runs of the algorithm, e.g. with different configurations."""
    jet_p4 = g.reinitialize_p4(data["reco_jet_p4"])
    jets = np.ascontiguousarray(
        np.stack([ak.to_numpy(x) for x in [jet_p4.pt, jet_p4.p, jet_p4.theta, jet_p4.phi]], axis=1),
        dtype=np.float64,
    )
    c_off, c = _flatten_cands(data["reco_cand_p4s"], data["reco_cand_pdgs"], data["reco_cand_charges"])
    e_off, e = _flatten_cands(data["event_reco_cand_p4s"], data["event_reco_cand_pdgs"], data["event_reco_cand_charges"])
    if len(c_off) != len(e_off):
        raise ValueError("Number of jets and number of events with candidates don't match !!")
    names = ["dxy", "dxy_error", "dz", "dz_error"]
    c_extra = {k: np.asarray(ak.flatten(data["reco_cand_%s" % k]), dtype=np.float64) for k in names}
    e_extra = {k: np.asarray(ak.flatten(data["event_reco_cand_%s" % k]), dtype=np.float64) for k in names}
    return HPSInputs(jets, c_off, c, c_extra, e_off, e, e_extra)


def run(inputs, P, DM, RANK):
    """Runs the numba kernel on prepared inputs with the numeric parameters from build_params."""
    c, e = inputs.c, inputs.e
    return run_hps(
        inputs.jets, inputs.c_off, c["pdg"], c["q"], c["pt"], c["p"], c["theta"], c["phi"], c["px"], c["py"], c["pz"],
        c["E"], inputs.e_off, e["pdg"], e["q"], e["pt"], e["p"], e["theta"], e["phi"], P, DM, RANK,
    )  # fmt: skip


class HPSTauBuilder:
    def __init__(self, cfg, verbosity=0):
        self.cfg = cfg
        self.verbosity = verbosity
        self.P, self.DM, self.RANK = build_params(cfg.builder)
        self.kinematicVariable = cfg.builder.get("kinematicVariable", "pt")

    def print_config(self):
        print(json.dumps(OmegaConf.to_container(self.cfg), indent=4))

    def process_jets(self, data):
        """Reconstructs one tau per jet. Jets without a tau candidate get a dummy tau with decay mode -1."""
        inputs = prepare_inputs(data)
        found, taus, features, counts, sig, strips, isoJet, isoEvt = run(inputs, self.P, self.DM, self.RANK)
        return self.write_taus(
            found, taus, features, counts, sig, strips, isoJet, isoEvt,
            inputs.c_off, inputs.c, inputs.c_extra, inputs.e_off, inputs.e, inputs.e_extra,
        )  # fmt: skip

    def write_taus(self, found, taus, features, counts, sig, strips, isoJet, isoEvt, c_off, c, c_extra, e_off, e, e_extra):
        n_jets = len(found)
        local_c = np.arange(c_off[-1]) - np.repeat(c_off[:-1], np.diff(c_off))
        local_e = np.arange(e_off[-1]) - np.repeat(e_off[:-1], np.diff(e_off))
        jet_of_c = np.repeat(np.arange(n_jets), np.diff(c_off))
        jet_of_e = np.repeat(np.arange(n_jets), np.diff(e_off))
        sig_idx = sig[local_c < counts[jet_of_c, 0]]
        strip_rows = strips[local_c < counts[jet_of_c, 1]]
        isoJet_idx = isoJet[local_c < counts[jet_of_c, 2]]
        isoEvt_idx = isoEvt[local_e < counts[jet_of_e, 3]]

        def cand_attrs(prefix, idx_jet, n_jet, idx_evt=None, n_evt=None):
            columns = {
                "pdgIds": (c["pdg"], e["pdg"]),
                "q": (c["q"], e["q"]),
                "d0": (c_extra["dxy"], e_extra["dxy"]),
                "d0err": (c_extra["dxy_error"], e_extra["dxy_error"]),
                "dz": (c_extra["dz"], e_extra["dz"]),
                "dzerr": (c_extra["dz_error"], e_extra["dz_error"]),
            }
            p4 = {k: (c[k], e[k]) for k in ["px", "py", "pz", "E"]}
            out = {}

            def collect(values):
                x = _gather(values[0], idx_jet, n_jet)
                if idx_evt is not None:
                    x = ak.concatenate([x, _gather(values[1], idx_evt, n_evt)], axis=1)
                return x

            parts = {k: ak.flatten(collect(v)) for k, v in p4.items()}
            num = n_jet if idx_evt is None else n_jet + n_evt
            out["%s_p4s" % prefix] = _p4_record(*[ak.to_numpy(parts[k]) for k in ["px", "py", "pz", "E"]], counts=num)
            for name, values in columns.items():
                out["%s_%s" % (prefix, name)] = collect(values)
            return out

        dm = np.where(found, taus[:, T_DM], -1).astype(np.int64)
        retVal = {"tau_p4s": _p4_record(taus[:, T_PX], taus[:, T_PY], taus[:, T_PZ], taus[:, T_E])}
        retVal.update(cand_attrs("tauSigCand", sig_idx, counts[:, 0]))
        retVal["tauStrip_p4s"] = _p4_record(*[strip_rows[:, i] for i in range(4)], counts=counts[:, 1])
        retVal.update(cand_attrs("tauIsoCand", isoJet_idx, counts[:, 2], isoEvt_idx, counts[:, 3]))

        def per_tau(values, default):
            return ak.Array(np.where(found, values, default))

        retVal.update(
            {
                "tauChargedIso_dR0p5": per_tau(taus[:, T_CH_ISO], -1.0),
                "tauGammaIso_dR0p5": per_tau(taus[:, T_GAMMA_ISO], -1.0),
                "tauNeutralHadronIso_dR0p5": per_tau(taus[:, T_NH_ISO], -1.0),
                "tauChargedIso_dR0p3": per_tau(features[:, F_CH_ISO_0P3], 0.0),
                "tauGammaIso_dR0p3": per_tau(features[:, F_GAMMA_ISO_0P3], 0.0),
                "tauNeutralHadronIso_dR0p3": per_tau(features[:, F_NH_ISO_0P3], 0.0),
                "tauStripOuterSum": per_tau(taus[:, T_OUTER_SUM], 0.0),
                "tau_passesStripOuterCut": ak.Array(found & (taus[:, T_PASSES_OUTER] > 0.5)),
                "tau_stripMassCorrection": per_tau(taus[:, T_STRIP_MASS_CORR], 0.0),
                "tau_signalConeSize": per_tau(taus[:, T_SIGNAL_CONE], -1.0),
                "tau_charge": per_tau(taus[:, T_CHARGE], 0.0),
                "tau_decaymode": ak.Array(dm),
                "tau_nGammas": ak.Array(np.where(found, features[:, F_N_GAMMAS], 0).astype(np.int64)),
                "tau_nCharged": ak.Array(np.where(found, taus[:, T_NUM_CHARGED], 0).astype(np.int64)),
                "tau_leadChargedCand_pt": per_tau(taus[:, T_LEAD_PT], 0.0),
                "tau_emEnergyFrac": per_tau(features[:, F_EM_FRAC], 0.0),
                "tau_dEta_strip": per_tau(features[:, F_DTHETA_STRIP], 0.0),
                "tau_dPhi_strip": per_tau(features[:, F_DPHI_STRIP], 0.0),
                "tau_dR_signal": per_tau(features[:, F_DR_SIGNAL], 0.0),
                "tau_dR_iso": per_tau(features[:, F_DR_ISO], 0.0),
            }
        )
        return retVal
