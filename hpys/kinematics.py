"""Kinematic helpers used by the HPS algorithm (copied from ml-tau-model)."""

import awkward as ak
import numpy as np
import vector


def angle3d(theta1, phi1, theta2, phi2):
    """Computes the true 3D opening angle between two directions in spherical
    coordinates via the dot product of unit vectors.

    Unlike the Euclidean approximation sqrt(Δθ² + Δφ²) used by deltaR_thetaPhi,
    this correctly accounts for the sin(θ) contraction of phi circles away from
    the equator and is exact for all separation sizes.
    """
    cos_angle = np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) + np.cos(
        theta1
    ) * np.cos(theta2)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def deltaPhi(phi1, phi2):
    """Calculates the difference in azimuthal angle of two objects.

    Args:
        phi1 : float
            The phi coordinate of the first object.
        phi2 : float
            The phi coordinate of the second object.

    Returns:
        dPhi : float
            The difference in azimuthal angle
    """
    diff = phi1 - phi2
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


def deltaTheta(theta1, theta2):
    """Calculates the difference in polar angle of two objects.

    Args:
        theta1 : float
            The theta coordinate of the first object.
        theta2 : float
            The theta coordinate of the second object.

    Returns:
        dTheta : float
            The difference in polar angle
    """
    return np.abs(theta1 - theta2)


def reinitialize_p4(p4_obj: ak.Array):
    """Reinitialized the 4-momentum for particle in order to access its properties.

    Args:
        p4_obj : ak.Array
            The particle represented by its 4-momenta

    Returns:
        p4 : ak.Array
            Particle with initialized 4-momenta, normalized to (rho, eta, phi, t).
    """
    # Normalise field names (aliases → canonical).
    name_map = {
        "rho": "pt",
        "x": "px",
        "y": "py",
        "z": "pz",
        "t": "energy",
        "e": "energy",
        "E": "energy",
        "tau": "mass",
        "m": "mass",
    }
    renamed = {name_map.get(f, f): p4_obj[f] for f in p4_obj.fields}

    # Pick the first complete non-redundant basis present in the data.
    # Mixing cylindrical (pt) and Cartesian (px/py) triggers vector's
    # "duplicate coordinates through momentum-aliases" error.
    for basis in (
        ("pt", "eta", "phi", "energy"),
        ("pt", "eta", "phi", "mass"),
        ("pt", "theta", "phi", "energy"),
        ("pt", "theta", "phi", "mass"),
        ("px", "py", "pz", "energy"),
        ("px", "py", "pz", "mass"),
    ):
        if all(k in renamed for k in basis):
            coords = {k: renamed[k] for k in basis}
            break
    else:
        raise ValueError(
            f"No supported 4-vector basis found in fields: {list(renamed)}"
        )

    p4 = vector.awk(ak.zip(coords))
    # Always return in (pt, eta, phi, energy) so downstream code can rely on
    # these being stored fields, not just computed properties.
    return vector.awk(
        ak.zip({"pt": p4.pt, "eta": p4.eta, "phi": p4.phi, "energy": p4.energy})
    )


def get_reduced_decaymodes(decaymodes: np.array):
    """Maps the full set of decay modes into a smaller subset, setting the rarer decaymodes under "Other" (# 15)"""
