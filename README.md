# HPyS

Python implementation of the hadrons-plus-strips (HPS) algorithm for reconstructing hadronically decaying tau leptons,
adapted to e+e- collisions (FCC-ee / CLIC). It uses the polar angle θ instead of the pseudorapidity η and the 3D opening angle instead of ΔR.

The aim of this repository is to optimize the HPS cuts for the e+e- environment. Every cut is a hyperparameter in
[`hpys/config/hps.yaml`](hpys/config/hps.yaml). The defaults follow the CMS HPS algorithm ([JINST 13 (2018) P10005](https://arxiv.org/abs/1809.02816)).

The algorithm, per reconstructed jet:

1. **Strips** (`StripAlgo`): e/γ candidates are clustered into strips, seeded in order of decreasing pT. The strip position is updated after each iteration (or after each candidate, `updateStripAfterEachCand`), for at most `maxStripBuildIterations` iterations.
   - `stripSize: dynamic` (default): a candidate is merged if Δθ < f(x_cand) + f(x_strip) and Δφ < g(x_cand) + g(x_strip), with f, g = a·x^-b restricted to [min, max] (CMS 2018: f = 0.20·pT^-0.66 in [0.05, 0.15], g = 0.35·pT^-0.71 in [0.05, 0.30]).
   - `stripSize: static`: fixed window, |Δθ| < 0.05, |Δφ| < 0.20 (CMS 2012).
2. **Combinatorics** (`decayModes`): charged candidates (h±, e±) and strips are combined into tau candidates for the decay modes
   h± (DM 0), h±π0 (DM 1), h±π0π0 (DM 2), h±h∓h± (DM 10) and h±h∓h±π0 (DM 11). Strips are assigned the π0 mass in the tau four-momentum; the optional strip-mass window (`min/maxStripMass`) acts on their visible mass.
3. **Preselection**:
   - |q| = 1 and leading charged candidate above `minLeadChargedCandPt`;
   - charged candidates and strip positions within the signal cone R_sig = `scale` / x, restricted to [`min`, `max`] (`signalCone`, default 3.0/pT in [0.05, 0.10]);
   - tau axis within `matchingConeSize` of the jet axis;
   - tau mass inside the decay-mode window. With `maxTauMassScaling` the upper limit is `maxTauMass`·sqrt(x / `refValue`), restricted to [`maxTauMass`, `maxLimit`]. With `stripMassCorrection` the window is widened on both sides by Δm, the change of the tau mass when the strip directions are varied by the strip size (CMS 2018, Eq. 5, written for θ instead of η).
4. **Isolation**: momentum sum of charged particles and photons within `isolationConeSize` of the tau axis that are not signal constituents. The momentum sum of e/γ in strips outside the signal cone is stored (`tauStripOuterSum`), and `tau_passesStripOuterCut` requires it to be below `maxStripOuterFraction` (0.10) of the tau momentum.
5. **Ranking** (`candidateRanking`): among the candidates that pass, the best one is chosen by comparing the listed criteria in order. Default is CMSSW's ordering: most charged constituents, then highest pT, then most strips, then lowest isolation. The CMS paper describes this as simply "the highest-pT candidate", which corresponds to `[pt]`.

`kinematicVariable` chooses whether all of the momentum-dependent quantities above (x) use pT, as in CMS, or p. Distances are 3D opening angles, so at e+e- colliders the collimation of the tau decay products scales with p rather than pT.

Jets without a tau candidate get a dummy tau with decay mode -1 so the jets and taus stay 1-to-1.

If several candidates are equal in a ranking criterion (within a relative 1e-9, i.e. up to floating-point rounding),
the next criterion decides, and among candidates that are equal in all criteria the first one built is kept.

## Implementation

- [`hpys/hps.py`](hpys/hps.py): the numba implementation. The jets and candidates are flattened into numpy arrays, and a
  compiled kernel processes the jets in parallel (about 10 µs per jet per thread). The configuration is converted into
  numeric arrays by `build_params`.
- [`hpys/reference.py`](hpys/reference.py): the original object-oriented implementation (about 50 ms per jet). It is kept
  as the reference: the tests require both implementations to give the same output for every field.

## Usage

```bash
pip install -e .[test]
hpys-process input.parquet output.parquet [-c my_config.yaml]
```

The default configuration is in [`hpys/config/hps.yaml`](hpys/config/hps.yaml). The input parquet is expected to contain the
`reco_jet_p4`, `reco_cand_*` and `event_reco_cand_*` fields produced by the ml-tau ntupelizer.

```python
from omegaconf import OmegaConf
from hpys import HPSTauBuilder

builder = HPSTauBuilder(OmegaConf.load("hpys/config/hps.yaml"))
taus = builder.process_jets(data)  # dict of awkward arrays, one entry per jet
```

## Hyperparameter optimization

[`hpys/tuning.py`](hpys/tuning.py) tunes the configuration with [Optuna](https://optuna.org) (multi-objective TPE),
on generator-level hadronic taus (decay modes 0, 1, 2, 10, 11):

1. balanced decay-mode accuracy (maximized): the mean over the decay modes of the fraction of taus reconstructed in the
   correct decay mode, so that the rare modes count as much as h±π0; taus that are not reconstructed count as wrong;
2. momentum resolution (minimized): half the 16–84% quantile range of p_reco / p_vis_gen relative to its median.

The first half of the jets is used for the optimization, and the Pareto-optimal configurations are evaluated on the
second half. The study starts from the default configuration and is stored in an SQLite database, so it can be resumed
or inspected with `optuna-dashboard`.

```bash
python -m hpys.tuning z_test.parquet --n-trials 3000 --output-dir optimization --study-name hps_tpe
```

The output directory gets one config file per Pareto-optimal trial (`trial_NNNNN.yaml`) and their held-out metrics
(`pareto_heldout.txt`, `.json`). One trial takes about 0.35 s for 137k jets.

## Tests

```bash
pytest tests
```

`tests/test_numba.py` compares the numba implementation with the reference on random jets and, if `z_test.parquet`
is present in the repository directory, on the first 300 jets of it.
Numba caches the compiled kernel; set `NUMBA_CACHE_DIR` if the package directory is not writable.
