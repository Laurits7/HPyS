import os

import numpy as np
import pytest
from omegaconf import OmegaConf

optuna = pytest.importorskip("optuna")

from hpys import hps, tuning  # noqa: E402

CFG = OmegaConf.load(tuning.DEFAULT_CONFIG)


def test_default_params_reproduce_default_config():
    cfg = tuning.suggest_config(optuna.trial.FixedTrial(tuning.default_params(CFG)), CFG)
    for a, b in zip(hps.build_params(cfg.builder), hps.build_params(CFG.builder)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("seed", range(5))
def test_sampled_configs_are_valid(seed):
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.RandomSampler(seed=seed))
    trial = study.ask()
    cfg = tuning.suggest_config(trial, CFG)
    P, DM, RANK = hps.build_params(cfg.builder)
    assert P[hps.P_SIGNAL_CONE_MAX] >= P[hps.P_SIGNAL_CONE_MIN]
    assert np.all(DM[:, hps.D_MAX_MASS] > DM[:, hps.D_MIN_MASS])
