import pytest
import numpy as np
from rdkit import Chem
from drug_env import DrugDiscoveryEnv


@pytest.fixture
def env():
    smiles_list = ["CCO", "C1=CC=CC=C1", "CC(=O)O"]
    return DrugDiscoveryEnv(smiles_list)


def test_env_initialization(env):
    assert env.action_space.n == 3
    assert env.observation_space.shape == (1,)
    assert isinstance(env.smiles_list, list)
    assert len(env.smiles_list) > 0


def test_env_reset_returns_observation(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (1,)
    assert 0.0 <= obs[0] <= 1.0


def test_step_do_nothing(env):
    env.reset()
    obs, reward, done, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert done is True
    assert truncated is False


def test_step_add_methyl(env):
    env.reset()
    obs, reward, done, truncated, info = env.step(1)
    assert isinstance(obs, np.ndarray)
    assert done is True


def test_step_add_hydroxyl(env):
    env.reset()
    obs, reward, done, truncated, info = env.step(2)
    assert isinstance(obs, np.ndarray)
    assert done is True


def test_invalid_smiles_fallback():
    # This tests _get_logp fallback behavior
    bad_env = DrugDiscoveryEnv(["INVALID_SMILES"])
    obs, _ = bad_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (1,)
    # It should fallback to -10 logP → (logP + 5) / 10 = -0.5 → clipped to 0.0
    assert 0.0 <= obs[0] <= 1.0


def test_render_rgb_array(env):
    env.reset()
    rgb = env.render(mode="rgb_array")
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape[-1] == 3  # RGB image
