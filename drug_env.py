import gym
from gym import spaces
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit import RDLogger

import numpy as np
import random
from PIL import Image
import logging


RDLogger.DisableLog('rdApp.*')

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


class DrugDiscoveryEnv(gym.Env):
    def __init__(self, smiles_list):
        super().__init__()
        self.smiles_list = smiles_list
        self.current_index = 0
        self.action_space = spaces.Discrete(3)  # 0: do nothing, 1: add methyl, 2: add hydroxyl
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        log.info("Initialized DrugDiscoveryEnv with %d molecules.", len(smiles_list))

    def reset(self):
        self.current_index = random.randint(0, len(self.smiles_list) - 1)
        self.smiles = self.smiles_list[self.current_index]
        self.mol = Chem.MolFromSmiles(self.smiles)
        log.debug("Reset environment with molecule: %s", self.smiles)
        return self._get_obs(), {}

    def step(self, action):
        log.debug("Action taken: %d", action)

        try:
            rw_mol = Chem.RWMol(self.mol)

            if action == 1:
                idx = rw_mol.AddAtom(Chem.Atom("C"))
                rw_mol.AddBond(0, idx, Chem.rdchem.BondType.SINGLE)

            elif action == 2:
                idx = rw_mol.AddAtom(Chem.Atom("O"))
                rw_mol.AddBond(0, idx, Chem.rdchem.BondType.SINGLE)

            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            self.mol = new_mol
            log.debug("Molecule modified successfully.")

        except Exception as e:
            log.warning("Failed to modify molecule: %s", str(e))

        reward = self._get_logp()
        done = True
        log.debug("Reward after action: %.3f", reward)
        return self._get_obs(), reward, done, False, {}

    def _get_logp(self):
        try:
            logp = Descriptors.MolLogP(self.mol)
            return logp
        except Exception as e:
            log.warning("Failed to compute logP: %s", str(e))
            return -10

    def _get_obs(self):
        raw = (self._get_logp() + 5) / 10
        return np.clip(np.array([raw], dtype=np.float32), 0.0, 1.0)

    def render(self, mode='human'):
        log.info("Rendering molecule...")
        img = Draw.MolToImage(self.mol, size=(250, 250))
        if mode == 'rgb_array':
            return np.array(img)
        elif mode == 'human':
            img.show()
