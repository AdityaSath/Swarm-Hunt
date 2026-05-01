import json
import os
import shutil
from typing import Any

from stable_baselines3 import PPO
import numpy as np


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def save_locked_policy(name: str, source: str) -> str:
    """Lock a policy under `models/{name}`.

    source can be:
      - 'scripted' to lock the scripted_actions policy (no model file), or
      - path to an SB3 .zip model file which will be copied into models/{name}.zip

    Returns path to the manifest file.
    """
    manifest = {'name': name}
    if source == 'scripted':
        manifest['type'] = 'scripted'
        manifest_path = os.path.join(MODELS_DIR, f'{name}.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        return manifest_path

    # otherwise assume a file path to an SB3 archive
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source model not found: {source}")
    dest = os.path.join(MODELS_DIR, f'{name}.zip')
    shutil.copyfile(source, dest)
    manifest['type'] = 'sb3'
    manifest['path'] = os.path.basename(dest)
    manifest_path = os.path.join(MODELS_DIR, f'{name}.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    return manifest_path


class LockedPolicy:
    """Runtime wrapper for a locked policy. Provides a unified .predict(obs) API
    that returns a flattened centralized action vector (as used by CentralizedSwarmGym).
    """

    def __init__(self, manifest_path: str):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.type = manifest.get('type')
        self._model = None

    def _load_sb3(self, repo_env):
        path = os.path.join(MODELS_DIR, self.manifest['path'])
        self._model = PPO.load(path, env=None)

    def predict(self, obs: Any, repo_env=None, deterministic: bool = True) -> np.ndarray:
        """Return a flattened centralized action vector.

        obs: for SB3 policies supply the flattened observation expected by the model.
        repo_env: when using the scripted policy we need access to the underlying
                  `Environment` instance to call `scripted_actions()`.
        """
        if self.type == 'scripted':
            if repo_env is None:
                raise ValueError('repo_env must be provided for scripted policies')
            # scripted_actions returns a dict agent->(vx,vy)
            scripted = repo_env.scripted_actions()
            n = repo_env.num_agents
            vec = np.zeros((2 * n,), dtype=np.float32)
            for i in range(n):
                vx, vy = scripted.get(i, (0.0, 0.0))
                vec[2 * i] = float(vx)
                vec[2 * i + 1] = float(vy)
            return vec

        if self.type == 'sb3':
            if self._model is None:
                self._load_sb3(repo_env)
            # stable-baselines3 expects np arrays
            action, _ = self._model.predict(obs, deterministic=deterministic)
            return np.asarray(action, dtype=np.float32)

        raise RuntimeError(f'Unknown locked policy type: {self.type}')


def get_manifest_path(name: str) -> str:
    path = os.path.join(MODELS_DIR, f'{name}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Locked policy manifest not found: {path}')
    return path
