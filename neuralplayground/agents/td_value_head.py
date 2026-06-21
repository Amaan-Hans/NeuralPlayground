"""Tabular TD(0) value head for TEM-R, indexed by physical grid state.

Each environment's table has one entry per state it actually has (room sizes
differ across the 16 parallel environments). Keying the table by state id
means two states that happen to hold the identical sensory object can still
carry different values — breaking the ambiguity created by having far fewer
distinct objects than states, where many states are indistinguishable from
TEM's sensory input alone.
"""

import numpy as np


class TDValueHead:
    """Per-environment TD(0) value estimator over physical grid states.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments (batch size).
    n_keys_per_env : list[int]
        Number of distinct states in each environment (``agent.n_states``).
    alpha : float
        TD learning rate.
    gamma : float
        Discount factor.
    """

    def __init__(self, n_envs: int, n_keys_per_env: list, alpha: float = 0.1, gamma: float = 0.9):
        self.n_envs = n_envs
        self.alpha = alpha
        self.gamma = gamma
        # One value table per environment, indexed by state id.
        self.V = [np.zeros(n) for n in n_keys_per_env]
        self._prev_key = [None] * n_envs

    def update(self, keys: list, rewards: list) -> np.ndarray:
        """Run one TD(0) backup per environment and return current value estimates.

        Parameters
        ----------
        keys : list[int or None], length n_envs
            State id observed this step for each environment. ``None`` if no
            valid observation is available yet (e.g. the dummy pre-reset
            placeholder) — that environment is skipped this step.
        rewards : list[float], length n_envs
            Reward received on arriving at keys[i].

        Returns
        -------
        values : np.ndarray, shape (n_envs,)
            V(s_t) for each environment (0.0 where keys[i] is None).
        """
        values = np.zeros(self.n_envs)
        for i in range(self.n_envs):
            k = keys[i]
            if k is None:
                continue
            r = rewards[i]
            k_prev = self._prev_key[i]
            if k_prev is not None:
                delta = r + self.gamma * self.V[i][k] - self.V[i][k_prev]
                self.V[i][k_prev] += self.alpha * delta
            self._prev_key[i] = k
            values[i] = self.V[i][k]
        return values

    def reset_env(self, env_idx: int):
        """Reset value estimates for a single environment (e.g. on episode end)."""
        self.V[env_idx][:] = 0.0
        self._prev_key[env_idx] = None

    def reset_all(self):
        """Reset all environments."""
        for i in range(self.n_envs):
            self.reset_env(i)
