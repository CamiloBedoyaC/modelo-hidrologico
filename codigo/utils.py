import numpy as np


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() == 0:
        return -np.inf
    obs = obs[mask]
    sim = sim[mask]
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return -np.inf
    return 1 - np.sum((obs - sim) ** 2) / denom


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency (2009)."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() == 0:
        return -np.inf
    obs = obs[mask]
    sim = sim[mask]
    r = np.corrcoef(obs, sim)[0, 1] if obs.size > 1 else np.nan
    alpha = np.std(sim) / np.std(obs) if np.std(obs) != 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return -np.inf
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def ensure_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(-1)
