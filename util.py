import os
import numpy as np
import scipy as sp

# Save and load numpy arrays with specified precision
def npsave(savename, array, precision='float32'):
    """
    Save a numpy array to a file with the specified precision.
    
    Args:
        savename (str): The name of the file to save the array to.
        array (array-like): The array to save.
    precision (str): The precision to use when saving the array. Default is 'float32'.
    """
    savearray = np.array(array, dtype=precision)
    np.save(savename, savearray)

def npload(savename, precision='float64'):
    """
    Load a numpy array from a file with the specified precision.
    
    Args:
        savename (str): The name of the file to load the array from.
    
    Returns:
        np.ndarray: The loaded numpy array.
    """
    return np.load(savename).astype(precision)

# Define the function to load the module
def load_params(path_str):
    import importlib.util

    # Construct the path to the set_params.py file
    params_path = os.path.join(path_str, "set_params.py")

    # Load the module
    spec = importlib.util.spec_from_file_location("set_params", params_path)
    set_params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(set_params)

    return set_params

# Estimate data size
def estimate_data_size(params):
    """Estimate the total data size in GB based on the parameters (ignored: true trajectory, observation, spin-up ensemble)."""
    dimension = params.J
    bytes_per_point = 8  # float64
    total_save_steps = params.N // params.obs_per
    num_ensemble_members = sum(params.m_reduced_list)
    num_seeds = len(params.seeds)
    num_param_variations = len(params.alpha_list)

    total_bytes = (
        bytes_per_point *
        dimension *
        total_save_steps *
        num_ensemble_members *
        num_seeds *
        num_param_variations
    )

    total_gb = total_bytes / (1000**3)
    return total_gb

# Ensemble reduction
def reduce_by_svd(X, m_reduced, method="helmert"):
    """Reduce an ensemble using a truncated singular value decomposition.

    The default Helmert construction preserves the ensemble mean and the
    sample covariance represented by the retained singular vectors.

    Args:
        X: ``(m, Nx)`` array of ensemble vectors.
        m_reduced: Reduced ensemble size. Must satisfy ``2 <= m_reduced <= m``.
        method: ``"helmert"`` (default) or ``None``. ``None`` returns the
            unscaled principal-component ensemble used by earlier versions.

    Returns:
        A ``(m_reduced, Nx)`` array of reduced ensemble vectors.

    Raises:
        ValueError: If the input shape, reduced size, or method is invalid.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")

    m, _ = X.shape
    if isinstance(m_reduced, (bool, np.bool_)) or not isinstance(
        m_reduced, (int, np.integer)
    ):
        raise ValueError("m_reduced must be an integer")
    if not 2 <= m_reduced <= m:
        raise ValueError("m_reduced must satisfy 2 <= m_reduced <= X.shape[0]")
    if method not in ("helmert", None):
        raise ValueError('method must be "helmert" or None')

    xmean = X.mean(axis=0)
    dX = X - xmean[None, :]
    U, S, _ = sp.linalg.svd(dX.T)
    if method == "helmert":
        n_modes = m_reduced - 1
        Q = sp.linalg.helmert(m_reduced)  # (n_modes, m_reduced)
        covariance_scale = np.sqrt(n_modes / (m - 1))
        retained_modes = min(n_modes, S.size)
        components = np.zeros((X.shape[1], n_modes), dtype=S.dtype)
        components[:, :retained_modes] = (
            U[:, :retained_modes] * S[:retained_modes]
        )
        dX_reduced = covariance_scale * components @ Q
        # dX_reduced has shape (Nx, m_reduced) and zero column mean.
    else:
        dX_reduced = U[:, :m_reduced] * S[:m_reduced]
    X_reduced = xmean[None, :] + dX_reduced.T
    return X_reduced


def reduce_by_sample(X, m_reduced):
    """
    Reduce the ensemble size using random sampling.
    This method does not keep the ensemble mean.
    Args:
        X: (m, Nx) array of ensemble vectors
        m_reduced: reduced ensemble size
    Returns:
        X_reduced: (m_reduced, Nx) array of reduced ensemble vectors
    """
    m, _ = X.shape
    return X[np.random.choice(m, m_reduced)]


# Compute metrics
def compute_traceP(X):
    # X: (T, Ne, Nx)
    trP = []
    for Xt in X:
        dXt = Xt - Xt.mean(axis=0, keepdims=True)
        P = np.cov(dXt.T)
        trP.append(np.trace(P))
    return np.array(trP)


def edim(B):
    sigma = np.linalg.svd(B, compute_uv=False)
    return np.sum(sigma) ** 2 / np.sum(sigma**2)


def compute_edims(X):
    edims = []
    for Xt in X:
        dXt = Xt - Xt.mean(axis=0, keepdims=True)
        edims.append(edim(dXt))
    return np.array(edims)
