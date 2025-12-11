import os
import numpy as np

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
def reduce_by_svd(X, m_reduced):
    # m, Nx = X.shape
    xmean = X.mean(axis=0)
    dX = X - xmean[None, :]
    U, S, _ = np.linalg.svd(dX.T)
    # dX_reduced = np.matmul(U[:, :m_reduced], np.diag(S[:m_reduced])@Vh[:m_reduced, :m_reduced])
    dX_reduced = U[:, :m_reduced] @ np.diag(S[:m_reduced])  # principal components
    X_reduced = xmean[None, :] + dX_reduced.T
    return X_reduced


def reduce_by_sample(X, m_reduced):
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
