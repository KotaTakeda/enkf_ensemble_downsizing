import numpy as np


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
