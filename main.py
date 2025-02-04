import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lorenz96_cython import rk4_cython
from lorenz96_cython import lorenz96_cython
from da.l96 import lorenz96
from da.scheme import rk4
from da.loss import loss_rms
from da.visualize import plot_loss
from da.etkf import ETKF
import visualize as vis
from util import load_params, reduce_by_svd
import pandas as pd
from util import compute_edims, compute_traceP
import argparse


# ============================
# Generate true trajectory
# ============================


def generate_trajectory(J, F, dt, N, data_dir=""):
    # Model params
    print("(J, F):", (J, F))
    # dt: time step size
    # dt = 0.
    print(f"dt: {dt}")

    # N: number of time step, 1 year : 360*20
    assert N > 360 * 20
    print(f"N: {N}")

    # Initial state near the stationary point
    x0 = F * np.ones(J)  # the stationary point
    x0[19] *= 1.001  # perturb

    #
    scheme = rk4_cython
    # p = (F, )
    p = (np.ones(J) * F,)

    result = np.zeros((N, len(x0)))
    x = x0

    # Spin up 1 year
    for n in range(1, 360 * 20):
        t = n * dt
        # x = scheme(lorenz96, t, x, p, dt) # without using cython
        x = scheme(lorenz96_cython, t, x, *p, dt)

    # Nature run on the attractor
    result[0] = x[:]
    for n in range(1, N):
        t = n * dt
        # x = scheme(lorenz96, t, x, p, dt) # without using cython
        x = scheme(lorenz96_cython, t, x, *p, dt)
        result[n] = x[:]

    # save the result
    x_true = result[::obs_per]  # save per
    print("x_true.shape", x_true.shape)
    np.save(f"{data_dir}/x_true_l96", x_true)
    print("save x_true_l96")
    return x_true


# ============================
# Setup for OSSE
# ============================


def run_osse(
    J, F, obs_per, dt, r, N_spinup, m_reduced_list, alpha_list, seeds, data_dir=""
):
    # Model parameter
    p = (np.ones(J) * F,)

    # Assimilation interval
    Dt = obs_per * dt

    # Model function for EnKF
    # def M(x, Dt):
    #     for _ in range(int(Dt/dt)):
    #         x = rk4(lorenz96, 0, x, p, dt)
    #     return x

    def M_cython(x, Dt):
        for _ in range(int(Dt / dt)):
            x = rk4_cython(lorenz96_cython, 0, x, *p, dt)
        return x

    # Observation
    H_diag = np.ones(J)
    H = np.diag(H_diag)
    print("diag of H:", H_diag)
    print("rank(H):", np.linalg.matrix_rank(H))

    # Model error covariance
    Q = np.zeros((J, J))  # not used

    # Observation error covariance
    print("r:", r)
    R = r**2 * np.eye(J)

    # Load true trajectory
    x_true = np.load(f"{data_dir}/x_true_l96.npy")

    # Initial ensemble size
    m0 = J + 1

    # ============================
    # Run OSSE
    # ============================
    Xa_dict = {}
    param_dict = {}
    filename = data_dir + "/{}-{}{}{}"

    # loop length
    print(f"{len(m_reduced_list)*len(alpha_list)*len(seeds)} loops")
    param_list = []
    for i, m_reduced in enumerate(m_reduced_list):
        for j, alpha in enumerate(alpha_list):
            for k, seed in enumerate(seeds):
                param_dict[(i, j, k)] = (m_reduced, alpha, seed)
                param_list.append(
                    {
                        "m_reduced": m_reduced,
                        "alpha": alpha,
                        "seed": seed,
                        "idx": (i, j, k),
                    }
                )
                try:
                    Xa = np.load(filename.format("xa", i, j, k) + ".npy")
                    Xa_spinup = np.load(filename.format("xa_spinup", i, j, k) + ".npy")
                    Xa = [*Xa_spinup, *Xa]
                    print(filename.format("xa", i, j, k) + ".npy loaded", Xa.shape)
                except FileNotFoundError:
                    print(i, j, k)

                    np.random.seed(seed)
                    # generate obs
                    y = (H @ x_true.T).T
                    y += np.random.normal(loc=0, scale=r, size=y.shape)  # R = r^2*I

                    # generate initial ensemble
                    x_0 = x_true[np.random.randint(len(x_true) - 1)]
                    P_0 = 25 * np.eye(J)
                    X_0 = x_0 + np.random.multivariate_normal(
                        np.zeros_like(x_0), P_0, m0
                    )  # (m0, dim_x)

                    # run spin-up
                    print("spin-up")
                    etkf = ETKF(M_cython, H, R, alpha=alpha, store_ensemble=True)
                    etkf.initialize(X_0)
                    for y_obs in tqdm(y[:N_spinup]):
                        etkf.forecast(Dt)
                        etkf.update(y_obs)

                    # save spin-up data
                    np.save(filename.format("xf_spinup", i, j, k), etkf.Xf)
                    np.save(filename.format("xa_spinup", i, j, k), etkf.Xa)
                    Xa_spinup = etkf.Xa

                    # reduce ensemble
                    X_reduced = reduce_by_svd(etkf.X, m_reduced)  # by SVD
                    # X_reduced = reduce_by_sample(etkf.X, m_reduced) # by random sampling
                    etkf.initialize(X_reduced)
                    # etkf.alpha = 1.0
                    print("assimilation")
                    for y_obs in tqdm(y[N_spinup:]):
                        etkf.forecast(Dt)
                        etkf.update(y_obs)

                    # save data
                    np.save(filename.format("xf", i, j, k), etkf.Xf)
                    np.save(filename.format("xa", i, j, k), etkf.Xa)

                    Xa = [*Xa_spinup, *etkf.Xa]
                    del Xa_spinup, etkf
                Xa_dict[(i, j, k)] = Xa
                del Xa
    return Xa_dict, param_dict


# ============================
# Summarize
# ============================


def summarize_results(m_reduced_list, alpha_list, seeds, N_spinup, T_inf, data_dir=""):
    # Load true trajectory
    x_true = np.load(f"{data_dir}/x_true_l96.npy")

    # Set filename format
    filename = data_dir + "/{}-{}{}{}"

    sup_se = np.zeros((len(m_reduced_list), len(alpha_list)))
    rmse = np.zeros((len(m_reduced_list), len(alpha_list)))
    # traceP = np.zeros((len(m_reduced_list), len(alpha_list)))
    ensdim = np.zeros((len(m_reduced_list), len(alpha_list)))
    for i, m_reduced in enumerate(m_reduced_list):
        for j, alpha in enumerate(alpha_list):
            se_tmp = []
            rmse_tmp = []
            # traceP_tmp = []
            ensdim_tmp = []
            print(i, j)
            for k, seed in enumerate(seeds):
                Xa = np.load(filename.format("xa", i, j, k) + ".npy")
                e = x_true[N_spinup:] - Xa.mean(axis=1)

                se_tmp.append(np.sum(e[T_inf:] ** 2, axis=-1))  # (T,)
                rmse_tmp.append(
                    np.mean(np.linalg.norm(e[T_inf:], axis=-1) / np.sqrt(J))
                )
                # traceP_tmp.append(np.mean(np.sqrt(compute_traceP(Xa[T_inf:]) / J)))
                ensdim_tmp.append(np.mean(compute_edims(Xa[T_inf:])))

            sup_se[i, j] = np.max(
                np.mean(np.array(se_tmp), axis=0), axis=0
            )  # sup_t(E[SE])
            rmse[i, j] = np.mean(rmse_tmp)  # E[mean_t(RMSE)]
            # traceP[i, j] = np.mean(traceP_tmp)  # E[mean_t(tr(Pa))]
            ensdim[i, j] = np.mean(ensdim_tmp)  # E[mean_t(D_ens)]

    df_sup_se = pd.DataFrame(sup_se, index=m_reduced_list, columns=alpha_list)
    df_rmse = pd.DataFrame(rmse, index=m_reduced_list, columns=alpha_list)
    # df_traceP = pd.DataFrame(traceP, index=m_reduced_list, columns=alpha_list)
    df_ensdim = pd.DataFrame(ensdim, index=m_reduced_list, columns=alpha_list)

    df_sup_se.to_csv(data_dir + "/" + "sup_se" + ".csv")
    df_rmse.to_csv(data_dir + "/" + "rmse" + ".csv")
    df_ensdim.to_csv(data_dir + "/" + "ensdim" + ".csv")
    return df_sup_se, df_rmse, df_ensdim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OSSE for Lorenz96 model")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory to save data"
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    # Use the function to load the parameters
    set_params = load_params(data_dir)

    # Access the parameters
    J = set_params.J
    F = set_params.F
    dt = set_params.dt
    N = set_params.N
    obs_per = set_params.obs_per
    T_inf = set_params.T_inf
    r = set_params.r
    N_spinup = set_params.N_spinup
    m_reduced_list = set_params.m_reduced_list
    alpha_list = set_params.alpha_list
    seeds = set_params.seeds

    x_true = generate_trajectory(J, F, dt, N, data_dir)
    Xa_dict, param_dict = run_osse(
        J, F, obs_per, dt, r, N_spinup, m_reduced_list, alpha_list, seeds, data_dir
    )
    df_sup_se, df_rmse, df_ensdim = summarize_results(
        m_reduced_list, alpha_list, seeds, N_spinup, T_inf, data_dir
    )
