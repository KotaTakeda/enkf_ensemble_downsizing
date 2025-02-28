import numpy as np
from tqdm import tqdm
from lorenz96_cython import rk4_cython
from lorenz96_cython import lorenz96_cython

from da.etkf import ETKF
from util import load_params, reduce_by_svd
import pandas as pd
from util import compute_edims
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


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
    print(f"N: {N}")

    try:
        x_true = np.load(f'{data_dir}/x_true_l96.npy')
        print('x_true_l96 loaded')
    except FileNotFoundError:
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
        np.save(f"{data_dir}/x_true_l96", x_true)
        print("save x_true_l96")
    
    print("x_true.shape", x_true.shape)

    return x_true


# ============================
# OSSE
# ============================


class OSSE:
    def __init__(
        self,
        J,
        F,
        obs_per,
        dt,
        r,
        N_spinup,
        m_reduced_list,
        alpha_list,
        seeds,
        data_dir="",
    ):
        # Model parameters
        self.J = J
        self.p = (np.ones(J) * F,)

        # Assimilation interval
        self.dt = dt
        self.Dt = obs_per * dt

        # Observation
        H_diag = np.ones(J)
        self.H = np.diag(H_diag)
        print("diag of H:", H_diag)
        print("rank(H):", np.linalg.matrix_rank(self.H))

        # Model error covariance
        self.Q = np.zeros((J, J))  # not used

        # Observation error covariance
        print("r:", r)
        self.r = r
        self.R = r**2 * np.eye(J)

        # Initial ensemble size
        self.m0 = J + 1

        # Number of spin-up steps
        self.N_spinup = N_spinup

        # m_reduced_list
        self.m_reduced_list = m_reduced_list

        # alpha_list
        self.alpha_list = alpha_list

        # seeds
        self.seeds = seeds

        # data
        self.data_dir = data_dir

        # Filename format
        self.savename = self.data_dir + "/{}-{}{}{}"

        # Load true trajectory
        # TODO: generate(N, N_spinup)
        self.x_true = np.load(f"{self.data_dir}/x_true_l96.npy")

    def M_cython(self, x, Dt):
        N = int(Dt / self.dt)
        for _ in range(N):
            x = rk4_cython(lorenz96_cython, 0, x, *self.p, self.dt)
        if Dt - N * self.dt > 0:
            x = rk4_cython(lorenz96_cython, 0, x, *self.p, Dt - N * self.dt)
        return x
    
    def process(self, i, j, k, m_reduced, alpha, seed):
        try:
            Xa = np.load(self.savename.format("xa", i, j, k) + ".npy")
            Xa_spinup = np.load(self.savename.format("xa_spinup", i, j, k) + ".npy")
            Xa = [*Xa_spinup, *Xa]
            print(self.savename.format("xa", i, j, k) + ".npy loaded")
        except FileNotFoundError:
            print(i, j, k)

            np.random.seed(seed)
            # generate obs
            y = (self.H @ self.x_true.T).T
            y += np.random.normal(loc=0, scale=self.r, size=y.shape)  # R = r^2*I

            # generate initial ensemble
            x_0 = self.x_true[np.random.randint(len(self.x_true) - 1)]
            P_0 = 25 * np.eye(self.J)
            X_0 = x_0 + np.random.multivariate_normal(
                np.zeros_like(x_0), P_0, self.m0
            )  # (m0, dim_x)

            # run spin-up
            print("spin-up")
            etkf = ETKF(self.M_cython, self.H, self.R, alpha=alpha, store_ensemble=True)
            etkf.initialize(X_0)
            for y_obs in tqdm(y[: self.N_spinup]):
                etkf.forecast(self.Dt)
                etkf.update(y_obs)

            # save spin-up data
            # np.save(self.savename.format("xf_spinup", i, j, k), etkf.Xf)
            np.save(self.savename.format("xa_spinup", i, j, k), etkf.Xa)
            Xa_spinup = etkf.Xa

            # reduce ensemble
            X_reduced = reduce_by_svd(etkf.X, m_reduced)  # by SVD
            # X_reduced = reduce_by_sample(etkf.X, m_reduced) # by random sampling
            etkf.initialize(X_reduced)
            # etkf.alpha = 1.0
            print("assimilation")
            for y_obs in tqdm(y[self.N_spinup :]):
                etkf.forecast(self.Dt)
                etkf.update(y_obs)

            # save data
            # np.save(self.savename.format("xf", i, j, k), etkf.Xf)
            np.save(self.savename.format("xa", i, j, k), etkf.Xa)

            Xa = [*Xa_spinup, *etkf.Xa]
            del Xa_spinup, etkf
        return Xa

    # Run OSSE
    def run(self, parallel="none"):
        assert parallel in [
            "none",
            "mp",
        ]  # none: no parallel, mp: multi process, mt is not supported

        Xa_dict = {}
        param_dict = {}
        # Loop
        print(
            f"{len(self.m_reduced_list)*len(self.alpha_list)*len(self.seeds)} loops"
        )  # Loop size
        if parallel == "none":
            for i, m_reduced in enumerate(self.m_reduced_list):
                for j, alpha in enumerate(self.alpha_list):
                    for k, seed in enumerate(self.seeds):
                        Xa_dict[(i, j, k)] = self.process(
                            i, j, k, m_reduced, alpha, seed
                        )
                        param_dict[(i, j, k)] = (m_reduced, alpha, seed)
        else:
            executor = (
                ProcessPoolExecutor() if parallel == "mp" else ThreadPoolExecutor()
            )
            with executor as exe:
                features = []
                param_list = []
                for i, m_reduced in enumerate(self.m_reduced_list):
                    for j, alpha in enumerate(self.alpha_list):
                        for k, seed in enumerate(self.seeds):
                            features.append(
                                exe.submit(
                                    self.process, i, j, k, m_reduced, alpha, seed
                                )
                            )
                            param_list.append(
                                {
                                    "m_reduced": m_reduced,
                                    "alpha": alpha,
                                    "seed": seed,
                                    "idx": (i, j, k),
                                }
                            )
                for i, f in enumerate(features):
                    Xa_dict[param_list[i]["idx"]] = f.result()
                    param_dict[param_list[i]["idx"]] = (
                        param_list[i]["m_reduced"],
                        param_list[i]["alpha"],
                        param_list[i]["seed"],
                    )

        return Xa_dict, param_dict
    

    # NOTE: E[se]/J \ge E[RMSE]^2
    def summarize_results(self, T_inf):
        """
        T_inf, integer (>0): compute sup/mean of the errors after t=T_inf, e.g. sup_t(E[SE[T_inf:]]).
        """
        try:
            df_sup_se = pd.read_csv(f"{self.data_dir}/sup_se.csv", index_col=0, header=0)
            df_mean_se = pd.read_csv(f"{self.data_dir}/mean_se.csv", index_col=0, header=0)
            df_sup_rmse = pd.read_csv(f"{self.data_dir}/sup_rmse.csv", index_col=0, header=0)
            df_mean_rmse = pd.read_csv(f"{self.data_dir}/mean_rmse.csv", index_col=0, header=0)
            # df_ensdim = pd.read_csv(f"{self.data_dir}/ensdim.csv", index_col=0, header=0)
        except FileNotFoundError:
            sup_se = np.zeros((len(m_reduced_list), len(alpha_list)))
            mean_se = np.zeros((len(m_reduced_list), len(alpha_list)))
            sup_rmse = np.zeros((len(m_reduced_list), len(alpha_list)))
            mean_rmse = np.zeros((len(m_reduced_list), len(alpha_list)))
            # traceP = np.zeros((len(m_reduced_list), len(alpha_list)))
            # ensdim = np.zeros((len(m_reduced_list), len(alpha_list)))
            for i, _ in enumerate(m_reduced_list):
                for j, _ in enumerate(alpha_list):
                    se_tmp = []
                    rmse_tmp = []
                    # traceP_tmp = []
                    # ensdim_tmp = []
                    print(i, j)
                    for k, _ in enumerate(seeds):
                        Xa = np.load(self.savename.format("xa", i, j, k) + ".npy")
                        se_tail = np.sum((self.x_true[N_spinup:] - Xa.mean(axis=1))[T_inf:] ** 2, axis=-1)

                        # SE
                        se_tmp.append(se_tail)  # (T,)

                        # RMSE
                        # assert np.allclose(np.sqrt(se_tail), np.linalg.norm(e[T_inf:], axis=-1))  # test
                        rmse_tmp.append(np.sqrt(se_tail / J))

                        # tr(P)
                        # traceP_tmp.append(np.mean(np.sqrt(compute_traceP(Xa[T_inf:]) / J)))

                        # D_ens
                        # ensdim_tmp.append(np.mean(compute_edims(Xa[T_inf:])))

                    se_tmp = np.array(se_tmp)  # (m, T - T_inf)
                    rmse_tmp = np.array(rmse_tmp)
                    assert np.allclose(np.sqrt(se_tmp / J), rmse_tmp)
                    print("se_tmp = (#seeds, T - T_inf)", se_tmp.shape)

                    sup_se[i, j] = np.max(np.mean(se_tmp, axis=0), axis=0)  # sup_t(E[SE])
                    mean_se[i, j] = np.mean(se_tmp)  # mean_t(E[SE])
                    sup_rmse[i, j] = np.max(np.mean(np.sqrt(se_tmp / self.J), axis=0), axis=0)  # sup_t(E[RMSE])
                    mean_rmse[i, j] = np.mean(np.sqrt(se_tmp / self.J))  # mean_t(E[RMSE])
                    # traceP[i, j] = np.mean(traceP_tmp)  # E[mean_t(tr(Pa))]
                    # ensdim[i, j] = np.mean(ensdim_tmp)  # E[mean_t(D_ens)]

            df_sup_se = pd.DataFrame(sup_se, index=m_reduced_list, columns=alpha_list)
            df_mean_se = pd.DataFrame(mean_se, index=m_reduced_list, columns=alpha_list)
            df_sup_rmse = pd.DataFrame(sup_rmse, index=m_reduced_list, columns=alpha_list)
            df_mean_rmse = pd.DataFrame(mean_rmse, index=m_reduced_list, columns=alpha_list)
            # df_traceP = pd.DataFrame(traceP, index=m_reduced_list, columns=alpha_list)
            # df_ensdim = pd.DataFrame(ensdim, index=m_reduced_list, columns=alpha_list)

            df_sup_se.to_csv(self.data_dir + "/" + "sup_se" + ".csv")
            df_sup_rmse.to_csv(self.data_dir + "/" + "sup_rmse" + ".csv")
            df_mean_se.to_csv(self.data_dir + "/" + "mean_se" + ".csv")
            df_mean_rmse.to_csv(self.data_dir + "/" + "mean_rmse" + ".csv")
            # df_ensdim.to_csv(self.data_dir + "/" + "ensdim" + ".csv")
        return df_sup_se, df_sup_rmse, df_mean_se, df_mean_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OSSE for Lorenz96 model")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory to save data"
    )
    parser.add_argument(
        "--parallel", type=str, default="none", help="Specify parallelization method"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    parallel = args.parallel

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

    _ = generate_trajectory(J, F, dt, N, data_dir)

    osse = OSSE(
        J,
        F,
        obs_per,
        dt,
        r,
        N_spinup,
        m_reduced_list,
        alpha_list,
        seeds,
        data_dir=data_dir,
    )
    _ = osse.run(parallel=parallel)
    _ = osse.summarize_results(T_inf)
