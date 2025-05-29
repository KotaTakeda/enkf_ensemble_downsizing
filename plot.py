import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as vis
from util import load_params


def summarize_rm(data_dir_r, logr_list, target_m, error_type="sup_se"):
    """
    Summarize the error for different r and m_reduced.
    Args:
        data_dir_r: str, directory where the data is stored. e.g. "data/l96/r{logr}".
        logr_list: list of float, list of logr values.
        target_m: int, target m_reduced value.
        error_type: str, filename of the error. ["sup_se", "mean_se", "sup_rmse", "mean_rmse"].
    Returns:
        fig, ax: matplotlib figure and axis objects.
    """
    # load params
    set_params = load_params(
        f"{data_dir_r}/r0"
    )  # assume r0 exists and has the same params except r
    m_reduced_list = set_params.m_reduced_list

    # load true data
    # x_true = np.load(f'{data_dir_r.format(logr_list[0])}/x_true_l96.npy')
    # rho = 2*np.linalg.norm(x_true, axis=-1).mean()  # time-averaged 2-norm of x_true

    # rmse_r = np.zeros((len(logr_list), len(m_reduced_list)))
    error_r = np.zeros((len(logr_list), len(m_reduced_list)))
    for i, logr in enumerate(logr_list):
        dir = f"{data_dir_r}/r{logr}"
        print(dir)

        # load params
        set_params = load_params(dir)

        # load data
        # df_mean_rmse = pd.read_csv(dir + "/mean_rmse.csv", index_col=0, header=0)
        df_error = pd.read_csv(dir + f"/{error_type}.csv", index_col=0, header=0)
        # print(df_error)

        # optimal for each m
        # rmse_opt = df_mean_rmse.to_numpy().min(axis=1)  # (len(m_reduced_list), )
        error_opt = df_error.to_numpy().min(axis=1)  # (len(m_reduced_list), )
        # rmse_r[i] = rmse_opt
        error_r[i] = error_opt

    J = set_params.J
    i_target_m = m_reduced_list.index(target_m)
    n_m_reduced = len(m_reduced_list)
    # print(f"i_target_m = {i_target_m}, n_m_reduced = {n_m_reduced}")
    if i_target_m >= n_m_reduced // 2:
        n_colors = 2 * (i_target_m) + 1
        print(n_colors)
        colors = vis.get_color_palette(n_colors, "coolwarm")[:n_m_reduced]
    else:
        n_colors = 2 * (n_m_reduced - i_target_m + 1) + 1  # 0, 1, 2, 3, 4, 5, 6
        colors = vis.get_color_palette(n_colors, "coolwarm")[
            n_m_reduced - i_target_m - 1 :
        ]
    markers = vis.get_marker_cycle()
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(m_reduced_list):
        ax.plot(
            0.1 ** (np.array(logr_list)),
            error_r[:, i],
            label=f"m = {m}",
            marker=next(markers),
            color=colors[i],
        )
    ax.plot(
        0.1 ** (np.array(logr_list)),
        J * 0.01 ** (np.array(logr_list)),
        color="black",
        # ls="-",
        lw=0.5,
        label="obs. noise level",
    )  # Jr^2
    # plt.plot(0.1**(np.array(logr_list)), rho**2*np.ones_like(logr_list), color="black", ls="-", label="$ \\rho^2 $")  # plot attractor radius
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r")
    ax.set_ylabel("error")
    ax.legend(bbox_to_anchor=(1, 1))
    fig.tight_layout()

    return fig, ax