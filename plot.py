import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as vis
from util import load_params


def summarize_r(data_dir_r, logr_list):
    # load params
    set_params = load_params(
        f"{data_dir_r}/r0"
    )  # assume r0 exists and has the same params except r
    m_reduced_list = set_params.m_reduced_list

    # load true data
    # x_true = np.load(f'{data_dir_r.format(logr_list[0])}/x_true_l96.npy')
    # rho = 2*np.linalg.norm(x_true, axis=-1).mean()  # time-averaged 2-norm of x_true

    rmse_r = np.zeros((len(logr_list), len(m_reduced_list)))
    sup_se_r = np.zeros((len(logr_list), len(m_reduced_list)))
    for i, logr in enumerate(logr_list):
        dir = f"{data_dir_r}/r{logr}"
        print(dir)

        # load params
        set_params = load_params(dir)

        # load data
        df_rmse = pd.read_csv(dir + "/rmse.csv", index_col=0, header=0)
        df_sup_se = pd.read_csv(dir + "/sup_se.csv", index_col=0, header=0)
        # print(df_sup_se)

        # optimal for each m
        rmse_opt = df_rmse.to_numpy().min(axis=1)  # (len(m_reduced_list), )
        sup_se_opt = df_sup_se.to_numpy().min(axis=1)  # (len(m_reduced_list), )
        rmse_r[i] = rmse_opt
        sup_se_r[i] = sup_se_opt

    J = set_params.J
    colors = vis.get_color_palette(len(m_reduced_list) + 1, "coolwarm")
    markers = vis.get_marker_cycle()
    fig = plt.figure(figsize=(6, 4))
    for i, m in enumerate(m_reduced_list):
        plt.plot(
            0.1 ** (np.array(logr_list)),
            sup_se_r[:, i],
            label=f"m = {m}",
            marker=next(markers),
            color=colors[i + 1],
        )
    plt.plot(
        0.1 ** (np.array(logr_list)),
        J * 0.01 ** (np.array(logr_list)),
        color="black",
        ls="--",
        label="obs. noise level",
    )  # Jr^2
    # plt.plot(0.1**(np.array(logr_list)), rho**2*np.ones_like(logr_list), color="black", ls="-", label="$ \\rho^2 $")  # plot attractor radius
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("r")
    plt.ylabel("worst error")  # limsupE[SE]
    plt.legend(bbox_to_anchor=(1, 1))
    fig.tight_layout()

    # save
    fig.savefig(f"{data_dir_r}/sup_se_r.pdf")
