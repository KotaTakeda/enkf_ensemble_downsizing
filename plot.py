import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize as vis
from util import load_params


def summarize_rm(data_dir_r, logr_list, target_m, error_type="sup_se"):
    """
    Summarize the error for different r and m_reduced.
    Args:
        data_dir_r (str): directory where the data is stored. e.g. "data/l96/r{logr}".
        logr_list (list of float): list of logr values.
        target_m (int): target m_reduced value.
        error_type (str): filename of the error. ["sup_se", "mean_se", "sup_rmse", "mean_rmse"].
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
            lw=1.5,
            ms=6,
        )
    ax.plot(
        0.1 ** (np.array(logr_list)),
        J * 0.01 ** (np.array(logr_list)),
        color="black",
        # ls="-",
        lw=0.5,
        label="Obs. noise scale",
    )  # Jr^2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$r$")
    ax.set_ylabel("error")
    ax.set_ylabel(_error_title(error_type))
    ax.set_title("Error vs. $r$ for different $m$")
    ax.legend(bbox_to_anchor=(1, 1))
    fig.tight_layout()

    return fig, ax


def _error_title(error_type):
    """Convert error_type string to a plot title."""
    title_map = {
        "sup_se": r"$\limsup_{n \rightarrow \infty} \mathbb{E}[\mathrm{SE}_n]$",
        "mean_se": r"$\operatorname{mean}_n \mathbb{E}[\mathrm{SE}_n]$",
        "sup_rmse": r"$\limsup_{n \rightarrow \infty} \mathbb{E}[\mathrm{RMSE}_n]$",
        "mean_rmse": r"$\operatorname{mean}_n \mathbb{E}[\mathrm{RMSE}_n]$",
    }
    return title_map.get(error_type, "Error")


# ========================
# Helper for time series plots
# ========================
import os
from util import npload


def plot_time_series(
    data_dir,
    target_m_list,
    target_alpha_list,
    plot_type="one sample",
    plot_metric="RMSE",
    plot_ylabel=True,
    plot_legend=True,
    k_seed=0,
    plot_per=1000,
    ylabel=None,
    ax=None,
    x_true_path=None,
    Xa_dict=None,
    title=None,
    plot_x0=False,
):
    """
    Helper to plot RMSE time series for given data_dir, m, alpha, etc.
    Args:
        data_dir: str
        target_m_list: list of int
        target_alpha_list: list of float
        plot_type: str ["one sample", "mean", ...]
        plot_metric: str ["RMSE", "SE"]
        plot_ylabel: bool
        plot_legend: bool
        k_seed: int
        plot_per: int, step for plotting
        ylabel: str, y-label
        ax: matplotlib axes or None
        x_true_path: path to x_true file (optional)
        Xa_dict: precomputed dictionary (optional)
        title: str, plot title
        plot_x0: bool, whether to plot RMSE for x0 (initial estimate)
    Returns:
        ax
    """
    

    # Load params
    set_params = load_params(data_dir)
    m_reduced_list = set_params.m_reduced_list
    alpha_list = set_params.alpha_list
    J = set_params.J
    seeds = set_params.seeds
    m0 = J + 1  # hard coding
    if target_alpha_list is None:
        target_alpha_list = alpha_list
    
    # Load x_true
    if x_true_path is None:
        x_true_file = os.path.join(data_dir, "x_true_l96.npy")
    else:
        x_true_file = x_true_path
    if not os.path.exists(x_true_file):
        raise FileNotFoundError(f"x_true file not found: {x_true_file}")
    x_true = npload(x_true_file)  # (T, J)
    
    # Try to get r from params
    try:
        r = set_params.r
    except Exception:
        # fallback: try to parse from data_dir
        import re

        m = re.search(r"r(\d+)", data_dir)
        r = float(m.group(1)) if m else 1.0
    
    # Time index
    t = np.arange(1, len(x_true)+1)
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # Plot observation noise level
    if plot_metric=="RMSE":
        ax.plot(
            t,
            r * np.ones_like(t),
            lw=1,
            c="black",
            ls="--",
            label="Observation noise level $ r $",
        )
    elif plot_metric=="SE":
        ax.plot(
            t,
            J * r**2 * np.ones_like(t),
            lw=1,
            c="black",
            ls="--",
            label="Observation noise level $ N_x r^2 $",
        )
    else:
        raise ValueError("Unknown plot_metric: {plot_metric} ")


    # Handle plot_type
    if plot_type == "one sample":
        seed_list = [k_seed] if k_seed in seeds else [seeds[0]]
    elif plot_type == "mean":
        seed_list = seeds
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    
    # Use default marker and color cycles
    markers = vis.get_marker_cycle()
    for i, m_reduced in enumerate(m_reduced_list):
        if len(target_m_list) > 0 and m_reduced not in target_m_list:
            continue
        marker = next(markers)
        colors = vis.get_color_palette(len(target_alpha_list), "viridis")
        for j, alpha in enumerate(alpha_list):
            if len(target_alpha_list) > 0 and alpha not in target_alpha_list:
                continue
            xa_allk = []
            for k, _ in enumerate(seed_list):
                # Load Xa for this (i, j, k)
                # Xa_dict is optional precomputed mapping, else try to load
                if Xa_dict is not None:
                    xa_dict = Xa_dict[i, j, k] # deprecated
                else:
                    # Try to load from convention: xa-m{m_reduced}-alpha{alpha}-seed{k}.npy
                    # This may need to be adapted to your file structure
                    savename_xa = os.path.join(data_dir, f"xa-{i}{j}{k}")
                    savename_xa_spinup = os.path.join(data_dir, f"xa_spinup-{i}{j}{k}")
                    savename_x0 = os.path.join(data_dir, f"x0-{i}{j}{k}")
                    xa_dict = {"xa": savename_xa, "xa_spinup": savename_xa_spinup, "x0": savename_x0}
                # Load Xa time series
                label = f"ETKF-{m_reduced} $\\alpha$={alpha}" # for plot legend
                xa = npload(xa_dict["xa"] + ".npy").mean(axis=1)  # (N, J)
                # If spinup data exists, load and prepend
                if os.path.exists(xa_dict["xa_spinup"] + ".npy"):
                    xa_spinup = npload(xa_dict["xa_spinup"] + ".npy")  # (N_spinup, m0, J)
                    if len(xa_spinup) > 0:
                        xa_spinup = xa_spinup.mean(axis=1)
                        xa = np.vstack([xa_spinup, xa])
                        label = f"ETKF-reduce({m0}$\\rightarrow${m_reduced}) $\\alpha$={alpha}"
                # Prepend x0 if needed
                if plot_x0 and os.path.exists(xa_dict["x0"] + ".npy"):
                    x0 = npload(xa_dict["x0"] + ".npy")  # (m0, J)
                    x0 = x0.mean(axis=0).reshape((1, -1))  # (1, J)
                    xa = np.vstack([x0, xa])
                
                xa_allk.append(xa)
            # 
            xa_allk = np.array(xa_allk)  # (len(k_list), N, J)
            
            # Align time series lengths
            if len(xa_allk[0]) == len(x_true):
                x_true_align = x_true
                t_align = t
            elif len(xa_allk[0]) == len(x_true) + 1:
                x_true_align = np.vstack([x_true[0], x_true])  # align time
                t_align = np.concatenate([[0], t])  # shift time index
            else:
                raise ValueError("Length mismatch between xa and x_true.")
            # Compute metric
            se = np.sum((x_true_align[None, :, :] - xa_allk)**2, axis=-1) # (len(k_list), N)
            if plot_metric=="RMSE":
                metric = np.mean(np.sqrt(se / J , axis=0)) # (N, )
            elif plot_metric=="SE":
                metric = np.mean(se, axis=0) # (N, )
            
            # Plot
            # FIXME: slice with plot_per before computing metric
            ax.plot(
                t_align[::plot_per],
                metric[::plot_per],
                label=label,
                lw=0.8,
                marker=marker,
                color=colors[j],
            )
    ax.set_xlabel("Obs. step $n$")
    if ylabel is None:
        ylabel = plot_metric
    if plot_ylabel:
        ax.set_ylabel(ylabel)
    if plot_legend:
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=8)
    ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)
    return ax


import numpy as np
import pyvista as pv
from IPython.display import Video

def animate_enkf_pyvista(x_true, x_esti, obs=None, 
                         t_info=None,
                         filename="enkf_vis.mp4", 
                         tail_len=20, 
                         colors=None,
                         ax_labels=('x', 'y', 'z')):
    """
    Fixed AttributeError by using coordinate positioning for text.
    """
    
    # 1. Setup Data & Colors
    T, m, _ = x_esti.shape
    x_mean = np.mean(x_esti, axis=1)

    style = {
        'true': 'blue',
        'mean': 'red',
        'ens':  'red',
        'obs':  'green'
    }
    if colors:
        style.update(colors)

    # 2. Calculate Global Bounds
    all_data = [x_true, x_esti.reshape(-1, 3)]
    if obs is not None:
        all_data.append(obs)
    
    concatenated = np.concatenate(all_data, axis=0)
    min_pt = np.min(concatenated, axis=0)
    max_pt = np.max(concatenated, axis=0)
    
    margin = (max_pt - min_pt) * 0.1
    bounds = [min_pt[0]-margin[0], max_pt[0]+margin[0],
              min_pt[1]-margin[1], max_pt[1]+margin[1],
              min_pt[2]-margin[2], max_pt[2]+margin[2]]

    # 3. Setup Plotter
    pv.set_plot_theme("document")
    pl = pv.Plotter(off_screen=True, window_size=[1024, 768])
    try:
        pl.open_movie(filename, framerate=30, codec='libx264')
    except Exception as e:
        print(f"Warning: libx264 not found. Falling back to default. Error: {e}")
        pl.open_movie(filename, framerate=30)
    bbox = pv.Box(bounds)
    pl.add_mesh(bbox, opacity=0.0, show_edges=False) 

    # --- Time Display Setup (Fixed) ---
    text_actor = None
    if t_info:
        t_start = t_info.get("t_start", 0.0)
        dt_vis = t_info.get("dt", 1.0)
        
        # Use tuple (x, y) coordinates (0 to 1) instead of 'upper_right'
        # (0.7, 0.9) places it roughly in the top-right corner
        text_actor = pl.add_text(
            f"Time: {t_start:.2f}", 
            position=(0.75, 0.9),  # <--- CHANGED HERE
            color='black', 
            font_size=12,
            font='arial'
        )

    # 4. Initialize Meshes
    line_true = pv.Spline(x_true[:2], 20) 
    pl.add_mesh(line_true, color=style['true'], line_width=4, label="True State")

    line_mean = pv.Spline(x_mean[:2], 20)
    pl.add_mesh(line_mean, color=style['mean'], line_width=4, label="Est (Mean)", opacity=0.7)

    n_show_ens = min(m, 20)
    ens_meshes = []
    for k in range(n_show_ens): 
        l = pv.Spline(x_esti[:2, k], 10)
        pl.add_mesh(l, color=style['ens'], line_width=1, opacity=0.5) 
        ens_meshes.append(l)

    points_obs = None
    if obs is not None:
        points_obs = pv.PolyData(obs[:1])
        pl.add_mesh(points_obs, color=style['obs'], point_size=15, 
                    render_points_as_spheres=True, label="Observation")

    # 5. Scene Settings
    pl.show_grid(
        color='black', 
        bounds=bounds,
        xtitle=ax_labels[0],
        ytitle=ax_labels[1],
        ztitle=ax_labels[2],
        font_size=12,
        bold=False
    )
    
    pl.add_legend(bcolor='white', size=(0.2, 0.2), face=None)
    
    pl.camera_position = 'xy'
    pl.camera.azimuth = 45
    pl.camera.elevation = 20
    pl.reset_camera()

    print(f"Processing animation: {filename} ...")

    # 6. Animation Loop
    for i in range(1, T):
        start = max(0, i - tail_len)
        end = i + 1
        
        if end - start < 2: continue

        # Update Time Text
        if text_actor:
            current_time = t_start + i * dt_vis
            # Now text_actor is a vtkTextActor which has SetInput
            text_actor.SetInput(f"Time: {current_time:.2f}")

        line_true.copy_from(pv.lines_from_points(x_true[start:end]))
        line_mean.copy_from(pv.lines_from_points(x_mean[start:end]))
        
        for k, mesh in enumerate(ens_meshes):
            mesh.copy_from(pv.lines_from_points(x_esti[start:end, k]))
            
        if obs is not None:
            points_obs.points = obs[i:i+1]
        
        pl.write_frame()

    pl.close()
    print("Animation saved.")
    
    return Video(filename, embed=True, html_attributes="controls loop autoplay")

# ========================
# Figure plotting functions
# ========================


def plot_fig2():
    """
    Plot Figure 2: summarize (r, m) for case1, annotate, and save.
    """
    print("Plot Figure 2: summarize (r, m) for case1, annotate, and save.")
    fig, ax = summarize_rm(
        "data/case1", logr_list=[0, 1, 2, 3, 4], target_m=14, error_type="sup_se"
    )
    arrowprops = dict(arrowstyle="->", edgecolor="black", facecolor="black")
    ax.annotate("", xy=(2e-4, 2e-6), xytext=(2e-4, 3e-3), arrowprops=arrowprops)
    ax.text(9e-5, 6e-3, "Obs. noise scale", fontdict=dict(fontsize=13, color="black", backgroundcolor="white"))
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig2.pdf", transparent=True)
    plt.close(fig)


def plot_fig6():
    """
    Plot Figure 6: summarize (r, m) for case3, annotate, and save.
    """
    print("Plot Figure 6: summarize (r, m) for case3, annotate, and save.")
    fig, ax = summarize_rm(
        "data/case3", logr_list=[0, 1, 2, 3, 4], target_m=16, error_type="sup_se"
    )
    arrowprops = dict(arrowstyle="->", edgecolor="black", facecolor="black")
    ax.annotate("", xy=(2e-4, 2e-6), xytext=(2e-4, 3e-3), arrowprops=arrowprops)
    ax.text(9e-5, 6e-3, "Obs. noise scale", fontdict=dict(fontsize=13, color="black", backgroundcolor="white"))
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig6.pdf", transparent=True)
    plt.close(fig)


def plot_fig3():
    """
    Plot Figure 3: time series for case2/N0-r4 and case2/N1-r4, stack vertically, add (a) and (b).
    """
    print("Plot Figure 3: time series for case2/N0-r4 and N1-r4.")
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for ax in axs:
        ax.set_ylim(1e-5, 10.0)
    # First subplot: case2/N1, m=14, all alphas
    plot_time_series(
        data_dir="data/case2/N1-r4",
        target_m_list=[14],
        target_alpha_list=None,  # all alphas
        plot_type="one sample",
        plot_ylabel=True,
        plot_legend=True,
        k_seed=0,
        ax=axs[0],
        title="(a) $N_{\\mathrm{spinup}}=720$",
        plot_x0=True,
    )
    # Second subplot: case2/N0, m=14, all alphas
    plot_time_series(
        data_dir="data/case2/N0-r4",
        target_m_list=[14],
        target_alpha_list=None,
        plot_type="one sample",
        plot_ylabel=True,
        plot_legend=True,
        k_seed=0,
        ax=axs[1],
        title="(b) $N_{\\mathrm{spinup}}=0$",
        plot_x0=True,
    )
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig3.pdf", transparent=True)
    plt.close(fig)



def plot_fig4():
    """
    Plot Figure 4: time series for case2/N0-r4-acc and N1-r4-acc, stack vertically, add (a) and (b).
    """
    print("Plot Figure 4: time series for case2/N0-r4-acc and N1-r4-acc.")
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for ax in axs:
        ax.set_ylim(1e-6, 10.0)
    # First subplot: case2/N1, m=14, all alphas
    plot_time_series(
        data_dir="data/case2/N1-r4-acc",
        target_m_list=[14],
        target_alpha_list=None,  # all alphas
        plot_type="one sample",
        plot_ylabel=True,
        plot_legend=True,
        k_seed=0,
        ax=axs[0],
        title="(a) $N_{\\mathrm{spinup}}=720$",
        plot_x0=True,
        plot_per=100
    )
    # Second subplot: case2/N0, m=14, all alphas
    plot_time_series(
        data_dir="data/case2/N0-r4-acc",
        target_m_list=[14],
        target_alpha_list=None,
        plot_type="one sample",
        plot_ylabel=True,
        plot_legend=True,
        k_seed=0,
        ax=axs[1],
        title="(b) $N_{\\mathrm{spinup}}=0$",
        plot_x0=True,
        plot_per=100
    )
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig4.pdf", transparent=True)
    plt.close(fig)


# def plot_fig7():
#     """ 
#     Plot Figure 7: time series for case4/r0-t3, m=14, RMSE metric.
#     """
#     print("Plot Figure 7: time series for case4/r0-t3, m=14, RMSE metric.")
#     fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
#     plot_time_series(
#         data_dir="data/case4/r0-t3",
#         target_m_list=[14],
#         target_alpha_list=None,
#         plot_type="mean",
#         plot_ylabel=True,
#         plot_legend=True,
#         ax=ax,
#         title="Expectation of RMSE",
#         plot_per=5000,
#     )
#     fig.tight_layout()
#     os.makedirs("figures", exist_ok=True)
#     fig.savefig("figures/fig7.pdf", transparent=True)
#     plt.close(fig)

def plot_fig7():
    """ 
    Plot Figure 7: time series for case4/r0-t3, m=14, SE metric.
    """
    print("Plot Figure 7: time series for case4/r0-t3, m=14, SE metric.")
    data_dir = "data/case4/r0-t3"
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
    plot_time_series(
        data_dir=data_dir,
        target_m_list=[14],
        target_alpha_list=None,
        plot_type="mean",
        plot_metric="SE",
        plot_ylabel=True,
        plot_legend=True,
        ax=ax,
        title="Expectation of SE",
        plot_per=5000,
    )
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig7.pdf", transparent=True)
    plt.close(fig)


# ========================
# Main
# ========================

if __name__ == "__main__":
    plot_fig2()
    # plot_fig3()
    # plot_fig4()
    plot_fig6()
    # plot_fig7()