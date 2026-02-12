J = 40
F = 8
dt = 0.01
N = 360 * 20 * 100  # Nature run period
obs_per = 5
T_inf = N // obs_per // 2
r = 1e-4  # Std. of obs. noise
N_spinup = 720  # Ensemble spin-up
m_reduced_list = [14]
alpha_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
seeds = [0]  # random seeds

