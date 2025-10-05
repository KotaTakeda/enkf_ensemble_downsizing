# J = 40
# F = 8
# dt = 0.01
# N = 360 * 20 * 11  # Nature run period
# obs_per = 5
# Nt = 360 * 20 * 10 // obs_per  # OSSE period
# T_inf = Nt // 2
# r = 1.0  # Std. of obs. noise
# N_spinup = 180 * 20 // obs_per  # Ensemble spin-up
# m_reduced_list = [12, 13, 14, 15, 16, 17, 18]
# alpha_list = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
# seeds = [n for n in range(10)]

# light test
J = 40
F = 8
dt = 0.01
N = 180 * 20 * 1  # Nature run period
obs_per = 5
T_inf = N // obs_per // 2
r = 1e-2  # Std. of obs. noise
N_spinup = 45 * 20 // obs_per  # Ensemble spin-up
m_reduced_list = [13, 14, 15]
alpha_list = [1.1]
seeds = [n for n in range(1)]
