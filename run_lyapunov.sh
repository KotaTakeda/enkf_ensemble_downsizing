# !/bin/sh
# Run Lyapunov analysis from the submodule
# Results are stored in data/case1_lyapunov and data/case3_lyapunov

set -e

# Ensure submodule is initialized
git submodule update --init --recursive

# Case F=8
echo "Running Lyapunov analysis for F=8"
python -m lyapunov.lorenz96.l96_lyapunov_np --data_dir=data/case1_lyapunov

# Case F=16
echo "Running Lyapunov analysis for F=16"
python -m lyapunov.lorenz96.l96_lyapunov_np --data_dir=data/case3_lyapunov

echo "All Lyapunov analyses completed."

# plot
python -m lyapunov.plot_lyapunov_exponents --data_dir=data/case1_lyapunov
python -m lyapunov.plot_lyapunov_exponents --data_dir=data/case3_lyapunov

echo "All Lyapunov exponents are visualized."


# copy figures
cp data/case1_lyapunov/lyapunov_exponents.pdf figures/fig1.pdf
cp data/case3_lyapunov/lyapunov_exponents.pdf figures/fig5.pdf
echo "All figures are copied."
