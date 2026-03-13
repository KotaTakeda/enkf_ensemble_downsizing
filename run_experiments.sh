#!/bin/sh
# Run ensemble downsizing experiments for the paper
# Each case corresponds to a figure in the manuscript
# Results are saved under data/case*/ and figures/ directories

set -e

# Case 1: F=8, varying r
for r in 0 1 2 3 4; do
    echo "Running Case 1 with r=10^$((-r))"
    python main.py --data_dir data/case1/r${r}
done

# Case 2: varying N_spinup
for N in 0 1; do
    echo "Running Case 2 with N_spinup=$((720*N))"
    python main.py --data_dir data/case2/N${N}
done
python main.py --data_dir data/case2/N0-r4

# Case2: additional run for N1-r4 with m=14
echo "Running Case 2 additional with m=14, N_spinup=720 and r=1e-4"
python main.py --data_dir data/case2/N1-r4

# Case2: additional run with accuract-initialization
echo "Running Case 2 additional with m=14, r=1e-4, accuract-initialization"
for N in 0 1; do
    echo "Running Case 2 with N_spinup=$((720*N))"
    python main.py --data_dir data/case2/N${N}-r4-acc
done

# Case 3: F=16, varying r
for r in 0 1 2 3 4; do
    echo "Running Case 3 with r=10^$((-r))"
    python main.py --data_dir data/case3/r${r}
done

echo "All ETKF experiments completed."

# plot
python plot.py