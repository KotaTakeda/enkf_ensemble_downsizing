# EnKF with Ensemble Downsizing

This repository contains the implementation of the **Ensemble Kalman Filter (EnKF)** with **ensemble downsizing**, developed for the paper:

> _Noise-scaled accuracy of the ensemble Kalman filter with an instability-based minimum ensemble size_,  
> Kota Takeda and Takemasa Miyoshi,
> under review.

[![DOI](https://zenodo.org/badge/913588982.svg)](https://doi.org/10.5281/zenodo.17319854)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KotaTakeda/enkf_ensemble_downsizing/v1.1.1?urlpath=%2Fdoc%2Ftree%2Ftest.ipynb)

## Overview

The purpose of this repository is to reproduce the EnKF numerical experiments in the above paper.  
The focus is on evaluating the minimum ensemble size $m^*$ required for asymptotic accuracy and verifying the theoretical estimate

$$
m^* = N_+ + 1,
$$

where $N_+$ is the number of positive Lyapunov exponents (LEs).

### Notes

- Cython implementation of the Lorenz 96 model (`lorenz96_cython`) is used for efficient numerical integration.
- An implementation for computing the Lyapunov exponents is available through a submodule.

## Installation

### Requirements

```sh
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Build Cython

```sh
cd lorenz96_cython
python setup.py build_ext --inplace
```

## Usage

Run `test.ipynb` or a script as follows.

### 1. Prepare a data directory

Create a directory for the experiment:

```sh
mkdir {data_dir_name}
```

Note: The directory `data/test` already exists with a config file `set_params.py`.

### 2. Create a parameter file

Inside `{data_dir_name}`, place a file named `set_params.py` specifying experiment parameters.

### 3. Run experiments

```sh
python3 main.py --data_dir={data_dir_name}
```

## Reproducing Figures in the Paper

### Generate Fig.1, 5

```sh
sh run_lyapunov.sh
```

### Generate Fig.2, 3, 4, 6, B1

```sh
sh run_experiments.sh
```

## Other information

### Repository organization

The repository is organized to allow reproduction of each figure in the manuscript.

```
root/
- data/
  - case1/          # F = 8, varying r (Fig.2)
    - r0/
    - r1/
    - ...
  - case2/          # varying N_spinup (Fig.3,4)
    - N0-r4/
    - N0-r4-acc/
    - N1-r4/
    - N1-r4-acc/
    - ...
  - case3/          # F = 16, varying r (Fig.6)
    - r0/
    - r1/
    - ...
  - case4/          # large r and small dt (Fig.B1)
    - r0-t3
  - case1_lyapunov/ # For the LE analysis (Fig.1)
  - case3_lyapunov/ # For the LE analysis (Fig.5)
  - test/           # For a light test in notebook
- figures/          # Stores generated figures (automatically generated)
- lyapunov/         # submodule to compute the LEs
- lorenz96_cython/  # C extension of the Lorenz 96 model
- main.py
- plot.py
- test.ipynb
- run_lyapunov.sh
- run_experiments.sh
- README.md
- ...
```

Each subfolder contains its own `set_params.py` and stores results in a consistent structure.
An example of saved filename format: `xa_ijk.npy` (`i`->`m`, `j`->`alpha`, `k`->`seed`).

### Parameters

The key parameters defined in `set_params.py`:

| Parameter (variable in code)             | Value(s)                        | Description                                 |
| ---------------------------------------- | ------------------------------- | ------------------------------------------- |
| $J$                                      | 40                              | Number of components in the Lorenz 96 model |
| $F$                                      | 8, 16                           | External force in the Lorenz 96 model       |
| $\Delta t$ (`dt`)                        | 0.01                            | Time step size for integration              |
| $N$                                      | 72,000                          | Total number of integration steps           |
| $m$ (`m_reduced_list`)                   | 12–18                           | Ensemble size after downsizing              |
| $\alpha$ (`alpha_list`)                  | 1.0–1.5                         | Inflation factor                            |
| $r$                                      | $10^0, 10^{-1}, \dots, 10^{-4}$ | Observation noise std.                      |
| $n_{obs}$ (`obs_per`)                    | 5 (F=8), 2 (F=16)               | Observation interval (integration steps)    |
| $N_{spinup}$                             | 720 (F=8), 1800 (F=16)          | Spin-up steps (assimilation steps)          |
| $(\omega_k)\_{k=1}^{n\_{seeds}}$ (`seeds`) | 0,1, ..., 9                     | Random seeds                                |

Others:

- $m_0$ is fixed as `m0=J+1`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
