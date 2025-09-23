# EnKF Ensemble Reduction

This repository contains the implementation of the **Ensemble Kalman Filter (EnKF)** with **ensemble downsizing**, developed for the paper:

> *Quantifying the minimum ensemble size for asymptotic accuracy of the ensemble Kalman filter using the degrees of instability*,  
> Kota Takeda and Takemasa Miyoshi

---

## Overview

The purpose of this repository is to reproduce the EnKF numerical experiments in the above paper.  
The focus is on evaluating the minimum ensemble size required for asymptotic accuracy and verifying the theoretical estimate
<br>
<center>
\[
m^* = N_+ + 1,
\]
</center>
where \(N_+\) is the number of positive Lyapunov exponents.

- **Cython implementation** of the Lorenz 96 model (`lorenz96_cython`) is used for efficient numerical integration.  
- **Lyapunov exponent analysis** is not performed here. That part is handled separately in the [`lyapunov`](https://github.com/KotaTakeda/lyapunov) repository.

---

## Installation

### Requirements
```sh
pip install -r requirements.txt
```

Developed with Python 3.10.6.

### Build Cython
```sh
cd lorenz96_cython
python setup.py build_ext --inplace
```

---

## Usage

### 1. Prepare data directory
Create a directory for the experiment:
```sh
mkdir {data_dir_name}
```

### 2. Create parameter file
Inside `{data_dir_name}`, place a file named `set_params.py` specifying experiment parameters.  
See `set_params_example.py` for reference.

### 3. Run experiment
```sh
python3 main.py --data_dir={data_dir_name} --parallel={parallel_method_name}
```

### 4. Generate figures
```sh
python3 plot.py
```
---

## Reproducing Figures in the Paper

The repository is organized to allow reproduction of each figure in the manuscript.

```
root/
- data/
  - case1/        # F = 8, varying r (Fig.2)
    - r0/
    - r1/
    - ...
  - case2/        # varying N_spinup (Fig.3,4)
    - N0/
    - N0-r4/
    - N1/
  - case3/        # F = 16, varying r (Fig.6)
    - r0/
    - r1/
    - ...
  - case1_lyapunov/ # For external LE analysis (Fig.1)
  - case3_lyapunov/ # For external LE analysis (Fig.5)
- figures/        # Stores generated figures
```

Each subfolder contains its own `set_params.py` and stores results in a consistent structure.
An example of saved filename format: `xa_ijk.npy` `i`->`m`, `j`->`alpha`, `k`->`seed`.

---

## Parameters

The key parameters defined in `set_params.py`:

| Parameter (variable in code)    | Value(s) | Description |
|----------------|----------|-------------|
| $J$            | 40   | Number of components in the Lorenz 96 model |
| $F$            | 8, 16   | External force in the Lorenz 96 model |
| $\Delta t$ (`dt`)    | 0.01 | Time step size for integration |
| $N$            | 72,000 | Total number of integration steps |
| $m$ (`m_reduced_list`) | 12–18  | Ensemble size after downsizing |
| $\alpha$ (`alpha_list`)      | 1.0–1.5 | Inflation factor |
| $r$            | $10^0, 10^{-1}, \dots, 10^{-4}$ | Obs. noise std. dev. |
| $n_{obs}$ (`obs_per`)  | 5 (F=8), 2 (F=16) | Observation interval |
| $N_{spinup}$   | 720 (F=8), 1800 (F=16) | Spin-up steps |
| (`seeds`)      | 0,1, ..., 9 | Random seeds |

- $m_0$ is automatically defined as `m0=J+1`.

---

## Notes

- This repository requires a **Cython build**.  
- Only the Lorenz 96 right-hand side is used; the Jacobian is not required.  
- Lyapunov exponents are computed in the [separate repository](https://github.com/KotaTakeda/lyapunov).

---

## License

MIT License.

