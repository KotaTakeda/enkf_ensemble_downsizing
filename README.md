# EnKF Ensemble Reduction


## Requirements
```sh
pip install git+https://github.com/KotaTakeda/da_py.git@v0.4.5
pip install -r requirements.txt
```

Developed by `Python 3.10.6`.

## Build
```sh
cd lorenz96_cython
python setup.py build_ext --inplace
```

## Computing Lyapunov exponents
https://github.com/KotaTakeda/lyapunov


## OSSE
Make dir `{data_dir_name}`.
```sh
mkdir {data_dir_name}
```

Make `set_params.py`, arrange parameters in it, and put it in `{data_dir_name}`.

Run `main.py`.
```sh
python3 main.py --data_dir={data_dir_name} --parallel={parallel_method_name}
```