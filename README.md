# Diffeomorphic Information Neural Estimation (DINE)
This is the implementation of our paper: Bao Duong and Thin Nguyen. [Diffeomorphic Information Neural Estimation](https://arxiv.org/abs/2211.10856). In AAAI Conference on Artificial Intelligence (AAAI 2023).

## Dependencies

```bash
conda create -n dine python=3.8
conda activate dine
conda install pytorch -c pytorch
conda install numpy pandas seaborn matplotlib scikit-learn scipy
pip install pytorch-lightning==1.5.3 causal-learn==0.1.3.0 CCIT==0.4 tf-slim tensorflow
```
__Note__: `causal-learn` is for `KCIT`, `CCIT` is for `CCIT`, `tensorflow` is for `CCMI` & `MIND`, `tensorflow` & `tf-slim` are for `CCMI`. Please comment out or remove the lines related to these methods from `src/methods/__init__.py` if you don't want to use them.

## Demo

```python
from src.data.data_gen import simulate_mi, simulate_cmi, simulate_cit
from src.methods import DINE, DINE_CIT

N = 1000 # sample size
d = 5    # dimensionality of X and Y
dz = 5   # dimensionality of Z

print('-' * 30)
print('Mutual Information (MI) estimation')
X, Y, Z, mi = simulate_mi(N=N, d=d, rho=0.8, random_state=0) # here Z is empty
est = DINE(X=X, Y=Y, Z=Z, random_state=0)
print(f'Ground truth: {mi:.4f}')      # 2.5541
print(f'Estimation:   {est:.4f}\n\n') # 2.5777

print('-' * 30)
print('Conditional Mutual Information (CMI) estimation')
X, Y, Z, cmi = simulate_cmi(N=N, d=d, dz=dz, rho=0.5, random_state=1)
est = DINE(X=X, Y=Y, Z=Z, random_state=0)
print(f'Ground truth: {cmi:.4f}')      # 0.7192
print(f'Estimation:   {est:.4f}\n\n')  # 0.7168

print('-' * 30)
print('Conditional Independence testing')
X, Y, Z, indep = simulate_cit(N=N, d=1, dz=dz, indep=True, random_state=1)
p_value = DINE_CIT(X=X, Y=Y, Z=Z, random_state=0)
print(f'\tConditional Independent: {p_value = :.4f}') # 0.2500

X, Y, Z, indep = simulate_cit(N=N, d=1, dz=dz, indep=False, random_state=1)
p_value = DINE_CIT(X=X, Y=Y, Z=Z, random_state=0)
print(f'\tConditional Dependent:   {p_value = :.4f}') # 0.0000
```

## Running experiments

For example, to run the "Conditional Mutual Information" experiment (Figure 2 in the paper):
```
python experiments/exp_cmi/main.py --methods DINE KSG CCMI --n_jobs=8
```
where available `methods` are DINE, MINE, MIND, KSG, CCMI, KCIT, DINE_CIT and `n_jobs` is the number of parallel jobs to run.

Modifiable configurations are stored in `experiments/exp_*/config/`, and result dataframes are stored in `experiments/exp_*/results/` after the command is finished.

## Citation

If you find our code helpful, please cite us as:
```
@article{duong2022diffeomorphic,
  title={Diffeomorphic Information Neural Estimation},
  author={Duong, Bao and Nguyen, Thin},
  journal={arXiv preprint arXiv:2211.10856},
  year={2022}
}
```
