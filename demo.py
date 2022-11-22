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