import sys
sys.path.append('')
from argparse import ArgumentParser
from src.utils.utils import read_config
from src.methods import *
from src.data.data_gen import simulate_mi, simulate_cmi, simulate_cit
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--method', type=str, default='DINE')
    parser.add_argument('-t', '--test', type=str, default='cmi')
    args = parser.parse_args()
    method = args.method
    print(f'{method = }'.ljust(100, '='))

    config_path = 'experiments/exp_test/configs.yml'
    configs = read_config(config_path)
    print(f'{configs = }')
    method_params = configs['methods'].get(method, {})
    print('Method params:', method_params)
    
    method = eval(method)
    for test_name, test_params in configs['testcases'].items():
        print(test_name.ljust(100, '-'))
        print(f'\t{test_params = }')
        X, Y, Z, gt = eval(f'simulate_{args.test}')(**test_params)
        mi = method(X, Y, Z, **method_params)
        print(f'\tGT = {gt: .4f}')
        print(f'\tEST = {mi:.4f}')
        print(f'\tPAE = {np.abs(mi - gt) / gt * 100:.2f}%')