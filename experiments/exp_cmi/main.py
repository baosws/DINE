import random
import sys

sys.path.append('')
from argparse import ArgumentParser
from itertools import product
from multiprocessing import Pool

import pandas as pd
import yaml
from src.data.data_gen import simulate_cmi
from src.methods import methods_dict
from src.utils.utils import Timer, read_config
from torch.cuda import device_count
from tqdm import tqdm, trange


def process(task):
    task_id, (dataset, (func, params)) = task
    X, Y, Z = dataset['data']
    rho = dataset['rho']
    GT = dataset['GT']
    N = dataset['N']
    d = dataset['d']
    random_state= dataset['random_state']

    gpus = [task_id % max(device_count(), 1)]
    res = dict(
        N=N,
        d=d,
        Method=func.__name__,
        GT=GT,
        rho=rho,
        random_state=random_state
    )
    try:
        with Timer() as t:
            res['Pred'] = func(X, Y, Z, gpus=gpus, **params)
        res['Time'] = t.elapsed
    except Exception as e:
        print(e)
        print('Error', task_id, res)
    
    return res

if __name__ == '__main__':
    # CONFIG------------------------------------------------------------
    parser = ArgumentParser()
    parser.add_argument('--methods', type=str, nargs='+', default=['DINE'])
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    methods = args.methods

    EXP_NUM = 'exp_cmi'
    config_path = f'experiments/{EXP_NUM}/configs.yml'
    configs = read_config(config_path)
    common_params = configs.get('common', {})
    print(f'{common_params = }')
    method_params = configs.get('methods', {})
    for method in methods:
        method_params[method] = method_params.get(method, {})
        print(f'{method}: {method_params[method]}')

    data_config = configs['data']
    T = data_config['T']

    # DATA GEN----------------------------------------------------------
    datasets = []
    res = []
    for rho in data_config['rho']:
        for d in data_config['d']:
            for N in data_config['N']:
                for i in range(T):
                    X, Y, Z, gt = simulate_cmi(N=N, d=d, dz=d, rho=rho, random_state=i)
                    datasets.append(dict(
                        data=(X, Y, Z),
                        N=N,
                        d=d,
                        rho=rho,
                        GT=gt,
                        random_state=i
                    ))

    # RUN---------------------------------------------------------------
    tasks = list(enumerate(product(datasets, [(methods_dict[method], method_params[method]) for method in methods])))
    if args.n_jobs > 1:
        with Pool(args.n_jobs) as p:
            res = list(tqdm(p.imap_unordered(process, tasks, chunksize=8), total=len(tasks)))
    else:
        res = list(map(process, tqdm(tasks)))

    # VISUALIZE---------------------------------------------------------
    df = pd.DataFrame(res)
    path = f'experiments/{EXP_NUM}/results/result_{"-".join(methods)}.csv'
    df.to_csv(path)