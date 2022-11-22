import sys

sys.path.append('')
from argparse import ArgumentParser
from itertools import product
from multiprocessing import Pool

import pandas as pd
from src.methods import methods_dict
from src.data.data_gen import simulate_cit
from src.utils.utils import Timer, read_config
from torch.cuda import device_count
from tqdm import tqdm

def process(task):
    task_id, (dataset, (func, params)) = task
    X, Y, Z = dataset['data']
    indep = dataset['indep']
    d = dataset['d']

    gpus = [task_id % max(device_count(), 1)]
    with Timer() as t:
        p_value = func(X, Y, Z, gpus=gpus, **params)
    res = dict(
        d=d,
        Method=func.__name__,
        P_value=p_value,
        GT=indep,
        Time=t.elapsed
    )
    return res

if __name__ == '__main__':
    # CONFIG------------------------------------------------------------
    parser = ArgumentParser()
    parser.add_argument('--methods', type=str, nargs='+', default=['DINE_CIT'])
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    methods = args.methods

    EXP_NUM = 'exp_cit'
    config_path = f'experiments/{EXP_NUM}/configs.yml'
    configs = read_config(config_path)
    method_params = configs.get('methods', {})
    for method in methods:
        method_params[method] = method_params.get(method, {})
        print(f'{method}: {method_params[method]}')

    data_config = configs['data']

    T = data_config['T']
    N = data_config['N']
    alpha = data_config['alpha']

    # DATA GEN----------------------------------------------------------
    datasets = []
    res = []
    for d in data_config['d']:
        for i in range(T):
            X, Y, Z, indep = simulate_cit(N, d=1, dz=d, indep=i < T // 2, random_state=i)
            datasets.append(dict(
                data=(X, Y, Z),
                d=d,
                indep=indep,
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