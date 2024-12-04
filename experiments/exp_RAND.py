# %% set up


import numpy as np
import numpy.random as rd
import pandas as pd
from env_config import EnvConfig
from env_testbed import Env
from dataset import Dataset
from mrt import MRT
import time
from joblib import Parallel, delayed
import json
from pathlib import Path


# %% parameters


## algorithm parameters
P0 = 0.5 # P(A = 1) in a random policy

ver = '1'
path = 'res_RAND' + ver + '/'
file_prefix = 'version' + ver + '_sigma2_' + str(0)+ '_reg_' + str(0)
file_res = path + file_prefix + '.txt'
params_env_path = 'params_env_V2/'
params_std_file = 'params_std_V2.json'

userid_all = np.loadtxt(params_env_path + 'user_ids.txt', dtype=int)
nitr = 100 # replications
n_jobs = 50 # parallel computing
seed = 2023


# %% experiments


## open and clean files
Path(path).mkdir(parents=True, exist_ok=True)
with open(file_res, 'w') as file:
    file.write('')

## initialization
rd.seed(seed)
seeds = rd.randint(low=1e3, high=1e6, size=nitr)


def experiment(itr):
    rd.seed(seeds[itr])
    dat_all = []
    save_cols = ['d', 'userid', 'R', 'R_imp', 'R0']
    for userid in userid_all:
        config = EnvConfig( # environment parameters
            userid, params_env_path, params_std_file,
        )
        env = Env(config)
        mrt = MRT(config, env)
        ## randomly generate actions w.p. P0
        dat = mrt.gen_data_fixed_prob(P0, config.W, config.D)

        ## baseline rewards
        dat_zero = mrt.gen_data_fixed_prob(0, config.W, config.D)
        dat.df['R0'] = dat_zero.df['R']
        ## save the results
        dat_all.append(dat.df.loc[:, save_cols])
    

    dat_all = pd.concat(dat_all)
    dat_all_mean = dat_all.groupby('d').mean()
    dat_all_std = dat_all.groupby('d').median()
    dat_all_std.columns = [x + '_median' for x in save_cols[1:]]
    out = pd.concat([dat_all_mean, dat_all_std], axis=1)
    out.to_csv(
        file_res, header=False, index=True, 
        mode='a', float_format='%.4f'
    )


# %% parallel computing


results = Parallel(n_jobs=n_jobs)(
    delayed(experiment)(itr) for itr in range(nitr)
)

