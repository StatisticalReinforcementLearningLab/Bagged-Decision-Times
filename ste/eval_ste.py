# %% set up


import numpy as np
import numpy.random as rd
import pandas as pd
from env_config import EnvConfig
from env_testbed import Env
from dataset import Dataset
from mrt import MRT
from pathlib import Path
import d3rlpy
import sys
jobid = int(sys.argv[1])


# %% specification


params_env_path = 'params_env_V2/'
params_std_file = 'params_std_V2.json'
userid_all = np.loadtxt(params_env_path + 'user_ids.txt', dtype=int)

scalar_list = np.arange(6)
seed_list = np.arange(len(scalar_list)) * 2024
N_test = 500

array_idx = np.unravel_index(jobid, (len(scalar_list), len(userid_all)))
scalar = scalar_list[array_idx[0]]
userid = userid_all[array_idx[1]]
seed = seed_list[array_idx[0]]

exp = '1'
logger_dir = 'd3rlpy_logs/exp_' + str(exp)
tensorboard_dir = 'tensorboard_logs/exp_' + str(exp)
experiment_name = 'job' + str(scalar) + '_user' + str(userid)
model_dir = logger_dir + '/' + experiment_name + '_model.d3'
path = 'results' + exp + '/'
filepath = path + 'res' + exp + '_' + str(scalar) + '_' + str(userid) + '.txt'


# %% calculate sum of rewards


dqn = d3rlpy.load_learnable(model_dir)

## calculate the sum of rewards
rd.seed(seed)
out = np.zeros((N_test, 4))
for n in range(N_test):
    out[n, 0] = scalar
    out[n, 1] = userid

    ## define the env
    config = EnvConfig(userid, params_env_path, params_std_file, scalar)
    env = Env(config, noise='random')
    mrt = MRT(config, env)
    K = config.K
    W = config.W
    D = config.D
    nweek = config.nweek
    
    ## zero policy
    dat_zero = mrt.gen_data_fixed_prob(0, W, D)
    out[n, 2] = np.sum(dat_zero.df['R'])

    ## optimal stationary policy
    dat = Dataset(config.userid, K, D)

    E0 = env.E0
    R0 = env.R0
    dat.df.loc[0, 'E'] = E0
    dat.df.loc[0, 'R'] = R0
    dat.df.loc[0, 'R_obs'] = R0
    dat.df.loc[0, 'R_imp'] = R0
    for d in range(0, D):
        Edm1 = np.array([dat.df.loc[d, 'E']])
        Rdm1 = np.array([dat.df.loc[d, 'R']])
        Cd = np.zeros((K, config.dC))
        Pd = np.zeros((K, config.dP))
        Ad = np.zeros((K, config.dA)).astype(int)
        Md = np.zeros((K, config.dM))
        for k in range(0, K):
            Cd[k] = env.gen_Ch(d, k)
            ## construct the state vector
            ## state indicators
            state_mask_d = ((d % W) == np.arange(1, W)).astype(int)
            state_mask_h = ((k % K) == np.arange(1, K)).astype(int)
            ## state C
            state_C = Cd[k]
            ## state M
            state_M = dat.df.loc[1:(d + 1), dat.col_M].values.reshape(-1)
            state_M = np.hstack([np.zeros(W * K), state_M])
            state_M = state_M[(d * K + k + 1):(W * K + d * K + k)]
            state_M = state_M[::-1] ## the latest observation is the first dimension
            state_M[((d * K + k) % (W * K)):] = 0 ## set useless M to zero
            ## state A
            state_A = dat.df.loc[1:(d + 1), dat.col_A].values.reshape(-1)
            state_A = np.hstack([np.zeros(W * K), state_A])
            state_A = state_A[(d * K + k + 1):(W * K + d * K + k)]
            state_A = state_A[::-1] ## the latest observation is the first dimension
            state_A[((d * K + k) % (W * K)):] = 0 ## set useless M to zero
            ## state R
            prev_R = dat.df.loc[:d, 'R_obs']
            state_R = dat.df.loc[prev_R.last_valid_index(), 'R_obs']
            ## state E
            state_E = dat.df.loc[0:d, 'E']
            state_E = np.hstack([np.zeros(W), state_E])
            state_E = state_E[(d + 1):(W + d + 1)]
            state_E = state_E[::-1]
            state_E[((d % W) + 1):] = 0
            ## combine states
            states = np.hstack([
                state_mask_d, state_mask_h, state_C, state_M, state_A, state_R, state_E
            ]).reshape(1, -1)
            ## greedy action by DQN
            Pd[k] = 1 ## not the true prob
            Ad[k] = dqn.predict(states).item()
            Md[k] = env.gen_Mh(Edm1, Rdm1, Cd[k], Ad[k], d, k)
            ## save at each time k
            dat.df.loc[d + 1, dat.col_C[k]] = Cd[k]
            dat.df.loc[d + 1, dat.col_P[k]] = Pd[k]
            dat.df.loc[d + 1, dat.col_A[k]] = Ad[k]
            dat.df.loc[d + 1, dat.col_M[k]] = Md[k]
        ## observe at the end of bag
        Ed = env.gen_Ed(Ad, Edm1, d)
        Rd = env.gen_Rd(Md, Ed, Rdm1, d)
        Od = env.gen_Od(Rdm1, d)
        if (d + 1) % W == 0:
            Rd_obs = Rd.copy()
        else:
            Rd_obs = np.array([np.nan])
        Rd_imp = Rd_obs.copy()
        dat.df.loc[d + 1, dat.col_C] = Cd.reshape(-1)
        dat.df.loc[d + 1, dat.col_P] = Pd.reshape(-1)
        dat.df.loc[d + 1, dat.col_A] = Ad.reshape(-1)
        dat.df.loc[d + 1, dat.col_M] = Md.reshape(-1)
        dat.df.loc[d + 1, 'E'] = Ed.reshape(-1)
        dat.df.loc[d + 1, 'R'] = Rd.reshape(-1)
        dat.df.loc[d + 1, 'R_obs'] = Rd_obs.reshape(-1)
        dat.df.loc[d + 1, 'R_imp'] = Rd_imp.reshape(-1)
        dat.df.loc[d + 1, 'O'] = Od.reshape(-1)
        dat.df.loc[d + 1, 'regret'] = 0
    out[n, 3] = np.sum(dat.df['R'])


Path(path).mkdir(parents=True, exist_ok=True)
with open(filepath, 'a') as file:
    np.savetxt(file, out, fmt='%.2f')

