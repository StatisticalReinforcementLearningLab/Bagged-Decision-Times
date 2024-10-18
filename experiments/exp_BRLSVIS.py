# %% set up


import numpy as np
import numpy.random as rd
import pandas as pd
from env_config import EnvConfig
from env_testbed import Env
from dataset import Dataset
from mrt import MRT
from artificial_data import ArtificialData
from BRLSVIS import BRLSVI
import time
from joblib import Parallel, delayed
import json
from pathlib import Path
import sys
jobid = int(sys.argv[1])


# %% parameters


sigma2list = [0.001, 0.002, 0.005, 0.01]
reglist = [2, 5, 10, 20, 50]
array_idx = np.unravel_index(jobid, (len(sigma2list), len(reglist)))
J = 0 # number of artifial episodes
sigma2 = sigma2list[array_idx[0]]
reg = reglist[array_idx[1]]
lam_beta = reg / sigma2 ## lamlist[array_idx[1]] # tuning parameter for beta
lam_theta = 0.1 # tuning parameter for theta
J_init = 100000 # number of artificial episodes for initialization

## algorithm parameters
L = 1 # number of days in a trajectory
B = 1 # number of bootstrap samples
P0 = 0.5 # initial P(A = 1)

ver = '1'
path = 'res_BRLSVIS' + ver + '/'
file_prefix = 'version' + ver + '_sigma2_' + str(sigma2)+ '_reg_' + str(reg)
file_res = path + file_prefix + '.txt'
params_env_path = 'params_env_V2/'
params_prior_file = 'params_prior_V3.json'
params_std_file = 'params_std_V2.json'

userid_all = np.loadtxt(params_env_path + 'user_ids.txt', dtype=int)
nitr = 100 # replications
n_jobs = 50 # parallel computing
seed = 2023


# %% helper


def create_art(config, L):
    with open(params_prior_file, 'r') as file:
        art_params = json.load(file)
    for key, value in art_params.items():
        art_params[key] = np.array(value)
    art_params['prob'] = 0.5 # action assignment probability
    ## shift the prior simultaneously with the testbed variants, assuming that 
    ## the previous trial has approximately the same STE as the new trial
    art_params['theta_R_mean'][1:6] += 0
    art_params['theta_R_mean'][6] += 0
    art_params['theta_E_mean'][2:7] -= 0
    return ArtificialData(art_params, config, L)


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
        art = create_art(config, L)

        dat = Dataset(config.userid, config.K, config.D)
        elapse = []

        E0 = env.E0
        R0 = env.R0
        dat.df.loc[0, 'E'] = E0
        dat.df.loc[0, 'R'] = R0
        dat.df.loc[0, 'R_obs'] = R0
        dat.df.loc[0, 'R_imp'] = R0

        ## warm up
        for d in range(0, config.D_warm):  # d = # of days in the history data
            start = time.time()
            Edm1 = np.array([dat.df.loc[d, 'E']])
            Rdm1 = np.array([dat.df.loc[d, 'R']])
            Cd = np.zeros((config.K, config.dC))
            Pd = np.zeros((config.K, config.dP))
            Ad = np.zeros((config.K, config.dA)).astype(int)
            Md = np.zeros((config.K, config.dM))
            ## assign treatments and observe new data
            for k in range(0, config.K):
                Cd[k] = env.gen_Ch(d, k)
                Pd[k] = P0
                Ad[k] = rd.choice([0, 1], size=1, p=[1 - Pd[k, 0], Pd[k, 0]])
                Md[k] = env.gen_Mh(Edm1, Rdm1, Cd[k], Ad[k], d, k)
            Ed = env.gen_Ed(Ad, Edm1, d)
            Rd = env.gen_Rd(Md, Ed, Rdm1, d)
            Od = env.gen_Od(Rdm1, d)
            if True:
                Rd_obs = Rd.copy()
            else:
                Rd_obs = np.array([np.nan])
            Rd_imp = Rd_obs.copy()
            ## save the new data
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
            elapse.append(time.time() - start)
            

        ## initialize BRLSVI
        brlsvi = BRLSVI(config, lam_beta, sigma2, standardize=False)
        # opt_beta = brlsvi.get_opt_Q(env, art)
        betas = np.zeros((brlsvi.dX, B))

        ## bootstrapped stationary LSVI
        for d in range(config.D_warm, config.D):
            start = time.time()
            b = 0
            comb_dat = art.combine_dataset(dat, d, J)
            ## BRLSVI
            beta = brlsvi.get_Q(betas[:, [b]], comb_dat, d + J)
            betas[:, [b]] = beta.copy()
            
            Rdm1_imp = dat.df.loc[d, 'R_obs']
            Edm1 = np.array([dat.df.loc[d, 'E']])
            Rdm1 = np.array([dat.df.loc[d, 'R']])
            Cd = np.zeros((config.K, config.dC))
            Pd = np.zeros((config.K, config.dP))
            Ad = np.zeros((config.K, config.dA)).astype(int)
            Md = np.zeros((config.K, config.dM))
            ## assign treatments and observe new data
            for k in range(0, config.K):
                Cd[k] = env.gen_Ch(d, k)
                Ad[k] = brlsvi.choose_A(betas, Edm1, Rdm1_imp, Cd, Ad, Md, d, k)
                Md[k] = env.gen_Mh(Edm1, Rdm1, Cd[k], Ad[k], d, k)
            Ed = env.gen_Ed(Ad, Edm1, d)
            Rd = env.gen_Rd(Md, Ed, Rdm1, d)
            Od = env.gen_Od(Rdm1, d)
            if True:
                Rd_obs = Rd.copy()
            else:
                Rd_obs = np.array([np.nan])
            ## save the new data
            dat.df.loc[d + 1, dat.col_C] = Cd.reshape(-1)
            dat.df.loc[d + 1, dat.col_P] = Pd.reshape(-1)
            dat.df.loc[d + 1, dat.col_A] = Ad.reshape(-1)
            dat.df.loc[d + 1, dat.col_M] = Md.reshape(-1)
            dat.df.loc[d + 1, 'E'] = Ed.reshape(-1)
            dat.df.loc[d + 1, 'R'] = Rd.reshape(-1)
            dat.df.loc[d + 1, 'R_obs'] = Rd_obs.reshape(-1)
            dat.df.loc[d + 1, 'O'] = Od.reshape(-1)
            elapse.append(time.time() - start)

        ## baseline rewards
        mrt = MRT(config, env)
        dat_zero = mrt.gen_data_fixed_prob(0, config.W, config.D)
        dat.df['R0'] = dat_zero.df['R']
        ## save the results
        dat_all.append(dat.df.loc[:, save_cols])
    

    dat_all = pd.concat(dat_all)
    dat_all_mean = dat_all.groupby('d').mean()
    dat_all_std = dat_all.groupby('d').std()
    dat_all_std.columns = [x + '_std' for x in save_cols[1:]]
    out = pd.concat([dat_all_mean, dat_all_std], axis=1)
    out.to_csv(
        file_res, header=False, index=True, 
        mode='a', float_format='%.4f'
    )


# %% parallel computing


results = Parallel(n_jobs=n_jobs)(
    delayed(experiment)(itr) for itr in range(nitr)
)

