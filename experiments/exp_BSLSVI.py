# %% set up


import numpy as np
import numpy.random as rd
import pandas as pd
from env_config import EnvConfig
from env_testbed import Env
from dataset import Dataset
from mrt import MRT
from artificial_data import ArtificialData
from predictor import Predictor
from SLSVI import SLSVI
import time
from joblib import Parallel, delayed
import json
from pathlib import Path
import sys
jobid = int(sys.argv[1])


# %% parameters


Jlist = [0, 50, 100, 200]
lamlist = [5, 10, 20]
array_idx = np.unravel_index(jobid, (len(Jlist), len(lamlist)))
J = Jlist[array_idx[0]] # number of artifial episodes
lam_beta = lamlist[array_idx[1]] # tuning parameter for beta
lam_theta = 0.1 # tuning parameter for theta
J_init = 100000 # number of artificial episodes for initialization

## algorithm parameters
L = 1 # number of days in a trajectory
B = 1 # number of bootstrap samples
P0 = 0.3 # initial P(A = 1)

ver = '1'
path = 'res_BSLSVI' + ver + '/'
file_prefix = 'version' + ver + '_J' + str(J) + '_lam' + str(lam_beta)
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

        ## initial value for parameters
        est_theta_R = art.theta_R_mean.reshape(-1)
        est_theta_O = art.theta_O_mean.reshape(-1)
        est_theta_M = art.theta_M_mean.reshape(-1)
        ## the indexes of parameters in the vector to be optimized
        d_theta_R = art.theta_R_mean.shape[0]
        d_theta_O = art.theta_O_mean.shape[0]
        d_theta_M = art.theta_M_mean.shape[0]
        idx_theta_R = np.arange(0, d_theta_R)
        idx_theta_O = np.arange(idx_theta_R[-1] + 1, idx_theta_R[-1] + 1 + d_theta_O)
        idx_theta_M = np.arange(idx_theta_O[-1] + 1, idx_theta_O[-1] + 1 + d_theta_M)
        idxes_theta = [idx_theta_R, idx_theta_O, idx_theta_M]
        sigma2s = [art.sigma2_R, art.sigma2_O, art.sigma2_M]
        ## initialize the predictor
        pred = Predictor(config.K, sigma2s, idxes_theta, lam=lam_theta)

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
            if (d + 1) % config.W == 0:
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
            

        ## initialize SLSVI
        lsvi = SLSVI(config, standardize=False)
        # opt_beta = lsvi.get_opt_Q(env, art)
        betas = np.zeros((lsvi.dX, B))
        for b in range(B):
            comb_dat = art.combine_dataset(dat, 0, J_init)
            beta = lsvi.initialize(comb_dat, J_init, lam_beta)
            betas[:, [b]] = beta.copy()

        ## bootstrapped stationary LSVI
        for d in range(config.D_warm, config.D):
            start = time.time()
            Rdm1_imp_B = []
            for b in range(B):
                comb_dat = art.combine_dataset(dat, d, J)
                ## predict missing rewards
                dat_RR_obs = dat.df.loc[0:d, 'R_obs'].values.copy()
                init_theta = np.hstack([est_theta_R, est_theta_O, est_theta_M])
                MLEs = pred.maximize_loglik(comb_dat, dat_RR_obs, init_theta)
                est_theta_R, est_theta_O, est_theta_M, est_Ru0, est_Ru = MLEs
                ## fill the RR matrix with predicted rewards
                comb_dat['comb_RRdm1'][np.isnan(comb_dat['comb_RRdm1'])] = est_Ru0.copy()
                comb_dat['comb_RR'][np.isnan(comb_dat['comb_RR'])] = est_Ru.copy()
                ## the predicted reward for the last day
                ## d days in history data. The index of the last day is d-1
                Rdm1_imp_B.append(comb_dat['comb_RR'][d - 1].copy())
                
                ## SLSVI
                if d % L == 0: ## history data contain L episodes
                    i = d // L ## number of episodes
                    weights = rd.exponential(scale=1, size=i + J)
                    mediator_weight = np.maximum(1 / np.max(np.abs(est_theta_R[1:6])), 1)
                    beta = lsvi.get_Q(
                        betas[:, [b]], comb_dat, weights, i + J, lam_beta, 
                        mediator_weight * est_theta_R.reshape(-1)
                    )
                    betas[:, [b]] = beta.copy()
            ## .loc contains the start and end of indices
            ## refill missing rewards in the last day
            dat.df.loc[d, 'R_imp'] = Rdm1_imp_B[0][0]
            
            Rdm1_imp = dat.df.loc[d, 'R_imp']
            Edm1 = np.array([dat.df.loc[d, 'E']])
            Rdm1 = np.array([dat.df.loc[d, 'R']])
            Cd = np.zeros((config.K, config.dC))
            Pd = np.zeros((config.K, config.dP))
            Ad = np.zeros((config.K, config.dA)).astype(int)
            Md = np.zeros((config.K, config.dM))
            ## assign treatments and observe new data
            l = d % L  ## day in the episode
            for k in range(0, config.K):
                Cd[k] = env.gen_Ch(d, k)
                Ad[k] = lsvi.choose_A(betas, Edm1, Rdm1_imp, Cd, Ad, Md, l, k)
                Md[k] = env.gen_Mh(Edm1, Rdm1, Cd[k], Ad[k], d, k)
            Ed = env.gen_Ed(Ad, Edm1, d)
            Rd = env.gen_Rd(Md, Ed, Rdm1, d)
            Od = env.gen_Od(Rdm1, d)
            if (d + 1) % config.W == 0:
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

