# %% set up


import numpy as np
import pandas as pd
from statsmodels.api import OLS
import json
np.set_printoptions(suppress=True)


# %% read data


# dropbox = "/Users/daiqigao/Dropbox (Harvard University)/"
dropbox = "/Users/dqgao/Library/CloudStorage/Dropbox-HarvardUniversity/"
folder_dat = dropbox + "*Shared/HeartStepsV2V3/Daiqi/Data/"
aim = '3'
aim_std = '2'
file_dat = folder_dat + 'aim-' + aim + '_with_R.csv'
file_params_std = 'params_std_V' + aim_std + '.json'
file_params_prior = 'params_prior_V' + aim + '_TS.json'

dat_org = pd.read_csv(file_dat, index_col=False)
userid_all = pd.unique(dat_org['id'])
dC, dP, dA, dM, dE, dR, dO = 1, 1, 1, 1, 1, 1, 1 # dimensions
n = len(userid_all)
K = 5


# %% standardization


with open(file_params_std, 'r') as file:
    std_params = json.load(file)

C_shift = std_params['C_shift']
C_scale = std_params['C_scale']
M_shift = std_params['M_shift']
M_scale = std_params['M_scale']
E_shift = std_params['E_shift']
E_scale = std_params['E_scale']
O_shift = std_params['O_shift']
O_scale = std_params['O_scale']

dat_org['C'] = (dat_org['C'] - C_shift) / C_scale
dat_org['M'] = (dat_org['M'] - M_shift) / M_scale
dat_org['E'] = (dat_org['E'] - E_shift) / E_scale
dat_org['E0'] = (dat_org['E0'] - E_shift) / E_scale
dat_org['O'] = (dat_org['O'] - O_shift) / O_scale


# %% build DAG


seed = 2024
alpha_cv = []
alpha_cv_std = []
dat_user_all = []

## fit the DAG
theta_M_mean = []
theta_M_var = []
sigma2_M = []
for i, userid in enumerate(userid_all):
    dat_user = dat_org[dat_org['id'] == userid]
    userid = dat_user['id'].to_numpy().reshape(-1, K)[:, 0]
    I = dat_user['I'].to_numpy().reshape(-1, K)
    C = dat_user['C'].to_numpy().reshape(-1, K)
    A = dat_user['A'].to_numpy().reshape(-1, K)
    M = dat_user['M'].to_numpy().reshape(-1, K)
    Ed = dat_user['E'].to_numpy().reshape(-1, K)[:, 0]
    Edm1 = dat_user['E0'].to_numpy().reshape(-1, K)[:, 0]
    Rd = dat_user['R'].to_numpy().reshape(-1, K)[:, 0]
    Rdm1 = dat_user['R0'].to_numpy().reshape(-1, K)[:, 0]
    O = dat_user['O'].to_numpy().reshape(-1, K)[:, 0]
    E0 = Edm1[0]
    R0 = Rdm1[0]

    D = Ed.shape[0]
    T = D * K


    ## model for M
    userid_rep = np.repeat(userid, K)
    Edm1_rep = np.repeat(Edm1, K)
    Rdm1_rep = np.repeat(Rdm1, K)
    I_h = I.reshape(-1)
    C_h = C.reshape(-1)
    A_h = A.reshape(-1)
    M_h = M.reshape(-1)
    idx_obs = (~np.isnan(M_h)) & (I_h == 1)
    M_cond = np.stack([
        np.ones(T), C_h, A_h, A_h * C_h
    ], axis=1) ## dim = dCM
    M_cond = M_cond[idx_obs, :]
    M_h = M_h[idx_obs]
    userid_M = userid_rep[idx_obs]
    model_M = OLS(M_h, M_cond)
    model_M_res = model_M.fit()
    theta_M_mean.append(model_M_res.params)
    theta_M_var.append(model_M_res.bse)
    sigma2_M.append(model_M_res.scale)

theta_M_var = np.var(np.array(theta_M_mean), axis=0)
theta_M_var = np.diag(theta_M_var)
theta_M_mean = np.mean(np.array(theta_M_mean), axis=0)
sigma2_M = np.mean(np.array(sigma2_M), axis=0)

digits = 3
digits_var = 3
art_para = {
    'theta_M_mean': np.round(theta_M_mean, digits).tolist(),
    'theta_M_var': np.round(theta_M_var, digits_var).tolist(),
    'sigma2_M': np.round(sigma2_M, digits).tolist(),
}

with open(file_params_prior, 'w') as file:
    file.write(json.dumps(art_para))


# %%
