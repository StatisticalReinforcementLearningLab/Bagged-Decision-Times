# %% set up


import numpy as np
import pandas as pd
from regression import BayesianLinearReg
import json
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


# %% read data


# dropbox = "/Users/daiqigao/Dropbox (Harvard University)/"
dropbox = "/Users/dqgao/Library/CloudStorage/Dropbox-HarvardUniversity/"
folder_dat = dropbox + "*Shared/HeartStepsV2V3/Daiqi/Data/"
aim = '3'
aim_std = '2'
folder_figure = 'Figures/'
file_dat = folder_dat + 'aim-' + aim + '_with_R.csv'
file_params_std = 'params_std_V' + aim_std + '.json'
file_params_prior = 'params_prior_V' + aim + '.json'

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
userid = dat_org['id'].to_numpy().reshape(-1, K)[:, 0]
I = dat_org['I'].to_numpy().reshape(-1, K)
C = dat_org['C'].to_numpy().reshape(-1, K)
A = dat_org['A'].to_numpy().reshape(-1, K)
M = dat_org['M'].to_numpy().reshape(-1, K)
Ed = dat_org['E'].to_numpy().reshape(-1, K)[:, 0]
Edm1 = dat_org['E0'].to_numpy().reshape(-1, K)[:, 0]
Rd = dat_org['R'].to_numpy().reshape(-1, K)[:, 0]
Rdm1 = dat_org['R0'].to_numpy().reshape(-1, K)[:, 0]
O = dat_org['O'].to_numpy().reshape(-1, K)[:, 0]
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
    np.ones(T), Edm1_rep, Rdm1_rep, C_h,
    A_h, A_h * Edm1_rep, A_h * Rdm1_rep, A_h * C_h
], axis=1) ## dim = dCM
M_cond = M_cond[idx_obs, :]
M_h = M_h[idx_obs]
userid_M = userid_rep[idx_obs]
model_M = BayesianLinearReg(M_cond.shape[1])
model_M.fit(M_cond, M_h, userid_M)
resid_M = M_h - (M_cond @ model_M.post_mu).reshape(-1)


## model for E
E_cond = np.hstack([
    np.ones((D, 1)), Edm1.reshape(-1, 1), A, A * Edm1.reshape(-1, 1)
])
model_E = BayesianLinearReg(E_cond.shape[1])
model_E.fit(E_cond, Ed, userid)
resid_E = Ed - (E_cond @ model_E.post_mu).reshape(-1)


## model for R
M_imp = dat_org.pivot(index=['id', 'day'], columns='time', values='M')
M_imp = M_imp.fillna(M_imp.mean()).to_numpy()
R_cond = np.hstack([
    np.ones((D, 1)), M_imp, Ed.reshape(-1, 1), Rdm1.reshape(-1, 1)
])
model_R = BayesianLinearReg(R_cond.shape[1])
model_R.fit(R_cond, Rd, userid)
resid_R = Rd - (R_cond @ model_R.post_mu).reshape(-1)


## model for O
O_cond = np.stack([
    np.ones(D), Rdm1
], axis=1)
model_O = BayesianLinearReg(O_cond.shape[1])
model_O.fit(O_cond, O, userid)
resid_O = O - (O_cond @ model_O.post_mu).reshape(-1)


theta_M_mean = model_M.post_mu.reshape(-1)
theta_M_var = model_M.post_Sigma
sigma2_M = model_M.sigma2
theta_E_mean = model_E.post_mu.reshape(-1)
theta_E_var = model_E.post_Sigma
sigma2_E = model_E.sigma2
theta_R_mean = model_R.post_mu.reshape(-1)
theta_R_var = model_R.post_Sigma
sigma2_R = model_R.sigma2
theta_O_mean = model_O.post_mu.reshape(-1)
theta_O_var = model_O.post_Sigma
sigma2_O = model_O.sigma2

digits = 3
digits_var = 8
art_para = {
    'C': np.round(dat_org['C'], digits).tolist(),
    'E0': np.round(dat_org['E0'], digits).tolist(),
    'R0': np.round(dat_org['R0'], digits).tolist(),
    'theta_M_mean': np.round(theta_M_mean, digits).tolist(),
    'theta_M_var': np.round(theta_M_var, digits_var).tolist(),
    'sigma2_M': np.round(sigma2_M, digits).tolist(),
    'theta_E_mean': np.round(theta_E_mean, digits).tolist(),
    'theta_E_var': np.round(theta_E_var, digits_var).tolist(),
    'sigma2_E': np.round(sigma2_E, digits).tolist(),
    'theta_R_mean': np.round(theta_R_mean, digits).tolist(),
    'theta_R_var': np.round(theta_R_var, digits_var).tolist(),
    'sigma2_R': np.round(sigma2_R, digits).tolist(),
    'theta_O_mean': np.round(theta_O_mean, digits).tolist(),
    'theta_O_var': np.round(theta_O_var, digits_var).tolist(),
    'sigma2_O': np.round(sigma2_O, digits).tolist(),
    'resid_M': np.round(resid_M, digits).tolist(),
    'resid_E': np.round(resid_E, digits).tolist(),
    'resid_R': np.round(resid_R, digits).tolist(),
    'resid_O': np.round(resid_O, digits).tolist(),
}

with open(file_params_prior, 'w') as file:
    file.write(json.dumps(art_para))


# %% plots


fig_size = (10, 2.8)
subplot_size = (1, 4)
color = '#53446B'
alpha = 0.7

fig, ax = plt.subplots(subplot_size[0], subplot_size[1], figsize=fig_size, constrained_layout=True)

j = 0
ax_idx = np.unravel_index(j, subplot_size)
ax_idx = j
ax[ax_idx].errorbar(
    np.arange(len(theta_M_mean)), theta_M_mean, 
    yerr=1.96 * np.sqrt(np.diag(theta_M_var)), capsize=3, 
    fmt=".", color=color, alpha=alpha)
ax[ax_idx].title.set_text(r'$\mu^{M}$')
ax[ax_idx].grid()

j += 1
ax_idx = np.unravel_index(j, subplot_size)
ax_idx = j
ax[ax_idx].errorbar(
    np.arange(len(theta_E_mean)), theta_E_mean, 
    yerr=1.96 * np.sqrt(np.diag(theta_E_var)), capsize=3, 
    fmt=".", color=color, alpha=alpha)
ax[ax_idx].title.set_text(r'$\mu^{E}$')
ax[ax_idx].grid()

j += 1
ax_idx = np.unravel_index(j, subplot_size)
ax_idx = j
ax[ax_idx].errorbar(
    np.arange(len(theta_R_mean)), theta_R_mean, 
    yerr=1.96 * np.sqrt(np.diag(theta_R_var)), capsize=3, 
    fmt=".", color=color, alpha=alpha)
ax[ax_idx].title.set_text(r'$\mu^{R}$')
ax[ax_idx].grid()

j += 1
ax_idx = np.unravel_index(j, subplot_size)
ax_idx = j
ax[ax_idx].errorbar(
    np.arange(len(theta_O_mean)), theta_O_mean, 
    yerr=1.96 * np.sqrt(np.diag(theta_O_var)), capsize=3, 
    fmt=".", color=color, alpha=alpha)
ax[ax_idx].title.set_text(r'$\mu^{O}$')
ax[ax_idx].grid()

fig.supxlabel('index')
fig.supylabel('value')
plt.savefig(folder_figure + 'prior_V' + aim + '.pdf')
plt.show()


# %%
