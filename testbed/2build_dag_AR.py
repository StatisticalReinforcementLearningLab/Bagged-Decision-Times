# %% set up


import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
import json
from regression import LinearReg
np.set_printoptions(suppress=True)


# %% read data


# dropbox = "/Users/daiqigao/Dropbox (Harvard University)/"
dropbox = "/Users/dqgao/Library/CloudStorage/Dropbox-HarvardUniversity/"
folder_dat = dropbox + "*Shared/HeartStepsV2V3/Daiqi/Data/"
aim = '2'
folder_params = 'params_env_AR_V' + aim + '/'
folder_pred = 'pred_V' + aim + '/'
folder_figure = 'Figures/'
file_dat = folder_dat + 'aim-' + aim + '_with_R.csv'
file_params_std = 'params_std_V' + aim + '.json'
file_params_env_prefix = folder_params + 'params_env_'
file_pred_prefix = folder_pred + 'pred_'
file_user_ids = folder_params + 'user_ids.txt'

dat_org = pd.read_csv(file_dat)
userid_all = pd.unique(dat_org['id'])
dC, dP, dA, dM, dE, dR, dO = 1, 1, 1, 1, 1, 1, 1 # dimensions
n = len(userid_all)
K = 5


# %% standardization parameters


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


alpha_l2_list = [0.2, 0.5, 1, 2, 5]
alpha_lap_list = [0.5, 1, 2, 5]
ncv = 5
seed = 2024
alpha_cv = []
alpha_cv_std = []
dat_user_all = []

for i, userid in enumerate(userid_all):
    dat_user = dat_org[dat_org['id'] == userid]
    ## fit the DAG
    I = dat_user['I'].to_numpy().reshape(-1, K)
    C = dat_user['C'].to_numpy().reshape(-1, K)
    A = dat_user['A'].to_numpy().reshape(-1, K)
    M = dat_user['M'].to_numpy().reshape(-1, K)
    Ed = dat_user['E'].to_numpy().reshape(-1, K)[:, 0]
    Edm1 = dat_user['E0'].to_numpy().reshape(-1, K)[:, 0]
    Rd = dat_user['R'].to_numpy().reshape(-1, K)[:, 0]
    Rdm1 = dat_user['R0'].to_numpy().reshape(-1, K)[:, 0]
    O = dat_user['O'].to_numpy().reshape(-1, K)[:, 0]
    E0 = Edm1[[0]]
    R0 = Rdm1[[0]]

    D = Ed.shape[0]
    T = D * K


    ## model for M
    Edm1_rep = np.repeat(Edm1, K)
    Rdm1_rep = np.repeat(Rdm1, K)
    I_h = I.reshape(-1)
    C_h = C.reshape(-1)
    A_h = A.reshape(-1)
    M_h = M.reshape(-1)
    idx_obs = (~np.isnan(M_h)) & (I_h == 1)
    if np.var(C_h) == 0:
        M_cond = np.stack([
            np.ones(T), Edm1_rep, Rdm1_rep, 
            A_h, A_h * Edm1_rep, A_h * Rdm1_rep, 
        ], axis=1) ## dim = dCM
    else:
        M_cond = np.stack([
            np.ones(T), Edm1_rep, Rdm1_rep, C_h,
            A_h, A_h * Edm1_rep, A_h * Rdm1_rep, A_h * C_h
        ], axis=1) ## dim = dCM
    M_cond = M_cond[idx_obs, :]
    M_h = M_h[idx_obs]
    model_M = RidgeCV(alphas=alpha_l2_list, fit_intercept=False, cv=ncv)
    model_M.fit(M_cond, M_h)
    alpha_M_l2 = model_M.alpha_
    theta_M_mean = model_M.coef_
    pred_M = model_M.predict(M_cond)
    resid_obs_M = M_h - pred_M
    sigma2_M_mean = np.var(resid_obs_M)
    if np.var(C_h) == 0:
        theta_M_mean = np.insert(theta_M_mean, [3, 6], [0, 0])
    ## fill in the missing entries
    resid_M = np.zeros(T)
    resid_M[idx_obs] = resid_obs_M
    resid_M[~idx_obs] = np.nan


    ## model for E
    E_cond = np.hstack([
        np.ones((D, 1)), Edm1.reshape(-1, 1), A, A * Edm1.reshape(-1, 1)
    ])
    ## Laplacian matrix
    dE_cond = E_cond.shape[1]
    L_E_A = np.zeros((dE_cond, dE_cond))
    L_E_A[2:(2 + K), 2:(2 + K)] = 1
    L_E_A[7:(7 + K), 7:(7 + K)] = 1
    np.fill_diagonal(L_E_A, 0)
    L_E_D = np.diag(np.sum(L_E_A, axis=0))
    L_E = L_E_D - L_E_A
    model_E = LinearReg(alpha_l2_list, alpha_lap_list, ncv)
    model_E.cv(E_cond, Ed, L_E, seed)
    alpha_E_l2, alpha_E_lap = model_E.alpha_l2, model_E.alpha_lap
    model_E.fit(E_cond, Ed, L_E, alpha_E_l2, alpha_E_lap)
    ## when unstandardized Ed == 0, there is collinearity in E_cond
    theta_E_mean = model_E.beta
    pred_E = model_E.predict(E_cond)
    resid_E = Ed - pred_E
    sigma2_E_mean = np.var(resid_E)


    ## model for R
    ## add A into the condition of R
    M_imp = dat_user.pivot(index='day', columns='time', values='M')
    M_imp = M_imp.fillna(M_imp.mean()).to_numpy()
    R_cond = np.hstack([
        np.ones((D, 1)), M_imp, Ed.reshape(-1, 1), Rdm1.reshape(-1, 1),
        A
    ])
    ## Laplacian matrix
    dR_cond = R_cond.shape[1]
    L_R_A = np.zeros((dR_cond, dR_cond))
    L_R_A[1:(1 + K), 1:(1 + K)] = 1
    L_R_A[8:(8 + K), 8:(8 + K)] = 1
    np.fill_diagonal(L_R_A, 0)
    L_R_D = np.diag(np.sum(L_R_A, axis=0))
    L_R = L_R_D - L_R_A
    model_R = LinearReg(alpha_l2_list, alpha_lap_list, ncv)
    model_R.cv(R_cond, Rd, L_R, seed)
    alpha_R_l2, alpha_R_lap = model_R.alpha_l2, model_R.alpha_lap
    model_R.fit(R_cond, Rd, L_R, alpha_R_l2, alpha_R_lap)
    theta_R_mean = model_R.beta
    pred_R = model_R.predict(R_cond)
    resid_R = Rd - pred_R
    sigma2_R_mean = np.var(resid_R)


    ## model for O
    O_cond = np.stack([
        np.ones(D), Rdm1
    ], axis=1)
    model_O = RidgeCV(alphas=alpha_l2_list, fit_intercept=False, cv=ncv)
    model_O.fit(O_cond, O)
    alpha_O_l2 = model_O.alpha_
    theta_O_mean = model_O.coef_
    pred_O = model_O.predict(O_cond)
    resid_O = O - model_O.predict(O_cond)
    sigma2_O_mean = np.var(resid_O)


    alpha_cv.append([alpha_M_l2, alpha_E_l2, alpha_E_lap, alpha_R_l2, alpha_R_lap, alpha_O_l2])

    digits = 3
    env_para = {
        'E0': np.round(E0, digits).tolist(),
        'R0': np.round(R0, digits).tolist(),
        'theta_C': np.round(0.0, digits).tolist(),
        'theta_M': np.round(theta_M_mean, digits).tolist(),
        'theta_E': np.round(theta_E_mean, digits).tolist(),
        'theta_R': np.round(theta_R_mean, digits).tolist(),
        'theta_O': np.round(theta_O_mean, digits).tolist(),
        'resid_C': np.round(C.reshape(-1), digits).tolist(),
        'resid_M': np.round(resid_M, digits).tolist(),
        'resid_E': np.round(resid_E, digits).tolist(),
        'resid_R': np.round(resid_R, digits).tolist(),
        'resid_O': np.round(resid_O, digits).tolist(),
    }
    predicted = {
        'pred_M': np.round(pred_M, digits).tolist(),
        'pred_E': np.round(pred_E, digits).tolist(),
        'pred_R': np.round(pred_R, digits).tolist(),
        'pred_O': np.round(pred_O, digits).tolist(),
    }

    with open(file_params_env_prefix + str(userid) + '.json', 'w') as file:
        file.write(json.dumps(env_para))


np.savetxt(file_user_ids, userid_all, fmt='%d')
alpha_cv = np.array(alpha_cv)


# %%
