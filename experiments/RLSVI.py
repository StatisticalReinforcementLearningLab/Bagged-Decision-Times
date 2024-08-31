# %% set up


import numpy as np
import numpy.random as rd
import numpy.linalg as LA
from env_config_base import EnvConfigBase
from dataset import Dataset


# %% LSVI


class LSVI():
    def __init__(
        self, env_config: EnvConfigBase, H: int, lam_beta: float, 
        sigma2=1, gamma=0.5, standardize=False
    ):
        self.K = env_config.K # number of decision times in a day
        self.H = H # horizon
        self.L = self.H // self.K # number of days in an episode
        self.gamma = gamma # discount factor
        self.standardize = standardize

        self.C_shift = env_config.C_shift
        self.C_scale = env_config.C_scale
        self.M_shift = env_config.M_shift
        self.M_scale = env_config.M_scale
        self.E_shift = env_config.E_shift
        self.E_scale = env_config.E_scale
        self.O_shift = env_config.O_shift
        self.O_scale = env_config.O_scale

        self.dC = env_config.dC
        self.dA = env_config.dA
        self.dM = env_config.dM
        self.dE = env_config.dE
        self.dR = env_config.dR
        self.dO = env_config.dO
        self.dX_h = [ # number of variables in the Q-function at time h
            1 + (h // self.K + 1) * self.dE + self.dR + (h + 1) * self.dC + h * self.dA 
            + self.dA * (1 + (h // self.K + 1) * self.dE + self.dR + (h + 1) * self.dC)
            for h in range(self.H)
        ]
        self.dX = sum(self.dX_h) # total number of variables
        
        ## prior mean = 0
        self.prior_beta_mean = [np.zeros((self.dX_h[h], 1)) for h in range(self.H)]
        ## prior variance = 1/lambda I
        self.prior_beta_var = [1/lam_beta * np.diag(np.ones(self.dX_h[h])) for h in range(self.H)]
        self.sigma2 = sigma2 # noise variance

        # create space to store posterior distribution
        self.post_beta_mean = [np.zeros((self.dX_h[h], 1)) for h in range(self.H)]
        self.post_beta_var = [np.zeros((self.dX_h[h], self.dX_h[h])) for h in range(self.H)]
        self.beta_tilde = [np.zeros((self.dX_h[h], 1)) for h in range(self.H)]


    def get_Q(self, beta0: np.ndarray, comb_dat: dict, ii: int):
        ## ii is the total number of episodes in the combined dataset
        ## R_{d-1} and R_d may contain missing data (no imputation)
        comb_CC = comb_dat['comb_CC'].reshape(ii, self.L * self.K, self.dC)
        comb_AA = comb_dat['comb_AA'].reshape(ii, self.L * self.K, self.dA)
        comb_EEdm1 = comb_dat['comb_EEdm1'].reshape(ii, self.L, self.dE)
        comb_EE = comb_dat['comb_EE'].reshape(ii, self.L, self.dE)
        comb_RRdm1 = comb_dat['comb_RRdm1'].reshape(ii, self.L, self.dR)
        comb_RR = comb_dat['comb_RR'].reshape(ii, self.L, self.dR)

        if self.standardize:
            comb_CC = (comb_CC - self.C_shift) / self.C_scale
            comb_EEdm1 = (comb_EEdm1 - self.E_shift) / self.E_scale
            comb_EE = (comb_EE - self.E_shift) / self.E_scale

        ## next episode
        intercept = np.ones((ii, 1))
        comb_A0 = np.zeros((ii, 1))
        comb_A1 = np.ones((ii, 1))
        ## state E in previous L days
        comb_E_prev_h = comb_EE[:, -1, :].reshape(ii, -1)
        ## state R0
        comb_R0_h = comb_RR[:, -1, :].reshape(ii, -1)
        ## state C
        comb_C_to_h = rd.choice(comb_CC.reshape(-1), size=(ii, 1), replace=True)
        ## state M
        comb_A_prev_h = np.zeros((ii, 0))
        ## main effect
        comb_main_h = np.hstack([
            intercept, comb_E_prev_h, comb_R0_h, 
            comb_C_to_h, comb_A_prev_h, 
        ])
        ## concatenate main and interaction effect
        comb_X_h_A0 = np.hstack([
            comb_main_h, 
            comb_A0, comb_A0 * comb_E_prev_h, 
            comb_A0 * comb_R0_h, comb_A0 * comb_C_to_h, 
        ])
        comb_X_h_A1 = np.hstack([
            comb_main_h, 
            comb_A1, comb_A1 * comb_E_prev_h, 
            comb_A1 * comb_R0_h, comb_A1 * comb_C_to_h, 
        ])
        ## optimal Q
        pred_Q_A0 = comb_X_h_A0 @ beta0
        pred_Q_A1 = comb_X_h_A1 @ beta0
        pred_Q_opt = np.maximum(pred_Q_A0, pred_Q_A1)

        comb_R = comb_RR[:, -1, :].copy()
        for h in range(self.H - 1, -1, -1):
            l = h // self.K  ## day in the episode
            ## state E in previous L days
            comb_E_prev_h = comb_EEdm1[:, :(l + 1), :].reshape(ii, -1)
            ## state R0
            comb_R0_h = comb_RRdm1[:, 0, :].reshape(ii, -1)
            ## state C
            comb_C_to_h = comb_CC[:, :(h + 1), :].reshape(ii, -1)
            ## state M
            comb_A_prev_h = comb_AA[:, :h, :].reshape(ii, -1)
            ## current action
            comb_A_h = comb_AA[:, h, :].copy()
            ## main effect
            comb_main_h = np.hstack([
                intercept, comb_E_prev_h, comb_R0_h, 
                comb_C_to_h, comb_A_prev_h, 
            ])
            ## concatenate main and interaction effect
            comb_X_h = np.hstack([
                comb_main_h, 
                comb_A_h, comb_A_h * comb_E_prev_h, 
                comb_A_h * comb_R0_h, comb_A_h * comb_C_to_h, 
            ])
            comb_X_h_A0 = np.hstack([
                comb_main_h, 
                comb_A0, comb_A0 * comb_E_prev_h, 
                comb_A0 * comb_R0_h, comb_A0 * comb_C_to_h, 
            ])
            comb_X_h_A1 = np.hstack([
                comb_main_h, 
                comb_A1, comb_A1 * comb_E_prev_h, 
                comb_A1 * comb_R0_h, comb_A1 * comb_C_to_h, 
            ])
            ## response
            if h == self.H - 1:
                comb_Y_h = comb_R + self.gamma * pred_Q_opt
            else:
                comb_Y_h = pred_Q_opt.copy()
    
            ## posterior variance
            post_var = comb_X_h.T @ comb_X_h / self.sigma2 + LA.inv(self.prior_beta_var[h])
            ## round the inverse posterior variance for numerical stability
            ## otherwise, the inverse posterior variance may be asymmetric
            post_var = np.round(LA.inv(post_var), decimals=6)
            self.post_beta_var[h] = post_var.copy()
            
            ## posterior mean
            post_mean = comb_X_h.T @ comb_Y_h / self.sigma2
            post_mean += LA.inv(self.prior_beta_var[h]) @ self.prior_beta_mean[h]
            post_mean = post_var @ post_mean
            self.post_beta_mean[h] = post_mean.copy()

            ## sample from the posterior
            rng = rd.default_rng()
            beta_tilde = rng.multivariate_normal(
                self.post_beta_mean[h].reshape(-1), self.post_beta_var[h], 1
            ).reshape(-1, 1)
            self.beta_tilde[h] = beta_tilde.copy()

            ## optimal Q
            pred_Q_A0 = comb_X_h_A0 @ beta_tilde
            pred_Q_A1 = comb_X_h_A1 @ beta_tilde
            pred_Q_opt = np.maximum(pred_Q_A0, pred_Q_A1)
        
        return self.beta_tilde


    def choose_A(self, dat: Dataset, d: int, k: int):
        intercept, A0, A1 = np.ones(1), np.zeros(1), np.ones(1)

        ## there are d days in the history, corresponding to indices 1 to d in dat
        ## currently on day d + 1
        i = d // self.L  ## the order of the episode
        m = i * self.L  ## start index of this episode
        h = (d - m) * self.K + k  ## order of the time in this episode
        ## state Edm1
        E_prev_h = dat.df.loc[m:d, 'E'].values  ## including m and d
        ## state R0
        R0_h = dat.df.loc[m:m, 'R'].values  ## dat.df.loc[m, 'R'] is a float
        ## state C
        C_to_h = dat.df.loc[(m + 1):(d + 1), dat.col_C].values.reshape(-1)
        if k < self.K - 1:
            C_to_h = C_to_h[:-(self.K - 1 - k)]
        ## state M
        A_prev_h = dat.df.loc[(m + 1):(d + 1), dat.col_A].values.reshape(-1)
        A_prev_h = A_prev_h[:-(self.K - k)]

        if self.standardize:
            C_to_h = (C_to_h - self.C_shift) / self.C_scale
            E_prev_h = (E_prev_h - self.E_shift) / self.E_scale

        ## main effect
        main = np.hstack([
            intercept, E_prev_h, R0_h, 
            C_to_h, A_prev_h, 
        ])
        ## concatenate main and interaction effect
        X_A0 = np.hstack([
            main, 
            A0, A0 * E_prev_h, 
            A0 * R0_h, A0 * C_to_h, 
        ]).reshape(1, -1)
        X_A1 = np.hstack([
            main, 
            A1, A1 * E_prev_h, 
            A1 * R0_h, A1 * C_to_h, 
        ]).reshape(1, -1)

        ## greedy action
        post_mean_A0 = (X_A0 @ self.beta_tilde[h]).item()
        post_mean_A1 = (X_A1 @ self.beta_tilde[h]).item()
        trt = int(post_mean_A1 >= post_mean_A0)
        return trt


# %%
