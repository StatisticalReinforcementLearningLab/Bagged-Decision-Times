# %% set up


import numpy as np
import numpy.random as rd
from env_config_base import EnvConfigBase


# %% environment class


class Env:
    def __init__(self, env_config: EnvConfigBase, noise='sequential'):
        self.K = env_config.K
        assert noise in ['sequential', 'random']
        self.noise = noise

        self.dC = env_config.dC
        self.dA = env_config.dA
        self.dM = env_config.dM
        self.dE = env_config.dE
        self.dR = env_config.dR
        self.dO = env_config.dO

        self.limits_C = env_config.limits_C
        self.limits_M = env_config.limits_M
        self.limits_E = env_config.limits_E
        self.limits_R = env_config.limits_R
        self.limits_O = env_config.limits_O

        self.E0 = env_config.E0
        self.R0 = env_config.R0
        self.theta_C = env_config.theta_C.reshape(-1, self.dC)
        self.theta_M = env_config.theta_M.reshape(-1, self.dM)
        self.theta_E = env_config.theta_E.reshape(-1, self.dE)
        self.theta_R = env_config.theta_R.reshape(-1, self.dR)
        self.theta_O = env_config.theta_O.reshape(-1, self.dO)
        self.resid_C = env_config.resid_C
        self.resid_M = env_config.resid_M
        self.resid_E = env_config.resid_E
        self.resid_R = env_config.resid_R
        self.resid_O = env_config.resid_O

        idx_na = np.isnan(self.resid_M) ## index of missing residuals in M
        self.resid_obs_M = self.resid_M[~idx_na] ## observed residuals in M

    
    def gen_Ch_mean(self, n = 1):
        size = (1, ) if n == 1 else (n, 1)
        return np.tile(self.theta_C, size)


    def gen_Ch(self, d, k):  ## currently only support n=1
        C_mean = self.gen_Ch_mean()
        if self.noise == 'sequential':
            ## cyclically sample the noise term from the residual vector
            C_noise = self.resid_C[(d * self.K + k) % len(self.resid_C)]
        elif self.noise == 'random':
            C_noise = rd.choice(self.resid_C, size=1)
        C_new = C_mean + C_noise
        return np.clip(C_new, self.limits_C[0], self.limits_C[1])


    def gen_Mh_mean(self, Edm1, Rdm1, Ch, Ah, n = 1):
        size = (1, ) if n == 1 else (n, 1)
        dCM = 1 + self.dE + self.dR + self.dC
        ## find the mean value conditional on the predictors
        M_cond = np.hstack([
            np.ones(size), Edm1, Rdm1, Ch
        ]).reshape(n, dCM)
        Mh_main = np.matmul(M_cond, self.theta_M[:4, :]).reshape(-1) 
        Mh_trt = (Ah * np.matmul(M_cond, self.theta_M[4:, :])).reshape(-1)
        Mh_trt = np.maximum(Mh_trt, 0)
        return Mh_main + Mh_trt


    def gen_Mh(self, Edm1, Rdm1, Ch, Ah, d, k):  ## currently only support n=1
        M_mean = self.gen_Mh_mean(Edm1, Rdm1, Ch, Ah)
        if self.noise == 'sequential':
            ## cyclically sample the noise term from the residual vector
            M_noise = self.resid_M[(d * self.K + k) % len(self.resid_M)]
            ## if this residual is missing, randomly sample from observed residuals
            if np.isnan(M_noise):
                M_noise = rd.choice(self.resid_obs_M, size=1)
        elif self.noise == 'random':
            M_noise = rd.choice(self.resid_obs_M, size=1)
        M_new = M_mean + M_noise
        return np.clip(M_new, self.limits_M[0], self.limits_M[1])


    def gen_Ed_mean(self, Ad, Edm1, n = 1):
        size = (1, ) if n == 1 else (n, 1)
        ## dimension of Ad is K * dA
        Ad = Ad.reshape(self.K * self.dA) if n == 1 else Ad.reshape(n, self.K * self.dA)
        ## find the mean value conditional on the predictors
        E_cond = np.hstack([
            np.ones(size), Edm1, Ad, Ad * Edm1
        ])
        E_cond = E_cond.reshape(n, 1 + self.dE + self.K * self.dA * (1 + self.dE))
        Ed = np.matmul(E_cond, self.theta_E).reshape(-1)
        return Ed


    def gen_Ed(self, Ad, Edm1, d):
        E_mean = self.gen_Ed_mean(Ad, Edm1)
        if self.noise == 'sequential':
            ## cyclically sample the noise term from the residual vector
            E_noise = self.resid_E[d % len(self.resid_E)]
        elif self.noise == 'random':
            E_noise = rd.choice(self.resid_E, size=1)
        E_new = E_mean + E_noise
        return np.clip(E_new, self.limits_E[0], self.limits_E[1])


    def gen_Rd_mean(self, Md, Ed, Rdm1, n = 1):
        size = (1, ) if n == 1 else (n, 1)
        ## dimension of Md is K * dM
        Md = Md.reshape(self.K * self.dM) if n == 1 else Md.reshape(n, self.K * self.dM)
        ## find the mean value conditional on the predictors
        R_cond = np.hstack([
            np.ones(size), Md, Ed, Rdm1
        ])
        R_cond = R_cond.reshape(n, 1 + self.K * self.dM + self.dE + self.dR)
        Rd = np.matmul(R_cond, self.theta_R).reshape(-1) 
        return Rd


    def gen_Rd(self, Md, Ed, Rdm1, d):
        R_mean = self.gen_Rd_mean(Md, Ed, Rdm1)
        if self.noise == 'sequential':
            ## cyclically sample the noise term from the residual vector
            R_noise = self.resid_R[d % len(self.resid_R)]
        elif self.noise == 'random':
            R_noise = rd.choice(self.resid_R, size=1)
        R_new = R_mean + R_noise
        return np.clip(R_new, self.limits_R[0], self.limits_R[1])


    def gen_Od_mean(self, Rdm1, n = 1):
        size = (1, ) if n == 1 else (n, 1)
        ## find the mean value conditional on the predictors
        O_cond = np.hstack([
            np.ones(size), Rdm1
        ])
        O_cond = O_cond.reshape(n, 1 + self.dO)
        Od = np.matmul(O_cond, self.theta_O).reshape(-1) 
        return Od


    def gen_Od(self, Rdm1, d):
        O_mean = self.gen_Od_mean(Rdm1)
        if self.noise == 'sequential':
            ## cyclically sample the noise term from the residual vector
            O_noise = self.resid_O[d % len(self.resid_O)]
        elif self.noise == 'random':
            O_noise = rd.choice(self.resid_O, size=1)
        O_new = O_mean + O_noise
        return np.clip(O_new, self.limits_O[0], self.limits_O[1])


# %%
