# %% set up


import numpy as np
import numpy.random as rd
import numpy.linalg as LA
from scipy.stats import norm
from env_config_base import EnvConfigBase
from dataset import Dataset


# %% TS


class TS():
    def __init__(self, env_config: EnvConfigBase, lam_beta: float, 
        sigma2=1, standardize=False
    ):
        self.K = env_config.K
        self.standardize = standardize

        self.C_shift = env_config.C_shift
        self.C_scale = env_config.C_scale
        self.M_shift = env_config.M_shift
        self.M_scale = env_config.M_scale
        self.E_shift = env_config.E_shift
        self.E_scale = env_config.E_scale

        self.dC = env_config.dC
        self.dA = env_config.dA
        self.dM = env_config.dM
        self.dE = env_config.dE
        self.dR = env_config.dR
        self.dAS = 1 + self.dE + self.dR + self.dC
        self.dX = (1 + self.dA) * self.dAS

        self.prior_beta_mean = np.zeros((self.dX, 1)) # prior mean
        self.prior_beta_var = 1/lam_beta * np.diag(np.ones(self.dX)) # prior variance
        self.sigma2 = sigma2 # noise variance


    def update(self, dat: Dataset, d: int, k: int):
        i = d * self.K + k # total number of decision times
        CC = dat.df.loc[1:(d + 1), dat.col_C].values.reshape(-1)[:i].reshape(-1, 1)
        AA = dat.df.loc[1:(d + 1), dat.col_A].values.reshape(-1)[:i].reshape(-1, 1)
        MM = dat.df.loc[1:(d + 1), dat.col_M].values.reshape(-1)[:i].reshape(-1, 1)
        EEdm1 = np.repeat(dat.df.loc[1:(d + 1), 'E'].values, self.K)[:i].reshape(-1, 1)
        RRdm1 = np.repeat(dat.df.loc[1:(d + 1), 'R_obs'].values, self.K)[:i].reshape(-1, 1)

        if self.standardize:
            CC = (CC - self.C_shift) / self.C_scale
            MM = (MM - self.M_shift) / self.M_scale
            EEdm1 = (EEdm1 - self.E_shift) / self.E_scale

        ## predictors
        XX = np.hstack([np.ones((i, 1)), EEdm1, RRdm1, CC, AA, AA * EEdm1, AA * RRdm1, AA * CC])
        ## posterior variance
        self.post_beta_var = XX.T @ XX / self.sigma2 + LA.inv(self.prior_beta_var)
        self.post_beta_var = LA.inv(self.post_beta_var)
        ## posterior mean
        self.post_beta_mean = XX.T @ MM / self.sigma2
        self.post_beta_mean += LA.inv(self.prior_beta_var) @ self.prior_beta_mean
        self.post_beta_mean = (self.post_beta_var @ self.post_beta_mean)

    
    def choose_A(self, Edm1, Rdm1, Ch):
        if self.standardize:
            Ch = (Ch - self.C_shift) / self.C_scale
            Edm1 = (Edm1 - self.E_shift) / self.E_scale

        ## advantage function
        X = np.hstack([1, Edm1, Rdm1, Ch]).reshape(1, -1)
        post_trt_mean = (X @ self.post_beta_mean[-self.dAS:, :]).item()
        post_trt_var = (X @ self.post_beta_var[-self.dAS:, -self.dAS:] @ X.T).item()
        P = 1 - norm.cdf(0, loc=post_trt_mean, scale=np.sqrt(post_trt_var))
        A = rd.binomial(n=1, p=P, size=1)
        return A, P


# %%
