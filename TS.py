# %% set up


import numpy as np
import numpy.random as rd
import numpy.linalg as LA
from scipy.stats import norm
from env_config_base import EnvConfigBase
from dataset import Dataset


# %% TS


class TS():
    def __init__(self, params_prior: dict, config: EnvConfigBase, standardize=False):
        self.K = config.K
        self.standardize = standardize

        self.C_shift = config.C_shift
        self.C_scale = config.C_scale
        self.M_shift = config.M_shift
        self.M_scale = config.M_scale

        self.dC = config.dC
        self.dA = config.dA
        self.dM = config.dM

        self.prior_beta_mean = params_prior["theta_M_mean"].reshape(-1, 1) # prior mean
        self.prior_beta_var = params_prior["theta_M_var"] # prior variance
        self.sigma2 = params_prior["sigma2_M"] # noise variance


    def update(self, dat: Dataset, d: int, k: int):
        i = d * self.K + k # total number of decision times
        CC = dat.df.loc[1:(d + 1), dat.col_C].values.reshape(-1)[:i].reshape(-1, 1)
        AA = dat.df.loc[1:(d + 1), dat.col_A].values.reshape(-1)[:i].reshape(-1, 1)
        MM = dat.df.loc[1:(d + 1), dat.col_M].values.reshape(-1)[:i].reshape(-1, 1)

        if self.standardize:
            CC = (CC - self.C_shift) / self.C_scale
            MM = (MM - self.M_shift) / self.M_scale

        ## predictors
        XX = np.hstack([np.ones((i, 1)), CC, AA, AA * CC])
        ## posterior variance
        self.post_beta_var = XX.T @ XX / self.sigma2 + LA.inv(self.prior_beta_var)
        self.post_beta_var = LA.inv(self.post_beta_var)
        ## posterior mean
        self.post_beta_mean = XX.T @ MM / self.sigma2
        self.post_beta_mean += LA.inv(self.prior_beta_var) @ self.prior_beta_mean
        self.post_beta_mean = (self.post_beta_var @ self.post_beta_mean)

    
    def choose_A(self, Ch):
        if self.standardize:
            Ch = (Ch - self.C_shift) / self.C_scale

        ## advantage function
        X = np.hstack([1, Ch]).reshape(1, -1)
        post_trt_mean = (X @ self.post_beta_mean[-2:, :]).item()
        post_trt_var = (X @ self.post_beta_var[-2:, -2:] @ X.T).item()
        P = 1 - norm.cdf(0, loc=post_trt_mean, scale=np.sqrt(post_trt_var))
        A = rd.binomial(n=1, p=P, size=1)
        return A, P


# %%
