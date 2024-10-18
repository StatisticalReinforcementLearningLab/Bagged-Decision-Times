# %% set up


import numpy as np
import numpy.random as rd
import numpy.linalg as LA
from statsmodels.api import WLS
from env_config_base import EnvConfigBase
from env_testbed import Env
from artificial_data import ArtificialData


# %% SRLSVI


class SRLSVI():
    def __init__(self, env_config: EnvConfigBase, lam_beta: float, 
        sigma2=1, gamma=0.99, standardize=False
    ):
        self.K = env_config.K
        self.L = 1
        self.gamma = gamma
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
        self.dS = 1 + self.dE + self.dR
        self.dX = (2**self.K + 1) * self.dS

        ## prior mean = 0
        self.prior_beta_mean = np.zeros((self.dX, 1))
        ## prior variance = 1/lambda I
        self.prior_beta_var = 1/lam_beta * np.diag(np.ones(self.dX))
        self.sigma2 = sigma2 # noise variance

        # create space to store posterior distribution
        self.post_beta_mean = np.zeros((self.dX, 1))
        self.post_beta_var = np.zeros((self.dX, self.dX))
        self.beta_tilde = np.zeros((self.dX, 1))


    def get_Q(
        self, beta: np.ndarray, comb_dat: dict, ii: int
    ):
        comb_CC = comb_dat['comb_CC'].reshape(ii, self.L * self.K, self.dC)
        comb_AA = comb_dat['comb_AA'].reshape(ii, self.L * self.K, self.dA)
        comb_MM = comb_dat['comb_MM'].reshape(ii, self.L * self.K, self.dM)
        comb_EEdm1 = comb_dat['comb_EEdm1'].reshape(ii, self.L, self.dE)
        comb_EE = comb_dat['comb_EE'].reshape(ii, self.L, self.dE)
        comb_RRdm1 = comb_dat['comb_RRdm1'].reshape(ii, self.L, self.dR)
        comb_RR = comb_dat['comb_RR'].reshape(ii, self.L, self.dR)

        if self.standardize:
            comb_CC = (comb_CC - self.C_shift) / self.C_scale
            comb_MM = (comb_MM - self.M_shift) / self.M_scale
            comb_EEdm1 = (comb_EEdm1 - self.E_shift) / self.E_scale
            comb_EE = (comb_EE - self.E_shift) / self.E_scale

        ## next trajectory
        intercept = np.ones((ii, 1))
        comb_A0 = np.zeros((ii, 1))
        comb_A1 = np.ones((ii, 1))
        ## predictors
        comb_Edm1_next = comb_EE[:, -1, :].copy()
        comb_Rdm1_next = comb_RR[:, -1, :].copy()
        ## main effect
        comb_X_next_main = np.hstack([
            intercept, comb_Edm1_next, comb_Rdm1_next, 
        ])
        pred_Q_opt = comb_X_next_main @ beta[:self.dS, :]
        inter = []
        for a in range(2**self.K):
            inter_a = comb_X_next_main @ beta[(self.dS * (a + 1)):(self.dS * (a + 2)), :]
            inter.append(inter_a)
        inter = np.hstack(inter)
        pred_Q_opt += np.amax(inter, axis=1, keepdims=True)

        ## update beta
        comb_inter = []
        comb_R = comb_RR[:, 0, :].copy()
        comb_Edm1 = comb_EEdm1[:, 0, :].copy()
        comb_Rdm1 = comb_RRdm1[:, 0, :].copy()
        comb_main = np.hstack([
            intercept, comb_Edm1, comb_Rdm1, 
        ])
        comb_AA = comb_AA.astype(int)[:, :, 0] # remove the last dim
        comb_AA_order = [int(''.join(map(str, row)), 2) for row in comb_AA]
        comb_AA_order = np.array(comb_AA_order).reshape(-1, 1)
        comb_inter = []
        for a in range(2**self.K):
            comb_inter.append((comb_AA_order == a) * comb_main)
        ## concatenate main and interaction effect
        comb_X = np.hstack([comb_main] + comb_inter)
        ## response
        comb_Y = comb_R + self.gamma * pred_Q_opt

        ## posterior variance
        post_var = comb_X.T @ comb_X / self.sigma2 + LA.inv(self.prior_beta_var)
        ## round the inverse posterior variance for numerical stability
        ## otherwise, the inverse posterior variance may be asymmetric
        post_var = np.round(LA.inv(post_var), decimals=6)
        self.post_beta_var = post_var.copy()
        
        ## posterior mean
        post_mean = comb_X.T @ comb_Y / self.sigma2
        post_mean += LA.inv(self.prior_beta_var) @ self.prior_beta_mean
        post_mean = post_var @ post_mean
        self.post_beta_mean = post_mean.copy()

        ## sample from the posterior
        rng = rd.default_rng()
        beta_tilde = rng.multivariate_normal(
            self.post_beta_mean.reshape(-1), self.post_beta_var, 1
        ).reshape(-1, 1)
        self.beta_tilde = beta_tilde.copy()
        
        return self.beta_tilde


    def choose_A(
        self, Edm1: np.ndarray, Rdm1_imp: np.ndarray, 
    ):
        intercept = np.ones(1)

        if self.standardize:
            Edm1 = (Edm1 - self.E_shift) / self.E_scale

        ## main effect
        main = np.hstack([
            intercept, Edm1, Rdm1_imp
        ]).reshape(1, -1)
        inter = []
        for a in range(2**self.K):
            inter_a = main @ self.beta_tilde[(self.dS * (a + 1)):(self.dS * (a + 2)), :]
            inter.append(inter_a)
        inter = np.hstack(inter)
        pred_AA_order = np.argmax(inter, axis=1).item()
        pred_AA = list(np.binary_repr(pred_AA_order, width=self.K))
        pred_AA = np.array(pred_AA, dtype=int)

        return pred_AA


    def gen_offline_data(self, env: Env, art: ArtificialData, N: int):
        ## generate a large offline dataset
        ## use matrix calculation for better computational efficiency
        Ed = rd.choice(art.E0, size=(N, 1), replace=True)
        Rd = rd.choice(art.R0, size=(N, 1), replace=True)
        CC = []
        AA = []
        MM = []
        EE = [Ed]
        RR = [Rd]
        RR_mean = [Rd]
        for l in range(self.L):
            Edm1 = Ed.copy()
            Rdm1 = Rd.copy()
            Cd = []
            Ad = []
            Md = []
            for k in range(self.K):
                Ch = env.gen_Ch_mean(n=N).reshape(N, 1)
                Ch += rd.choice(env.resid_C, size=(N, 1), replace=True)
                Ch = np.clip(Ch, env.limits_C[0], env.limits_C[1])
                ## choose actions w.p. 0.5
                Ah = rd.choice([0, 1], size=(N, 1), replace=True)
                Mh = env.gen_Mh_mean(Edm1, Rdm1, Ch, Ah, n=N).reshape(N, 1)
                Mh += rd.choice(env.resid_obs_M, size=(N, 1), replace=True)
                Mh = np.clip(Mh, env.limits_M[0], env.limits_M[1])
                Cd.append(Ch)
                Ad.append(Ah)
                Md.append(Mh)
            Cd = np.hstack(Cd)
            Ad = np.hstack(Ad)
            Md = np.hstack(Md)
            Ed = env.gen_Ed_mean(Ad, Edm1, n=N).reshape(N, 1)
            Ed += rd.choice(env.resid_E, size=(N, 1), replace=True)
            Ed = np.clip(Ed, env.limits_E[0], env.limits_E[1])
            ## save the mean of Rd
            Rd_mean = env.gen_Rd_mean(Md, Ed, Rdm1, n=N).reshape(N, 1)
            Rd = Rd_mean + rd.choice(env.resid_R, size=(N, 1), replace=True)
            Rd = np.clip(Rd, env.limits_R[0], env.limits_R[1])
            CC.append(Cd)
            AA.append(Ad)
            MM.append(Md)
            EE.append(Ed)
            RR_mean.append(Rd_mean)
            RR.append(Rd)
        comb_dat = {
            'comb_CC': np.hstack(CC),
            'comb_AA': np.hstack(AA),
            'comb_MM': np.hstack(MM),
            'comb_EEdm1': np.hstack(EE[:-1]),
            'comb_EE': np.hstack(EE[1:]),
            'comb_RRdm1': np.hstack(RR[:-1]),
            ## use mean of Rd in the response to reduce noise
            'comb_RR': np.hstack(RR_mean[1:]),
        }
        return comb_dat
    
    
    def get_opt_Q(
        self, env: Env, art: ArtificialData, 
        N=100000, threshold=0.01, nitr=20
    ):
        ## generate a large offline dataset
        comb_dat = self.gen_offline_data(env, art, N)
        ## initialize
        beta = self.initialize(comb_dat, N, 0)
        ## iteratively improve beta through SRLSVI
        itr = 0
        beta_old = np.zeros((self.dX, 1))
        theta_R = np.zeros_like(env.theta_R) ## no reward engineering
        while LA.norm(beta - beta_old) > threshold and itr < nitr:
            beta_old = beta.copy()
            comb_dat = self.gen_offline_data(env, art, N)
            weights = np.ones(N)
            beta = self.get_Q(beta, comb_dat, weights, N, 0, theta_R)
            itr += 1
        return beta
        

# %%
