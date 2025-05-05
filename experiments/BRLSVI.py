# %% set up


import numpy as np
import numpy.random as rd
import numpy.linalg as LA
from statsmodels.api import WLS
from env_config_base import EnvConfigBase
from env_testbed import Env
from artificial_data import ArtificialData


# %% BRLSVI


class BRLSVI():
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
        self.dAC = 1 + self.dE + self.dR + self.dC
        self.dX = (
            2 + (self.K - 1) * self.dM + (self.K - 1) * self.dA 
            + 2 * self.dE + 2 * self.dR + self.dC
            + self.K * self.dA * self.dAC
        )

        ## prior mean = 0
        self.prior_beta_mean = np.zeros((self.dX, 1))
        ## prior variance = 1/lambda I
        self.prior_beta_var = 1/lam_beta * np.diag(np.ones(self.dX))
        self.sigma2 = sigma2 # noise variance

        # create space to store posterior distribution
        self.post_beta_mean = np.zeros((self.dX, 1))
        self.post_beta_var = np.zeros((self.dX, self.dX))
        self.beta_tilde = np.zeros((self.dX, 1))


    def initialize(self, comb_dat: dict, ii: int, lam_beta: float):
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

        intercept = np.ones((ii, 1))
        comb_A0 = np.zeros((ii, 1))
        comb_A1 = np.ones((ii, 1))
        pred_Q_opt = np.zeros((ii, self.dR))

        mod_X = []
        mod_Y = []
        for k in range(self.K - 1, -1, -1):
            comb_R_h = comb_RR[:, 0, :].copy() ## response
            comb_E_h = comb_EE[:, 0, :].copy()
            ## predictors
            comb_C_h = comb_CC[:, k, :].copy()
            comb_A_h = comb_AA[:, k, :].copy()
            comb_M_h = comb_MM[:, k, :].copy()
            comb_Edm1_h = comb_EEdm1[:, 0, :].copy()
            comb_Rdm1_h = comb_RRdm1[:, 0, :].copy()
            ## M before h
            comb_M_prev_h = comb_MM[:, :(self.K - 1), :].copy()
            comb_M_prev_h[:, k:, :] = 0
            comb_M_prev_h = comb_M_prev_h.reshape(ii, (self.K - 1) * self.dM)
            ## A before h
            comb_A_prev_h = comb_AA[:, :(self.K - 1), :].copy()
            comb_A_prev_h[:, k:, :] = 0
            comb_A_prev_h = comb_A_prev_h.reshape(ii, (self.K - 1) * self.dA)
            ## main effect
            comb_main_h = np.hstack([
                intercept, k * intercept, 
                comb_M_prev_h, comb_A_prev_h, 
                comb_Edm1_h, k * comb_Edm1_h, 
                comb_Rdm1_h, k * comb_Rdm1_h, 
                comb_C_h, 
            ])
            ## interaction effect
            comb_inter_h = np.zeros((ii, self.K * self.dAC))
            comb_inter_h[:, (k * self.dAC):((k + 1) * self.dAC)] = np.hstack([
                comb_A_h, comb_A_h * comb_Edm1_h, 
                comb_A_h * comb_Rdm1_h, comb_A_h * comb_C_h
            ])
            comb_inter_h_A0 = np.zeros((ii, self.K * self.dAC))
            comb_inter_h_A0[:, (k * self.dAC):((k + 1) * self.dAC)] = np.hstack([
                comb_A0, comb_A0 * comb_Edm1_h, 
                comb_A0 * comb_Rdm1_h, comb_A0 * comb_C_h
            ])
            comb_inter_h_A1 = np.zeros((ii, self.K * self.dAC))
            comb_inter_h_A1[:, (k * self.dAC):((k + 1) * self.dAC)] = np.hstack([
                comb_A1, comb_A1 * comb_Edm1_h, 
                comb_A1 * comb_Rdm1_h, comb_A1 * comb_C_h
            ])
            ## concatenate main and interaction effect
            comb_X_h = np.hstack([comb_main_h, comb_inter_h])
            comb_X_h_A0 = np.hstack([comb_main_h, comb_inter_h_A0])
            comb_X_h_A1 = np.hstack([comb_main_h, comb_inter_h_A1])
            ## response
            if k == self.K - 1:
                comb_Y_h = comb_R_h + self.gamma * pred_Q_opt
            else:
                comb_Y_h = pred_Q_opt
            mod_X.append(comb_X_h)
            mod_Y.append(comb_Y_h)

            ## weighted ridge regression
            mod_lm = WLS(np.vstack(mod_Y), np.vstack(mod_X), weights=1)
            mod_res = mod_lm.fit_regularized(
                method='elastic_net', alpha=lam_beta, L1_wt=0
            )
            beta = mod_res.params.reshape(-1, 1)

            ## optimal Q
            pred_Q_A0 = comb_X_h_A0 @ beta
            pred_Q_A1 = comb_X_h_A1 @ beta
            pred_Q_opt = np.maximum(pred_Q_A0, pred_Q_A1)

        ## weighted ridge regression
        ## combine the data from all h
        mod_X = np.vstack(mod_X)
        mod_Y = np.vstack(mod_Y)
        mod_lm = WLS(mod_Y, mod_X, weights=1)
        mod_res = mod_lm.fit_regularized(
            method='elastic_net', alpha=lam_beta, L1_wt=0
        )
        
        return mod_res.params.reshape(-1, 1)


    def get_Q(
        self, beta: np.ndarray, comb_dat: dict, ii: int, theta_R=np.zeros(8)
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
        comb_C_next = rd.choice(comb_CC.reshape(-1), size=(ii, 1), replace=True)
        comb_M_next = np.zeros((ii, (self.K - 1) * self.dM))
        comb_A_next = np.zeros((ii, (self.K - 1) * self.dA))
        ## main effect
        comb_X_next_main = np.hstack([
            intercept, 0 * intercept, 
            comb_M_next, comb_A_next, 
            comb_Edm1_next, 0 * comb_Edm1_next, 
            comb_Rdm1_next, 0 * comb_Rdm1_next, 
            comb_C_next, 
        ])
        ## concatenate main and interaction effect
        comb_X_h_A0 = np.hstack([
            comb_X_next_main, 
            comb_A0, comb_A0 * comb_Edm1_next, 
            comb_A0 * comb_Rdm1_next, comb_A0 * comb_C_next, 
            np.zeros((ii, (self.K - 1) * self.dAC)), 
        ])
        comb_X_h_A1 = np.hstack([
            comb_X_next_main, 
            comb_A1, comb_A1 * comb_Edm1_next, 
            comb_A1 * comb_Rdm1_next, comb_A1 * comb_C_next, 
            np.zeros((ii, (self.K - 1) * self.dAC)), 
        ])
        ## optimal Q
        pred_Q_A0 = comb_X_h_A0 @ beta
        pred_Q_A1 = comb_X_h_A1 @ beta
        pred_Q_opt = np.maximum(pred_Q_A0, pred_Q_A1)

        mod_X = []
        mod_Y = []
        for k in range(self.K - 1, -1, -1):
            comb_R_h = comb_RR[:, 0, :].copy() ## response
            comb_E_h = comb_EE[:, 0, :].copy()
            ## predictors
            comb_C_h = comb_CC[:, k, :].copy()
            comb_A_h = comb_AA[:, k, :].copy()
            comb_M_h = comb_MM[:, k, :].copy()
            comb_Edm1_h = comb_EEdm1[:, 0, :].copy()
            comb_Rdm1_h = comb_RRdm1[:, 0, :].copy()
            ## M before h
            comb_M_prev_h = comb_MM[:, :(self.K - 1), :].copy()
            comb_M_prev_h[:, k:, :] = 0
            comb_M_prev_h = comb_M_prev_h.reshape(ii, (self.K - 1) * self.dM)
            ## A before h
            comb_A_prev_h = comb_AA[:, :(self.K - 1), :].copy()
            comb_A_prev_h[:, k:, :] = 0
            comb_A_prev_h = comb_A_prev_h.reshape(ii, (self.K - 1) * self.dA)
            ## main effect
            comb_main_h = np.hstack([
                intercept, k * intercept, 
                comb_M_prev_h, comb_A_prev_h, 
                comb_Edm1_h, k * comb_Edm1_h, 
                comb_Rdm1_h, k * comb_Rdm1_h, 
                comb_C_h, 
            ])
            ## interaction effect
            comb_inter_h = np.zeros((ii, self.K * self.dAC))
            comb_inter_h[:, (k * self.dAC):((k + 1) * self.dAC)] = np.hstack([
                comb_A_h, comb_A_h * comb_Edm1_h, 
                comb_A_h * comb_Rdm1_h, comb_A_h * comb_C_h
            ])
            comb_inter_h_A0 = np.zeros((ii, self.K * self.dAC))
            comb_inter_h_A0[:, (k * self.dAC):((k + 1) * self.dAC)] = np.hstack([
                comb_A0, comb_A0 * comb_Edm1_h, 
                comb_A0 * comb_Rdm1_h, comb_A0 * comb_C_h
            ])
            comb_inter_h_A1 = np.zeros((ii, self.K * self.dAC))
            comb_inter_h_A1[:, (k * self.dAC):((k + 1) * self.dAC)] = np.hstack([
                comb_A1, comb_A1 * comb_Edm1_h, 
                comb_A1 * comb_Rdm1_h, comb_A1 * comb_C_h
            ])
            ## concatenate main and interaction effect
            comb_X_h = np.hstack([comb_main_h, comb_inter_h])
            comb_X_h_A0 = np.hstack([comb_main_h, comb_inter_h_A0])
            comb_X_h_A1 = np.hstack([comb_main_h, comb_inter_h_A1])
            ## response
            if k == self.K - 1:
                comb_Y_h = comb_R_h + self.gamma * pred_Q_opt + theta_R[self.K + 1] * comb_E_h
                for kk in range(self.K):
                    comb_Y_h +=  theta_R[kk + 1] * comb_MM[:, kk, :]
            else:
                comb_Y_h = pred_Q_opt.copy()
            mod_X.append(comb_X_h)
            mod_Y.append(comb_Y_h)

            ## optimal Q
            pred_Q_A0 = comb_X_h_A0 @ beta
            pred_Q_A1 = comb_X_h_A1 @ beta
            pred_Q_opt = np.maximum(pred_Q_A0, pred_Q_A1)

        ## Bayesian linear regression
        mod_X = np.vstack(mod_X)
        mod_Y = np.vstack(mod_Y)

        ## posterior variance
        post_var = mod_X.T @ mod_X / self.sigma2 + mod_X.shape[0] * LA.inv(self.prior_beta_var)
        ## round the inverse posterior variance for numerical stability
        ## otherwise, the inverse posterior variance may be asymmetric
        post_var = np.round(LA.inv(post_var), decimals=6)
        self.post_beta_var = post_var.copy()
        
        ## posterior mean
        post_mean = mod_X.T @ mod_Y / self.sigma2
        post_mean += mod_X.shape[0] * LA.inv(self.prior_beta_var) @ self.prior_beta_mean
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
        self, betas: np.ndarray, Edm1: np.ndarray, Rdm1_imp: np.ndarray, 
        Cd: np.ndarray, Ad: np.ndarray, Md: np.ndarray, l: int, k: int
    ):
        intercept, A0, A1 = np.ones(1), np.zeros(1), np.ones(1)
        ## M before h
        M_prev_h = Md[:(self.K - 1)]
        M_prev_h[k:] = 0
        M_prev_h = M_prev_h.reshape(-1)
        ## A before h
        A_prev_h = Ad[:(self.K - 1)]
        A_prev_h[k:] = 0
        A_prev_h = A_prev_h.reshape(-1)

        if self.standardize:
            Cd = (Cd - self.C_shift) / self.C_scale
            M_prev_h = (M_prev_h - self.M_shift) / self.M_scale
            Edm1 = (Edm1 - self.E_shift) / self.E_scale

        ## main effect
        main = np.hstack([
            intercept, k * intercept, 
            M_prev_h, A_prev_h, 
            Edm1, k * Edm1, 
            Rdm1_imp, k * Rdm1_imp, 
            Cd[k],
        ])
        ## interaction effect
        inter_A0 = np.zeros(self.K * self.dAC)
        inter_A0[(k * self.dAC):((k + 1) * self.dAC)] = np.hstack([
            A0, A0 * Edm1, A0 * Rdm1_imp, A0 * Cd[k]
        ])
        inter_A1 = np.zeros(self.K * self.dAC)
        inter_A1[(k * self.dAC):((k + 1) * self.dAC)] = np.hstack([
            A1, A1 * Edm1, A1 * Rdm1_imp, A1 * Cd[k]
        ])
        ## concatenate main and interaction effect
        X_A0 = np.hstack([main, inter_A0]).reshape(1, -1)
        X_A1 = np.hstack([main, inter_A1]).reshape(1, -1)

        ## greedy action
        pred_Q0 = X_A0 @ betas
        pred_Q1 = X_A1 @ betas
        pred_AA = (pred_Q1 >= pred_Q0).astype(int).reshape(-1)
        return pred_AA[0]


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
        ## iteratively improve beta through BRLSVI
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
