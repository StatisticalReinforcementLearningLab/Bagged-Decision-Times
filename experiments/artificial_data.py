# %% set up


import numpy as np
import numpy.random as rd
from env_config_base import EnvConfigBase
from dataset import Dataset


# %% artifical data


class ArtificialData:
    def __init__(self, params_art: dict, env_config: EnvConfigBase, L: int):
        if params_art:
            self.C = params_art["C"]
            self.E0 = params_art["E0"]
            self.R0 = params_art["R0"]

            self.prob = params_art["prob"]

            self.theta_M_mean = params_art["theta_M_mean"]
            self.theta_M_var = params_art["theta_M_var"]
            self.sigma2_M = params_art["sigma2_M"]

            self.theta_E_mean = params_art["theta_E_mean"]
            self.theta_E_var = params_art["theta_E_var"]
            self.sigma2_E = params_art["sigma2_E"]

            self.theta_R_mean = params_art["theta_R_mean"]
            self.theta_R_var = params_art["theta_R_var"]
            self.sigma2_R = params_art["sigma2_R"]

            self.theta_O_mean = params_art["theta_O_mean"]
            self.theta_O_var = params_art["theta_O_var"]
            self.sigma2_O = params_art["sigma2_O"]

            self.resid_M = params_art["resid_M"]
            self.resid_E = params_art["resid_E"]
            self.resid_R = params_art["resid_R"]
            self.resid_O = params_art["resid_O"]

        self.dC = env_config.dC
        self.dA = env_config.dA
        self.dM = env_config.dM
        self.dE = env_config.dE
        self.dR = env_config.dR
        self.dO = env_config.dO

        self.limits_M = env_config.limits_M
        self.limits_E = env_config.limits_E
        self.limits_R = env_config.limits_R
        self.limits_O = env_config.limits_O

        self.L = L
        self.K = env_config.K


    def draw_art_data(self, B: int, J: int, noise = False):
        size = (B, J)
        if J == 0: 
            art_CC = np.zeros((B, 0, self.L * self.K, self.dC))
            art_AA = np.zeros((B, 0, self.L * self.K, self.dA))
            art_MM = np.zeros((B, 0, self.L * self.K, self.dM))
            art_EE = np.zeros((B, 0, self.L + 1, self.dE))
            art_RR = np.zeros((B, 0, self.L + 1, self.dR))
            art_OO = np.zeros((B, 0, self.L, self.dO))
            return art_CC, art_AA, art_MM, art_EE, art_RR, art_OO

        seeds = rd.randint(low=1e3, high=1e6, size=4)
        dCM = (1 + self.dA) * (1 + self.dE + self.dR + self.dC) ## dim of conditional vars
        rng = rd.default_rng(seeds[0])
        theta_M_tilde = rng.multivariate_normal(
            self.theta_M_mean, self.theta_M_var, size ## for dM=1
        )
        theta_M_tilde = theta_M_tilde.reshape(size + (dCM, self.dM))


        dCE = 1 + self.dE + 2 * self.K * self.dA ## dim of conditional vars
        rng = rd.default_rng(seeds[1])
        theta_E_tilde = rng.multivariate_normal(
            self.theta_E_mean, self.theta_E_var, size ## for dE=1
        )
        theta_E_tilde = theta_E_tilde.reshape(size + (dCE, self.dE))


        dCR = 1 + self.K * self.dM + self.dE + self.dR ## dim of conditional vars
        rng = rd.default_rng(seeds[2])
        theta_R_tilde = rng.multivariate_normal(
            self.theta_R_mean, self.theta_R_var, size ## for dR=1
        )
        theta_R_tilde = theta_R_tilde.reshape(size + (dCR, self.dR))


        dCO = 1 + self.dR ## dim of conditional vars
        rng = rd.default_rng(seeds[3])
        theta_O_tilde = rng.multivariate_normal(
            self.theta_O_mean, self.theta_O_var, size ## for dO=1
        )
        theta_O_tilde = theta_O_tilde.reshape(size + (dCO, self.dO))


        ## dim -2 is used for matrix multiplication and data concatenation
        E0_tilde = rd.choice(self.E0, size=size + (1, self.dR), replace=True)
        R0_tilde = rd.choice(self.R0, size=size + (1, self.dR), replace=True)


        art_CC = []
        art_AA = []
        art_MM = []
        art_EE = [E0_tilde.copy()]
        art_RR = [R0_tilde.copy()]
        art_OO = []
        for l in range(0, self.L):
            E0_tilde = art_EE[-1]
            R0_tilde = art_RR[-1]
            C_tilde_l = []
            A_tilde_l = []
            M_tilde_l = []
            for k in range(0, self.K):
                C_tilde = rd.choice(
                    self.C, size=size + (1, self.dC)
                )
                
                A_tilde = rd.choice(
                    [0, 1], size=size + (1, self.dA), p=[1 - self.prob, self.prob]
                )

                M_cond = np.concatenate((
                    np.ones(size + (1, 1)), E0_tilde, R0_tilde, C_tilde,
                ), axis=-1) ## dim = B*J*1*dCM
                M_tilde_main = np.matmul(M_cond, theta_M_tilde[:, :, :4, :])
                M_tilde_trt = A_tilde * np.matmul(M_cond, theta_M_tilde[:, :, 4:, :])
                M_tilde_trt = np.maximum(M_tilde_trt, 0)
                M_tilde = M_tilde_main + M_tilde_trt
                if noise:
                    M_tilde += rd.choice(self.resid_M, size + (1, self.dM))
                M_tilde = np.clip(M_tilde, self.limits_M[0], self.limits_M[1])

                C_tilde_l.append(C_tilde.copy())
                A_tilde_l.append(A_tilde.copy())
                M_tilde_l.append(M_tilde.copy())
            
            C_tilde_l = np.concatenate(C_tilde_l, axis=-2)
            A_tilde_l = np.concatenate(A_tilde_l, axis=-2)
            M_tilde_l = np.concatenate(M_tilde_l, axis=-2)
            art_CC.append(C_tilde_l.copy())
            art_AA.append(A_tilde_l.copy())
            art_MM.append(M_tilde_l.copy())

            A_tilde_l = A_tilde_l.reshape(size + (1, self.K * self.dA))
            M_tilde_l = M_tilde_l.reshape(size + (1, self.K * self.dM))

            E_cond = np.concatenate(
                (np.ones(size + (1, 1)), E0_tilde, A_tilde_l, A_tilde_l * E0_tilde), axis=-1
            ) ## dim = B*J*1*dCE
            E_tilde = np.matmul(E_cond, theta_E_tilde) ## dim = B*J*1*dE
            if noise:
                E_tilde += rd.choice(self.resid_E, size + (1, self.dE))
            E_tilde = np.clip(E_tilde, self.limits_E[0], self.limits_E[1])
            art_EE.append(E_tilde.copy())

            R_cond = np.concatenate(
                (np.ones(size + (1, 1)), M_tilde_l, E_tilde, R0_tilde), axis=-1
            ) ## dim = B*J*1*dCR
            R_tilde = np.matmul(R_cond, theta_R_tilde) ## dim = B*J*1*dR
            if noise:
                R_tilde += rd.choice(self.resid_R, size + (1, self.dR))
            R_tilde = np.clip(R_tilde, self.limits_R[0], self.limits_R[1])
            art_RR.append(R_tilde.copy())

            O_cond = np.concatenate(
                (np.ones(size + (1, 1)), R0_tilde), axis=-1
            ) ## dim = B*J*1*dCO
            O_tilde = np.matmul(O_cond, theta_O_tilde) ## dim = B*J*1*dO
            if noise:
                O_tilde += rd.choice(self.resid_O, size + (1, self.dO))
            O_tilde = np.clip(O_tilde, self.limits_O[0], self.limits_O[1])
            art_OO.append(O_tilde.copy())

        art_CC = np.concatenate(art_CC, axis=-2)
        art_AA = np.concatenate(art_AA, axis=-2)
        art_MM = np.concatenate(art_MM, axis=-2)
        art_EE = np.concatenate(art_EE, axis=-2)
        art_RR = np.concatenate(art_RR, axis=-2)
        art_OO = np.concatenate(art_OO, axis=-2)
        return art_CC, art_AA, art_MM, art_EE, art_RR, art_OO


    def combine_dataset(self, dat: Dataset, d: int, J: int, noise = False):
        l = d // self.L
        ## artificial data
        art_CC, art_AA, art_MM, art_EE, art_RR, art_OO = self.draw_art_data(1, J, noise)
        ## combine the artificial sample with the observed sample
        b = 0
        comb_CC = np.concatenate([
            dat.df.loc[1:d, dat.col_C].values.reshape(d, self.K, self.dC), 
            art_CC[b].reshape(self.L * J, self.K, self.dC), 
        ], axis=0)
        comb_AA = np.concatenate([
            dat.df.loc[1:d, dat.col_A].values.reshape(d, self.K, self.dA), 
            art_AA[b].reshape(self.L * J, self.K, self.dA), 
        ], axis=0)
        comb_MM = np.concatenate([
            dat.df.loc[1:d, dat.col_M].values.reshape(d, self.K, self.dM), 
            art_MM[b].reshape(self.L * J, self.K, self.dM), 
        ], axis=0)
        comb_EEdm1 = np.concatenate([
            dat.df.loc[0:(d-1), 'E'].values.reshape(d, self.dE), 
            art_EE[b][:, :-1, :].reshape(self.L * J, self.dE), 
        ], axis=0)
        comb_EE = np.concatenate([
            dat.df.loc[1:d, 'E'].values.reshape(d, self.dE), 
            art_EE[b][:, 1:, :].reshape(self.L * J, self.dE), 
        ], axis=0)
        comb_RRdm1 = np.concatenate([
            dat.df.loc[0:(d-1), 'R_obs'].values.reshape(d, self.dR), 
            art_RR[b][:, :-1, :].reshape(self.L * J, self.dR), 
        ], axis=0)
        comb_RR = np.concatenate([
            dat.df.loc[1:d, 'R_obs'].values.reshape(d, self.dR), 
            art_RR[b][:, 1:, :].reshape(self.L * J, self.dR), 
        ], axis=0)
        comb_OO = np.concatenate([
            dat.df.loc[1:d, 'O'].values.reshape(d, self.dO), 
            art_OO[b].reshape(self.L * J, self.dO), 
        ], axis=0)
        comb_dat = {
            'comb_CC': comb_CC,
            'comb_AA': comb_AA,
            'comb_MM': comb_MM,
            'comb_EEdm1': comb_EEdm1,
            'comb_EE': comb_EE,
            'comb_RRdm1': comb_RRdm1,
            'comb_RR': comb_RR,
            'comb_OO': comb_OO
        }
        return comb_dat


# %%
