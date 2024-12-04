# %% set up


import numpy as np
import numpy.random as rd
from env_config_base import EnvConfigBase
from env_testbed_AR import Env
from dataset import Dataset


# %% 


class MRT:
    def __init__(self, config: EnvConfigBase, env: Env):
        self.config = config
        self.env = env
        self.userid = config.userid
        self.K = config.K

        self.dC = config.dC
        self.dP = config.dP
        self.dA = config.dA
        self.dM = config.dM
        self.dE = config.dE
        self.dR = config.dR
        self.dO = config.dO


    ## generate an offline dataset with fixed action probability
    def gen_data_fixed_prob(self, P_fixed, W, D):
        dat = Dataset(self.userid, self.K, D)
        E0 = self.env.E0
        R0 = self.env.R0
        dat.df.loc[0, 'E'] = E0
        dat.df.loc[0, 'R'] = R0
        dat.df.loc[0, 'R_obs'] = R0
        dat.df.loc[0, 'R_imp'] = R0
        for d in range(0, D):
            Edm1 = np.array([dat.df.loc[d, 'E']])
            Rdm1 = np.array([dat.df.loc[d, 'R']])
            Cd = np.zeros((self.K, self.dC))
            Pd = np.zeros((self.K, self.dP))
            Ad = np.zeros((self.K, self.dA)).astype(int)
            Md = np.zeros((self.K, self.dM))
            for k in range(0, self.K):
                Cd[k] = self.env.gen_Ch(d, k)
                Pd[k] = P_fixed
                Ad[k] = rd.choice([0, 1], size=1, p=[1 - Pd[k, 0], Pd[k, 0]])
                Md[k] = self.env.gen_Mh(Edm1, Rdm1, Cd[k], Ad[k], d, k)
            Ed = self.env.gen_Ed(Ad, Edm1, d)
            Rd = self.env.gen_Rd(Md, Ed, Rdm1, Ad, d)
            Od = self.env.gen_Od(Rdm1, d)
            if (d + 1) % W == 0:
                Rd_obs = Rd.copy()
            else:
                Rd_obs = np.array([np.nan])
            Rd_imp = Rd_obs.copy()
            dat.df.loc[d + 1, dat.col_C] = Cd.reshape(-1)
            dat.df.loc[d + 1, dat.col_P] = Pd.reshape(-1)
            dat.df.loc[d + 1, dat.col_A] = Ad.reshape(-1)
            dat.df.loc[d + 1, dat.col_M] = Md.reshape(-1)
            dat.df.loc[d + 1, 'E'] = Ed.reshape(-1)
            dat.df.loc[d + 1, 'R'] = Rd.reshape(-1)
            dat.df.loc[d + 1, 'R_obs'] = Rd_obs.reshape(-1)
            dat.df.loc[d + 1, 'R_imp'] = Rd_imp.reshape(-1)
            dat.df.loc[d + 1, 'O'] = Od.reshape(-1)
            dat.df.loc[d + 1, 'regret'] = 0
        return dat


# %%
