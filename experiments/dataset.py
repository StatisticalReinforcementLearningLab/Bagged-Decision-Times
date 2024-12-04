# %% set up


import numpy as np
import pandas as pd


# %% 


class Dataset:
    def __init__(self, userid, K, D):
        self.userid = userid
        self.K = K
        self.D = D

        self.col_C = ['C' + str(k + 1) for k in range(self.K)]
        self.col_P = ['P' + str(k + 1) for k in range(self.K)]
        self.col_A = ['A' + str(k + 1) for k in range(self.K)]
        self.col_M = ['M' + str(k + 1) for k in range(self.K)]
        self.col_opt_A = ['opt_A' + str(k + 1) for k in range(self.K)]
        self.col_mse = ['mse' + str(k + 1) for k in range(self.K)]
        self.dat_columns = ['userid', 'd'] + \
            self.col_C + self.col_P + self.col_A + self.col_M + \
            ['E', 'R', 'R_obs', 'R_imp', 'O', 'regret', 'R0']
        self.var_columns = {
            'col_C': self.col_C, 
            'col_P': self.col_P, 
            'col_A': self.col_A, 
            'col_M': self.col_M, 
        }

        self.df = pd.DataFrame(
            np.zeros((self.D + 1, 2 + 4 * self.K + 7)), 
            columns=self.dat_columns
        )
        self.df['userid'] = self.userid
        self.df['d'] = np.arange(0, self.D + 1)

