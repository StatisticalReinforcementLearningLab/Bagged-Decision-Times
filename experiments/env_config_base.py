# %% set up


import numpy as np
import numpy.linalg as LA
import json


# %% environment config


class EnvConfigBase:
    def __init__(self, userid, params_env_path, params_std_file):
        self.userid = userid

        self.nweek = 4 * 9 ## number of weeks in an episode
        self.W = 7 ## number of days in a week
        self.K = 5 ## number of decision points in a day
        self.D_warm = 7
        self.D = self.nweek * self.W

        self.dC = 1
        self.dP = 1
        self.dA = 1
        self.dM = 1
        self.dE = 1
        self.dR = 1
        self.dO = 1

        with open(params_std_file, 'r') as file:
            std_params = json.load(file)

        self.C_shift = std_params['C_shift']
        self.C_scale = std_params['C_scale']
        self.M_shift = std_params['M_shift']
        self.M_scale = std_params['M_scale']
        self.E_shift = std_params['E_shift']
        self.E_scale = std_params['E_scale']
        self.O_shift = std_params['O_shift']
        self.O_scale = std_params['O_scale']

        self.limits_C = std_params['C_limit']
        self.limits_M = std_params['M_limit']
        self.limits_E = std_params['E_limit']
        self.limits_R = std_params['R_limit']
        self.limits_O = std_params['O_limit']

        self.E0 = None
        self.R0 = None
        self.theta_C = None
        self.theta_M = None
        self.theta_E = None
        self.theta_R = None
        self.theta_O = None
        self.resid_C = None
        self.resid_M = None
        self.resid_E = None
        self.resid_R = None
        self.resid_O = None

