# %% set up


import numpy as np
import numpy.linalg as LA
import json


# %% environment config


class EnvConfigBase:
    def __init__(self, userid, params_env_path, params_std_file):
        self.userid = userid

        self.nweek = 4 * 9 * 7 ## number of weeks in an episode
        self.W = 1 ## number of days in a week
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


    def get_Phi_C(self, theta_M, theta_E, theta_R, C):
        Phi1 = np.zeros((2, 2))
        Phi1[0, 0] = np.sum(theta_R[1:6]) * np.sum(theta_M[[2, 6]]) + theta_R[7]
        Phi1[0, 1] = np.sum(theta_R[1:6]) * np.sum(theta_M[[1, 5]]) + theta_R[6]
        Phi1[1, 1] = theta_E[6]
        Phi0 = np.zeros((2, 2))
        Phi0[0, 0] = np.sum(theta_R[1:6]) * np.sum(theta_M[2]) + theta_R[7]
        Phi0[0, 1] = np.sum(theta_R[1:6]) * np.sum(theta_M[1]) + theta_R[6]
        Phi0[1, 1] = theta_E[6]
        C1 = np.zeros(2)
        C1[0] = (
            np.sum(theta_R[1:6]) * np.sum(theta_M[[0, 4]]) + theta_R[0] +
            np.sum(theta_R[1:6] * np.mean(C, axis=0)) * np.sum(theta_M[[3, 7]])
        )
        C1[1] = theta_E[0] + np.sum(theta_E[2:7])
        C0 = np.zeros(2)
        C0[0] = (
            np.sum(theta_R[1:6]) * np.sum(theta_M[0]) + theta_R[0] +
            np.sum(theta_R[1:6] * np.mean(C, axis=0)) * np.sum(theta_M[3])
        )
        C0[1] = theta_E[0]
        return Phi1, Phi0, C1, C0


    def get_shift(self, shift0, theta_M, theta_E, theta_R, C):
        shift_M1 = ((0.99 - theta_R[7]) / np.sum(theta_M[[2, 6]]) - np.sum(theta_R[1:6])) / self.K
        shift_M0 = ((0.99 - theta_R[7]) / np.sum(theta_M[2]) - np.sum(theta_R[1:6])) / self.K
        shift_M_all = np.array([shift_M1, shift_M0])
        shift_M_all = shift_M_all[shift_M_all >= 0]
        shift_M = np.min(np.hstack([shift_M_all, shift0]))
        theta_R[1:6] += shift_M

        Phi1, Phi0, C1, C0 = self.get_Phi_C(theta_M, theta_E, theta_R, C)
        R_lim, E_lim = LA.inv(np.eye(2) - Phi1) @ C1
        alpha = np.array([3, 3])
        alpha[1] = E_lim

        shift_R1 = (alpha[0] - C1[0] - alpha[1] * Phi1[0, 1]) / alpha[0] - Phi1[0, 0]
        shift_R0 = (alpha[0] - C0[0] - alpha[1] * Phi0[0, 1]) / alpha[0] - Phi0[0, 0]
        shift_R_all = np.array([shift_R1, shift_R0])
        shift_R_all = shift_R_all[shift_R_all <= 0]
        shift_R = np.min(np.hstack([shift_R_all, 0]))
        theta_R[7] += shift_R

        return shift_M, shift_R


    def get_shift_user(self, shift0):
        params_env = self.get_params_env_dict()
        return self.get_shift(
            shift0, 
            params_env['theta_M'], params_env['theta_E'], 
            params_env['theta_R'], params_env['resid_C']
        )

