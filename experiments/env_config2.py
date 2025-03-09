# %% set up


import numpy as np
from env_config_base import EnvConfigBase
import json


# %% environment config


class EnvConfig(EnvConfigBase):
    def __init__(
        self, userid, params_env_path, params_std_file, 
    ):
        EnvConfigBase.__init__(self, userid, params_env_path, params_std_file)
        
        file_params_env = params_env_path + 'params_env_' + str(userid) + '.json'
        with open(file_params_env, 'r') as file:
            params_env = json.load(file)
        for key, value in params_env.items():
            params_env[key] = np.array(value)
        params_env['theta_R'][1:6] += 0.03
        params_env['theta_R'][1:6] = np.maximum(params_env['theta_R'][1:6], 0)
        params_env['theta_M'][1:3] = np.maximum(params_env['theta_M'][1:3], 0)
        params_env['theta_R'][6] += 0
        params_env['theta_E'][2:7] += 0

        self.E0 = params_env["E0"]
        self.R0 = params_env["R0"]
        self.theta_C = params_env["theta_C"]
        self.theta_M = params_env["theta_M"]
        self.theta_E = params_env["theta_E"]
        self.theta_R = params_env["theta_R"]
        self.theta_O = params_env["theta_O"]
        self.resid_C = params_env["resid_C"]
        self.resid_M = params_env["resid_M"]
        self.resid_E = params_env["resid_E"]
        self.resid_R = params_env["resid_R"]
        self.resid_O = params_env["resid_O"]

