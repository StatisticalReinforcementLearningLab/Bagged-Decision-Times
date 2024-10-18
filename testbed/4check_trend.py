# %% set up


import numpy as np
import numpy.random as rd
import pandas as pd
from env_config import EnvConfig
from env_testbed import Env
from mrt import MRT
import matplotlib.pyplot as plt
import json


# %% parameters


W = 4 * 9 ## number of weeks in an episode
D = 7 ## number of days in a week
K = 5 ## number of decision points in a day
nday = W * D
n_jobs = 50 # parallel computing
seed = 2023

# dimensions
dC, dP, dA, dM, dE, dR, dO = 1, 1, 1, 1, 1, 1, 1
var_dim = {
    'dC': dC,
    'dA': dA,
    'dM': dM,
    'dE': dE,
    'dR': dR,
    'dO': dO,
}

dropbox = ""
folder_dat = dropbox + "Data/"
aim = '2'
folder_params = 'params_env_V' + aim + '/'
folder_figure = 'Figures/'
file_dat = folder_dat + 'aim-' + aim + '_with_R.csv'
file_params_env_prefix = folder_params + 'params_env_'
file_params_std = 'params_std_V' + aim + '.json'

userid_all = np.loadtxt(folder_params + 'user_ids.txt', dtype=int)


# %% generate data for 9 months


userid_plot = userid_all[:4]
rd.seed(seed)
M_all = {}
R_all = {}
E_all = {}
O_all = {}
for userid in userid_plot:
    config = EnvConfig(
        userid, folder_params, file_params_std,
    )
    env = Env(config)
    mrt = MRT(config, env)
    dat = mrt.gen_data_fixed_prob(0.5, config.W, config.D)
    M_all[userid] = dat.df.loc[:, dat.col_M].values.reshape(-1)
    R_all[userid] = dat.df.loc[:, 'R'].values.reshape(-1)
    E_all[userid] = dat.df.loc[:, 'E'].values.reshape(-1)
    O_all[userid] = dat.df.loc[:, 'O'].values.reshape(-1)


# %% standardization


with open(file_params_std, 'r') as file:
    std_params = json.load(file)

C_shift = std_params['C_shift']
C_scale = std_params['C_scale']
M_shift = std_params['M_shift']
M_scale = std_params['M_scale']
E_shift = std_params['E_shift']
E_scale = std_params['E_scale']
O_shift = std_params['O_shift']
O_scale = std_params['O_scale']

dat_org = pd.read_csv(file_dat)
dat_org['C'] = (dat_org['C'] - C_shift) / C_scale
dat_org['M'] = (dat_org['M'] - M_shift) / M_scale
dat_org['E'] = (dat_org['E'] - E_shift) / E_scale
dat_org['E0'] = (dat_org['E0'] - E_shift) / E_scale
dat_org['O'] = (dat_org['O'] - O_shift) / O_scale


# %% plots parameters


fig_size = (4, 2.8)
ylim_M = [-3, 3]
ylim_R = [-3, 3]
ylim_E = [-2, 4]
ylim_O = [-2, 4]

colors = ['#A40407', '#53446B', '#F16C93', '#C0AECE']
alpha = 0.7


# %% original data


## plot M
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    plt.plot(
        dat_org.loc[dat_org['id'] == userid, 'M'].values, 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_M)
plt.xlabel("decision time")
plt.ylabel("M")
# plt.title("original 3-month episode")
plt.savefig(folder_figure + 'trend_M_original.pdf', bbox_inches="tight")
plt.show()


## plot R
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    plt.plot(
        dat_org.loc[(dat_org['id'] == userid) & (dat_org['time'] == 1), 'R'].values, 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_R)
plt.xlabel("day")
plt.ylabel("R")
# plt.title("original 3-month episode")
plt.savefig(folder_figure + 'trend_R_original.pdf', bbox_inches="tight")
plt.show()


## plot E
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    plt.plot(
        dat_org.loc[(dat_org['id'] == userid) & (dat_org['time'] == 1), 'E'].values, 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_E)
plt.xlabel("day")
plt.ylabel("E")
# plt.title("original 3-month episode")
plt.savefig(folder_figure + 'trend_E_original.pdf', bbox_inches="tight")
plt.show()


## plot O
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    plt.plot(
        dat_org.loc[(dat_org['id'] == userid) & (dat_org['time'] == 1), 'O'].values, 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_O)
plt.xlabel("day")
plt.ylabel("O")
# plt.title("original 3-month episode")
plt.savefig(folder_figure + 'trend_O_original.pdf', bbox_inches="tight")
plt.show()


# %% generated data truncated to 3 months


## plot M
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    max_day = np.max(dat_org.loc[dat_org['id'] == userid, 'day'])
    plt.plot(
        M_all[userid][:(max_day * K)], 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_M)
plt.xlabel("decision time")
plt.ylabel("M")
# plt.title("generated 3-month epsiode")
plt.savefig(folder_figure + 'trend_M_generated_truncated.pdf', bbox_inches="tight")
plt.show()


## plot R
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    max_day = np.max(dat_org.loc[dat_org['id'] == userid, 'day'])
    plt.plot(
        R_all[userid][:max_day], 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_R)
plt.xlabel("day")
plt.ylabel("R")
# plt.title("generated 3-month epsiode")
plt.savefig(folder_figure + 'trend_R_generated_truncated.pdf', bbox_inches="tight")
plt.show()


## plot E
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    max_day = np.max(dat_org.loc[dat_org['id'] == userid, 'day'])
    plt.plot(
        E_all[userid][:max_day], 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_E)
plt.xlabel("day")
plt.ylabel("E")
# plt.title("generated 3-month epsiode")
plt.savefig(folder_figure + 'trend_E_generated_truncated.pdf', bbox_inches="tight")
plt.show()


## plot O
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    max_day = np.max(dat_org.loc[dat_org['id'] == userid, 'day'])
    plt.plot(
        O_all[userid][:max_day], 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_O)
plt.xlabel("day")
plt.ylabel("O")
# plt.title("generated 3-month epsiode")
plt.savefig(folder_figure + 'trend_O_generated_truncated.pdf', bbox_inches="tight")
plt.show()


# %% generated data


## plot M
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    plt.plot(
        M_all[userid], 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_M)
plt.xlabel("decision time")
plt.ylabel("M")
# plt.title("generated 9-month episode")
plt.savefig(folder_figure + 'trend_M_generated.pdf', bbox_inches="tight")
plt.show()


## plot R
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    plt.plot(
        R_all[userid], 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_R)
plt.xlabel("day")
plt.ylabel("R")
# plt.title("generated 9-month episode")
plt.savefig(folder_figure + 'trend_R_generated.pdf', bbox_inches="tight")
plt.show()


## plot E
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    plt.plot(
        E_all[userid], 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_E)
plt.xlabel("day")
plt.ylabel("E")
# plt.title("generated 9-month episode")
plt.savefig(folder_figure + 'trend_E_generated.pdf', bbox_inches="tight")
plt.show()


## plot O
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
for i, userid in enumerate(userid_plot):
    plt.plot(
        O_all[userid], 
        label=userid, color=colors[i], alpha=alpha
    )
plt.legend(loc='upper right')
plt.ylim(ylim_O)
plt.xlabel("day")
plt.ylabel("O")
# plt.title("generated 9-month episode")
plt.savefig(folder_figure + 'trend_O_generated.pdf', bbox_inches="tight")
plt.show()


# %%
