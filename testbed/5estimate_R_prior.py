# %% set up


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssm
np.set_printoptions(suppress=True)


# %% read data


# dropbox = "/Users/daiqigao/Dropbox (Harvard University)/"
dropbox = "/Users/dqgao/Library/CloudStorage/Dropbox-HarvardUniversity/"
folder_dat = dropbox + "*Shared/HeartStepsV2V3/Daiqi/Data/"
folder_figure = 'Figures/'
aim = '3'
file_dat = folder_dat + 'aim-' + aim + '_with_R.csv'
dat_all = pd.read_csv(folder_dat + 'aim-' + aim + '.csv')


n = len(pd.unique(dat_all['userid']))
K = 5
D = 7
dat = dat_all[[
    'userid', 'day', 'date', 'decision.time', 'availability', 'logpresteps', 
    'action', 'reward', 'app.view', 'unprompted.bouts', 'fitbit.worn'
]]
dat.columns = ['id', 'day', 'date', 'time', 'I', 'C', 'A', 'M', 'E', 'O', 'fitbit_worn']
userid_all = pd.unique(dat['id'])
dC, dP, dA, dM, dE, dR, dO = 1, 1, 1, 1, 1, 1, 1 # dimensions


# %% remove data after dropout, standardization, missing data


dat_org = []
dat_std = []
dat_std_comp = []
dropout_summary = []

for userid in userid_all:
    idx = (dat['id'] == userid)
    dat_user1 = dat.loc[idx, :]
    dat_user1['E'] = dat_user1['E'].astype(float)

    ## dropout: the last day a user wore the Fitbit
    dropout_idx = np.max(np.nonzero(dat_user1.loc[dat_user1['time'] == 1, 'fitbit_worn']))
    dropout_day = dat_user1['day'].iloc[K * dropout_idx] ## index to day
    start_date = dat_user1.loc[(dat_user1['day'] == 1) & (dat_user1['time'] == 1), 'date'].values[0]
    dropout_date = dat_user1['date'].iloc[K * dropout_idx]
    dropout_summary.append([
        userid, start_date, dropout_date, 
        dropout_day, np.max(dat_user1['day']) - dropout_day
    ])
    dat_user1 = dat_user1[dat_user1['day'] <= dropout_day]

    ## smooth E
    E = dat_user1.loc[dat_user1['time'] == 1, 'E'].copy()
    E = E.shift(periods=np.arange(D).tolist(), fill_value=0).values
    E_ewm = 0
    w = (D - 1) / D
    for d in range(D):
        E_ewm += w**d * E[:, d]
    E_ewm *= (1 - w) / (1 - w**D)
    E_ewm = np.sqrt(E_ewm) ## sqrt of average app view
    E_init = E_ewm[D - 1] ## save initial E

    ## after study starts
    dat_user1 = dat_user1[dat_user1['day'] > 0]
    dat_user1['E'] = np.repeat(E_ewm[D:], K)

    ## find Edm1
    day_max = np.max(dat_user1['day'])
    dat_user1['E0'] = 0.0
    dat_user1.loc[dat_user1['day'] > 1, 'E0'] = \
        dat_user1.loc[dat_user1['day'] < day_max, 'E'].tolist()
    dat_user1.loc[dat_user1['day'] == 1, 'E0'] = E_init
    
    ## smooth O
    O_ewm = dat_user1.loc[dat_user1['time'] == 1, 'O'].copy()
    O_ewm = O_ewm.ewm(alpha=0.5, adjust=False).mean()
    dat_user1.loc[:, 'O_ewm'] = O_ewm.repeat(K).values
 
    ## standardization
    dat_user2 = dat_user1.copy()
    dat_user2['M'] = (dat_user2['M'] - dat_user2['M'].mean()) / dat_user2['M'].std()
    dat_user2['C'] = (dat_user2['C'] - dat_user2['C'].mean()) / dat_user2['C'].std()
    if dat_user2['E'].std() > 0:
        dat_user2['E'] = (dat_user2['E'] - dat_user2['E'].mean()) / dat_user2['E'].std()
        dat_user2['E0'] = (dat_user2['E0'] - dat_user2['E0'].mean()) / dat_user2['E0'].std()
    dat_user2['O'] = (dat_user2['O'] - dat_user2['O'].mean()) / dat_user2['O'].std()
    dat_user2['O_ewm'] = (
        dat_user2['O_ewm'] - dat_user2['O_ewm'].mean()
    ) / dat_user2['O_ewm'].std()

    ## complete data
    M = dat_user1['M'].to_numpy().reshape(-1, K)
    day_avail = (np.sum(np.isnan(M), axis=1) == 0)
    idx_avail = dat_user1.loc[dat_user1['time'] == 1, 'day'][day_avail]
    idx_avail = np.sort(np.hstack([idx_avail.index + j for j in range(K)]))
    dat_user3 = dat_user2.loc[idx_avail, :]
    
    dat_org.append(dat_user1)
    dat_std.append(dat_user2)
    dat_std_comp.append(dat_user3)


dat_org = pd.concat(dat_org)
dat_std = pd.concat(dat_std)
dat_std_comp = pd.concat(dat_std_comp)
print([dat_org.shape[0] / K, dat_std_comp.shape[0] / K])


## save the dropout statistics
# dropout_summary = pd.DataFrame(
#     dropout_summary,
#     columns=['id', 'study start date', 'dropout date', 'in study', 'dropout']
# )
# dropout_summary.set_index('id', inplace=True)
# dropout_summary.to_csv(folder_dat + 'dropout.csv')


# %% plot missing M


fig_size = (4, 2.8)
color = '#53446B'
alpha = 0.7

## plot decision times with missing M
dat_plot_M_1 = dat_org[['id', 'M']]
dat_plot_M_1['M_na'] = pd.isna(dat_plot_M_1['M'])
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(dat_plot_M_1.groupby('id')['M_na'].sum(), color=color, alpha=alpha)
plt.title("decision times with missing M")
plt.ylabel("frequency")
plt.savefig(folder_figure + 'hist_M_missing_h_V' + aim + '.pdf')
plt.show()


## plot days with missing M
dat_plot_M_21 = dat_org.groupby('id').size() / K
dat_plot_M_22 = dat_std_comp.groupby('id').size() / K
dat_plot_M_2 = pd.concat([dat_plot_M_21, dat_plot_M_22], axis=1, join='outer')
dat_plot_M_2 = dat_plot_M_2.fillna(0)
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(dat_plot_M_2.loc[:, 0] - dat_plot_M_2.loc[:, 1], color=color, alpha=alpha)
plt.title("days with missing M")
plt.ylabel("frequency")
plt.savefig(folder_figure + 'hist_M_missing_d_V' + aim + '.pdf')
plt.show()


# %% initialization


# Set the parameters of the HMM
obs_dim = 1    # emission dimension
state_dim = 1   # latent state dimension
input_dim = 6    # input dimension
N_iters = 10

O_comp = dat_std_comp['O'].to_numpy().reshape(-1, K)[:, [0]]
## E_{d} + M_{d, 1:K} -> R_{d} -> O_{d+1}
obs = []
for userid in userid_all:
    O = dat_std_comp.loc[dat_std_comp['id'] == userid, 'O_ewm']
    O = O.to_numpy().reshape(-1, K)[:, [0]]
    if O.shape[0] > 0:
        O = np.insert(O[1:], O.shape[0] - 1, O[-1], axis=0)
    obs.append(O)
obs = np.vstack(obs)
M = dat_std_comp['M'].to_numpy().reshape(-1, K)
Ed = dat_std_comp['E'].to_numpy().reshape(-1, K)[:, [0]]
inpt = np.hstack([M, Ed])
np.random.seed(2024)

## https://github.com/lindermanlab/ssm/blob/master/ssm/observations.py
## line 1158-1163 contains randomness in M step
## the source code of the lds.py file in the SSM package has been modified
lds = ssm.LDS(
    N=obs_dim, D=state_dim, M=input_dim, dynamics_kwargs={"l2_penalty_b": 1e8}, 
    transitions="inputdriven", dynamics="gaussian", emissions="gaussian"
)
lds_lps, lds_q = lds.fit(
    obs, inpt, 
    method="laplace_em", num_iters=N_iters, initialize=True
)


plt.figure(figsize=(6, 4))
plt.plot(lds_lps, label="Laplace-EM")
plt.xlim(0, N_iters)
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("Convergence of Laplace-EM Algorithm")
plt.savefig(folder_figure + 'LDS_initilization_EM_V' + aim + '.pdf')
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(lds_q.mean_continuous_states[0], O_comp, '.')
plt.xlabel("estimated R")
plt.ylabel("O")
plt.title(r"Standardized O V.S. $\hat{R}$")
plt.savefig(folder_figure + 'LDS_initilization_O_R_V' + aim + '.pdf')
plt.show()

print(np.mean(lds_q.mean_continuous_states[0]))
print(lds.params)
As_init, bs_init, Vs_init, sqrt_Sigmas_init = lds.dynamics.params
Cs_init, Fs_init, ds_init, inv_etas_init = lds.emissions.params


# %% fit R


N_iters = 10
dat_org['R'] = 0.0
dat_org['R0'] = 0.0
notif_summary = []


for i, userid in enumerate(userid_all):
    dat_user = dat_std[dat_std['id'] == userid]
    I = dat_user['I'].to_numpy().reshape(-1, K)
    C = dat_user['C'].to_numpy().reshape(-1, K)
    A = dat_user['A'].to_numpy().reshape(-1, K)
    M = dat_user.pivot(index='day', columns='time', values='M')
    M = M.fillna(M.mean()).to_numpy()
    Ed = dat_user['E'].to_numpy().reshape(-1, K)[:, [0]]
    Edm1 = dat_user['E0'].to_numpy().reshape(-1, K)[:, [0]]
    O = dat_user['O_ewm'].to_numpy().reshape(-1, K)[:, [0]]
    E0 = Edm1[0]

    notif_summary.append([O.shape[0], np.sum(I), np.sum(A)])

    ## fit the LDS model
    obs = np.insert(O[1:], O.shape[0] - 1, O[-1], axis=0)
    inpt = np.hstack([M, Ed])
    lds = ssm.LDS(
        N=obs_dim, D=state_dim, M=input_dim,
        transitions="inputdriven", dynamics="gaussian", emissions="gaussian"
    )
    lds.dynamics._As = As_init
    lds.dynamics.bs = bs_init
    lds.dynamics.Vs = Vs_init
    lds.dynamics._sqrt_Sigmas = sqrt_Sigmas_init
    lds.emissions._Cs = Cs_init
    lds.emissions.Fs = Fs_init
    lds.emissions.ds = ds_init
    lds.emissions.inv_etas = inv_etas_init
    lds_lps, lds_q = lds.fit(
        obs, inputs=inpt, 
        method="laplace_em", num_iters=N_iters, initialize=False
    )
    ## estimated rewards
    Rd = lds_q.mean_continuous_states[0].reshape(-1)
    Rdm1 = np.insert(Rd[:-1], 0, Rd[0])

    # ## Plot the log probabilities of the true and fit models
    # plt.plot(lds_lps, label="EM")
    # plt.xlim(0, N_iters)
    # plt.xlabel("EM Iteration")
    # plt.ylabel("Log Probability")
    # plt.show()
    # ## histogram
    # plt.plot(Rd)
    # plt.show()

    dat_org.loc[dat_org['id'] == userid, 'R'] = np.repeat(Rd, K)
    dat_org.loc[dat_org['id'] == userid, 'R0'] = np.repeat(Rdm1, K)

dat_org.to_csv(file_dat)
notif_summary = np.array(notif_summary)
np.sum(notif_summary[:, 2] / notif_summary[:, 0] < 0.5)


# %%
