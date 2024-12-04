# %% set up


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


# %% parameters


exp = '1'
ver = '1'
path_figure = 'Figures/'

nweek = 4 * 9 ## number of weeks in an episode
W = 7 ## number of days in a week
L = 1 ## update interval
K = 5 ## number of decision points in a day
H = L * K
D = nweek * W
D_warm = 7
nitr = 300

save_cols = ['userid', 'R', 'R_imp', 'R0']
res_columns = ['W'] + [x + '_mean' for x in save_cols] + [x + '_median' for x in save_cols]


opt_sigma2s = {
    'BRLSVI': 0.005, 
    'SRLSVI': 1, 
    'RLSVI': 0.01, 
    'RAND': 0,
}
opt_regs = {
    'BRLSVI': 5, 
    'SRLSVI': 10, 
    'RLSVI': 2, 
    'RAND': 0,
}
methods = ['BRLSVI', 'SRLSVI', 'RLSVI', 'RAND']
method_names = ['BRLSVI', 'SRLSVI', 'RLSVI', 'RAND']

R_cum_df = pd.DataFrame()
for m in methods:
    path_BTS = 'res_' + m + ver + '/'
    sigma2 = opt_sigma2s[m]
    reg_beta = opt_regs[m]
    version = 'version' + ver + '_sigma2_' + str(sigma2) + '_reg_' + str(reg_beta)
    dat = pd.read_csv(
        path_BTS + version + '.txt', 
        index_col=None, header=None, names=res_columns
    )

    dat_R = dat['R_mean'].values.reshape(-1, D + 1)
    dat_R_diff = dat_R - dat['R0_mean'].values.reshape(-1, D + 1)
    R_cum = np.cumsum(dat_R_diff, axis=1)
    R_cum_df = pd.concat([R_cum_df, pd.DataFrame({
        'method': np.array([m] * (D + 1)), 
        'time': np.arange(D + 1), 
        'value': np.mean(R_cum, axis=0),
        'value_max': np.quantile(R_cum, 0.975, axis=0),
        'value_min': np.quantile(R_cum, 0.025, axis=0),
    })])


colors = ['#b2182b', '#993404', '#253494', '#807dba']
for i, m in enumerate(methods):
    dat_plot = R_cum_df.loc[R_cum_df['method'] == m]
    line, = plt.plot(
        dat_plot['time'], dat_plot['value'], 
        label=f'{method_names[i]}', color = colors[i]
    )
    plt.fill_between(
        dat_plot['time'], 
        dat_plot['value_min'], dat_plot['value_max'], 
        color=line.get_color(), alpha=0.2
    )
plt.legend(loc='upper left')
plt.axvline(x=D_warm, linestyle='dotted', color='gray')
plt.xlim(0, D)
plt.xlabel('day')
plt.ylabel('cumulative reward - baseline')
plt.grid()
plt.savefig(path_figure + 'compare' + exp + '_' + ver + '.pdf')
plt.show()


# %%


exp = '1'
ver = '1'
path_figure = 'Figures/'

opt_sigma2s = {
    'BRLSVI': 0.005, 
    'BRLSVIS': 0.005, 
}
opt_regs = {
    'BRLSVI': 5, 
    'BRLSVIS': 5, 
}
methods = ['BRLSVI', 'BRLSVIS']
method_names = [r'$S_{hat}$', r'$S_{check}$']

R_cum_df = pd.DataFrame()
for m in methods:
    path_BTS = 'res_' + m + ver + '/'
    sigma2 = opt_sigma2s[m]
    reg_beta = opt_regs[m]
    version = 'version' + ver + '_sigma2_' + str(sigma2) + '_reg_' + str(reg_beta)
    dat = pd.read_csv(
        path_BTS + version + '.txt', 
        index_col=None, header=None, names=res_columns
    )

    dat_R = dat['R_mean'].values.reshape(-1, D + 1)
    dat_R_diff = dat_R - dat['R0_mean'].values.reshape(-1, D + 1)
    R_cum = np.cumsum(dat_R_diff, axis=1)
    R_cum_df = pd.concat([R_cum_df, pd.DataFrame({
        'method': np.array([m] * (D + 1)), 
        'time': np.arange(D + 1), 
        'value': np.mean(R_cum, axis=0),
        'value_max': np.quantile(R_cum, 0.975, axis=0),
        'value_min': np.quantile(R_cum, 0.025, axis=0),
    })])


colors = ['#b2182b', '#253494']
for i, m in enumerate(methods):
    dat_plot = R_cum_df.loc[R_cum_df['method'] == m]
    line, = plt.plot(
        dat_plot['time'], dat_plot['value'], 
        label=f'{method_names[i]}', color = colors[i]
    )
    plt.fill_between(
        dat_plot['time'], 
        dat_plot['value_min'], dat_plot['value_max'], 
        color=line.get_color(), alpha=0.2
    )
plt.legend(loc='upper left')
plt.axvline(x=D_warm, linestyle='dotted', color='gray')
plt.xlim(0, D)
plt.xlabel('day')
plt.ylabel('cumulative reward - baseline')
plt.grid()
plt.savefig(path_figure + 'compare' + exp + '_state_' + ver + '.pdf')
plt.show()


# %%
