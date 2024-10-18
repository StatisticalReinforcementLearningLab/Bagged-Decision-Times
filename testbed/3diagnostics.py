# %% set up


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None


# %% figures


dropbox = ""
folder_dat = dropbox + "Data/"
aim = '2'
folder_params = 'params_env_RC_V' + aim + '/'
folder_pred = 'pred_V' + aim + '/'
folder_figure = 'Figures/'
file_params_env_prefix = folder_params + 'params_env_'

userid_all = np.loadtxt(folder_params + 'user_ids.txt', dtype=int)
dat_org = pd.read_csv(folder_dat + 'aim-' + aim + '_with_R.csv')


# %% coefficients in the testbed


fig_size = (10, 10)
subplot_size = (5, 4)
color = '#53446B'
alpha = 0.7

## for RC
fig_size = (10, 12)
subplot_size = (6, 4)

env_param_M = []
env_param_E = []
env_param_R = []
env_param_O = []
for userid in userid_all:
    with open(file_params_env_prefix + str(userid) + '.json', 'r') as file:
        params = json.load(file)
        env_param_M.append(params['theta_M'])
        env_param_E.append(params['theta_E'])
        env_param_R.append(params['theta_R'])
        env_param_O.append(params['theta_O'])
env_param_M = np.array(env_param_M)
env_param_E = np.array(env_param_E)
env_param_R = np.array(env_param_R)
env_param_O = np.array(env_param_O)

fig, axes = plt.subplots(
    subplot_size[0], subplot_size[1], figsize=fig_size, constrained_layout=True
)

## for general parameters
bin_width = 0.1
xlim = [-1, 1]
xtick_width = 0.5
ytick_width = 5
## A -> E
bin_width2 = 0.01
xlim2 = [-0.1, 0.1]
xtick_width2 = 0.05
ytick_width2 = 10
## M -> R
bin_width3 = 0.02
xlim3 = [-0.2, 0.2]
xtick_width3 = 0.1
ytick_width3 = 10
## R -> O
bin_width4 = 0.25
xlim4 = [-1, 1.5]
xtick_width4 = 0.5
ytick_width4 = 5

##### M ##### 0-7
for j in range(env_param_M.shape[1]):
    ax_idx = np.unravel_index(j, subplot_size)
    X = env_param_M[:, j].reshape(-1)
    bin_min = np.floor(np.min(X) / bin_width) * bin_width
    bin_max = np.ceil(np.max(X) / bin_width) * bin_width
    bins = np.arange(bin_min, bin_max + bin_width, bin_width)
    axes[ax_idx].grid()
    values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
    axes[ax_idx].title.set_text(r'$\theta^{M}_{%s}$' % j)
    axes[ax_idx].set_xticks(np.arange(xlim[0], xlim[1] + xtick_width, xtick_width))
    axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width, ytick_width))

##### E ##### 8-10
## intercept
## E
for l in [0, 1]:
    j += 1
    ax_idx = np.unravel_index(j, subplot_size)
    X = env_param_E[:, l].reshape(-1)
    bin_min = np.floor(np.min(X) / bin_width) * bin_width
    bin_max = np.ceil(np.max(X) / bin_width) * bin_width
    bins = np.arange(bin_min, bin_max + bin_width, bin_width)
    axes[ax_idx].grid()
    values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
    axes[ax_idx].title.set_text(r'$\theta^{E}_{%s}$' % l)
    axes[ax_idx].set_xticks(np.arange(xlim[0], xlim[1] + xtick_width, xtick_width))
    axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width, ytick_width))
## A
j += 1
ax_idx = np.unravel_index(j, subplot_size)
X = env_param_E[:, 2:7].reshape(-1)
bin_min = np.floor(np.min(X) / bin_width2) * bin_width2
bin_max = np.ceil(np.max(X) / bin_width2) * bin_width2
bins = np.arange(bin_min, bin_max + bin_width2, bin_width2)
axes[ax_idx].grid()
values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
axes[ax_idx].title.set_text(r'$\theta^{E}_{2:6}$')
axes[ax_idx].set_xticks(np.arange(xlim2[0], xlim2[1] + xtick_width2, xtick_width2))
axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width2, ytick_width2))
## empty
j += 1
ax_idx = np.unravel_index(j, subplot_size)
X = env_param_E[:, 7:12].reshape(-1)
bin_min = np.floor(np.min(X) / bin_width2) * bin_width2
bin_max = np.ceil(np.max(X) / bin_width2) * bin_width2
bins = np.arange(bin_min, bin_max + bin_width2, bin_width2)
axes[ax_idx].grid()
values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
axes[ax_idx].title.set_text(r'$\theta^{E}_{7:11}$')
axes[ax_idx].set_xticks(np.arange(xlim2[0], xlim2[1] + xtick_width2, xtick_width2))
axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width2, ytick_width2))

##### R #####
## intercept
j += 1
ax_idx = np.unravel_index(j, subplot_size)
X = env_param_R[:, 0].reshape(-1)
bin_min = np.floor(np.min(X) / bin_width) * bin_width
bin_max = np.ceil(np.max(X) / bin_width) * bin_width
bins = np.arange(bin_min, bin_max + bin_width, bin_width)
axes[ax_idx].grid()
values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
axes[ax_idx].title.set_text(r'$\theta^{R}_{0}$')
axes[ax_idx].set_xticks(np.arange(xlim[0], xlim[1] + xtick_width, xtick_width))
axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width, ytick_width))
## A
j += 1
ax_idx = np.unravel_index(j, subplot_size)
X = env_param_R[:, 1:6].reshape(-1)
bin_min = np.floor(np.min(X) / bin_width3) * bin_width3
bin_max = np.ceil(np.max(X) / bin_width3) * bin_width3
bins = np.arange(bin_min, bin_max + bin_width3, bin_width3)
axes[ax_idx].grid()
values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
axes[ax_idx].title.set_text(r'$\theta^{R}_{1:5}$')
axes[ax_idx].set_xticks(np.arange(xlim3[0], xlim3[1] + xtick_width3, xtick_width3))
axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width3, ytick_width3))
## E, R
for l in [6,7]:
    j += 1
    ax_idx = np.unravel_index(j, subplot_size)
    X = env_param_R[:, l].reshape(-1)
    bin_min = np.floor(np.min(X) / bin_width) * bin_width
    bin_max = np.ceil(np.max(X) / bin_width) * bin_width
    bins = np.arange(bin_min, bin_max + bin_width, bin_width)
    axes[ax_idx].grid()
    values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
    axes[ax_idx].title.set_text(r'$\theta^{R}_{%s}$' % l)
    axes[ax_idx].set_xticks(np.arange(xlim[0], xlim[1] + xtick_width, xtick_width))
    axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width, ytick_width))


##### O #####
for l in range(env_param_O.shape[1]):
    j += 1
    ax_idx = np.unravel_index(j, subplot_size)
    X = env_param_O[:, l].reshape(-1)
    bin_min = np.floor(np.min(X) / bin_width4) * bin_width4
    bin_max = np.ceil(np.max(X) / bin_width4) * bin_width4
    bins = np.arange(bin_min, bin_max + bin_width4, bin_width4)
    axes[ax_idx].grid()
    values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
    axes[ax_idx].title.set_text(r'$\theta^{O}_{%s}$' % l)
    axes[ax_idx].set_xticks(np.arange(xlim4[0], xlim4[1] + xtick_width4, xtick_width4))
    axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width4, ytick_width4))


##### others & empty #####
## R -> E
# j += 1
# ax_idx = np.unravel_index(j, subplot_size)
# X = env_param_E[:, 12].reshape(-1)
# bin_min = np.floor(np.min(X) / bin_width) * bin_width
# bin_max = np.ceil(np.max(X) / bin_width) * bin_width
# bins = np.arange(bin_min, bin_max + bin_width, bin_width)
# axes[ax_idx].grid()
# values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
# axes[ax_idx].title.set_text(r'$\theta^{E}_{12}$')
# axes[ax_idx].set_xticks(np.arange(xlim[0], xlim[1] + xtick_width, xtick_width))
# axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width, ytick_width))
                                  
## A -> R
# j += 1
# ax_idx = np.unravel_index(j, subplot_size)
# X = env_param_R[:, 8:12].reshape(-1)
# bin_min = np.floor(np.min(X) / bin_width2) * bin_width2
# bin_max = np.ceil(np.max(X) / bin_width2) * bin_width2
# bins = np.arange(bin_min, bin_max + bin_width2, bin_width2)
# axes[ax_idx].grid()
# values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
# axes[ax_idx].title.set_text(r'$\theta^{R}_{8:12}$')
# axes[ax_idx].set_xticks(np.arange(xlim2[0], xlim2[1] + xtick_width2, xtick_width2))
# axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width2, ytick_width2))
                                  
for l in range(2):
    j += 1
    ax_idx = np.unravel_index(j, subplot_size)
    axes[ax_idx].axis('off')

## R -> C
env_param_C = []
for userid in userid_all:
    with open(file_params_env_prefix + str(userid) + '.json', 'r') as file:
        params = json.load(file)
        env_param_C.append(params['theta_C'])
env_param_C = np.array(env_param_C)

for l in range(env_param_C.shape[1]):
    j += 1
    ax_idx = np.unravel_index(j, subplot_size)
    X = env_param_C[:, l].reshape(-1)
    bin_min = np.floor(np.min(X) / bin_width) * bin_width
    bin_max = np.ceil(np.max(X) / bin_width) * bin_width
    bins = np.arange(bin_min, bin_max + bin_width, bin_width)
    axes[ax_idx].grid()
    values, _, _ = axes[ax_idx].hist(X, bins=bins, color=color, alpha=alpha)
    axes[ax_idx].title.set_text(r'$\theta^{C}_{%s}$' % l)
    axes[ax_idx].set_xticks(np.arange(xlim[0], xlim[1] + xtick_width, xtick_width))
    axes[ax_idx].set_yticks(np.arange(0, np.max(values) + ytick_width, ytick_width))

for l in range(1):
    j += 1
    ax_idx = np.unravel_index(j, subplot_size)
    axes[ax_idx].axis('off')

fig.supxlabel('coefficient')
fig.supylabel('frequency')
# fig.suptitle(r'Histogram of Coefficients for M')
plt.savefig(folder_figure + 'testbed_coef_RC_V' + aim + '.pdf')
plt.show()


# %% residuals in the testbed


fig_size = (12, 12)
subplot_size = (7, len(userid_all) // 7)
color = '#53446B'
alpha = 0.7
xtick_width = 1
ytick_width = 1


fig, axes = plt.subplots(subplot_size[0], subplot_size[1], figsize=fig_size, constrained_layout=True)
for i, userid in enumerate(userid_all):
    with open(file_params_env_prefix + str(userid) + '.json', 'r') as file:
        params = json.load(file)
    with open(folder_pred + 'pred_' + str(userid) + '.json', 'r') as file:
        predicted = json.load(file)
    pred_M = np.array(predicted['pred_M'])
    resid_M = np.array(params['resid_M'])
    resid_M = resid_M[~np.isnan(resid_M)]

    ax_idx = np.unravel_index(i, subplot_size)
    axes[ax_idx].grid()
    axes[ax_idx].plot(pred_M, resid_M, '.', color=color, alpha=alpha)
    axes[ax_idx].title.set_text(str(userid))
    axes[ax_idx].set_xticks(np.arange(-2, 2 + xtick_width, xtick_width))
    axes[ax_idx].set_yticks(np.arange(-2, 2 + ytick_width, ytick_width))

fig.supxlabel('predicted M')
fig.supylabel('residual')
# fig.suptitle(r'Residual in the Model of M')
plt.savefig(folder_figure + 'testbed_resid_M_V' + aim + '.pdf')
plt.show()


fig, axes = plt.subplots(subplot_size[0], subplot_size[1], figsize=fig_size, constrained_layout=True)
for i, userid in enumerate(userid_all):
    with open(file_params_env_prefix + str(userid) + '.json', 'r') as file:
        params = json.load(file)
    with open(folder_pred + 'pred_' + str(userid) + '.json', 'r') as file:
        predicted = json.load(file)
    pred_E = np.array(predicted['pred_E'])
    resid_E = np.array(params['resid_E'])

    ax_idx = np.unravel_index(i, subplot_size)
    axes[ax_idx].grid()
    axes[ax_idx].plot(pred_E, resid_E, '.', color=color, alpha=alpha)
    axes[ax_idx].title.set_text(str(userid))
    axes[ax_idx].set_xticks(np.arange(-2, 2 + xtick_width, xtick_width))
    axes[ax_idx].set_yticks(np.arange(-2, 2 + ytick_width, ytick_width))

fig.supxlabel('predicted E')
fig.supylabel('residual')
# fig.suptitle(r'Residual in the Model of E')
plt.savefig(folder_figure + 'testbed_resid_E_V' + aim + '.pdf')
plt.show()


fig, axes = plt.subplots(subplot_size[0], subplot_size[1], figsize=fig_size, constrained_layout=True)
for i, userid in enumerate(userid_all):
    with open(file_params_env_prefix + str(userid) + '.json', 'r') as file:
        params = json.load(file)
    with open(folder_pred + 'pred_' + str(userid) + '.json', 'r') as file:
        predicted = json.load(file)
    pred_R = np.array(predicted['pred_R'])
    resid_R = np.array(params['resid_R'])

    ax_idx = np.unravel_index(i, subplot_size)
    axes[ax_idx].grid()
    axes[ax_idx].plot(pred_R, resid_R, '.', color=color, alpha=alpha)
    axes[ax_idx].title.set_text(str(userid))
    axes[ax_idx].set_xticks(np.arange(-2, 2 + xtick_width, xtick_width))
    axes[ax_idx].set_yticks(np.arange(-2, 2 + ytick_width, ytick_width))

fig.supxlabel('predicted R')
fig.supylabel('residual')
# fig.suptitle(r'Residual in the Model of R')
plt.savefig(folder_figure + 'testbed_resid_R_V' + aim + '.pdf')
plt.show()


fig, axes = plt.subplots(subplot_size[0], subplot_size[1], figsize=fig_size, constrained_layout=True)
for i, userid in enumerate(userid_all):
    with open(file_params_env_prefix + str(userid) + '.json', 'r') as file:
        params = json.load(file)
    with open(folder_pred + 'pred_' + str(userid) + '.json', 'r') as file:
        predicted = json.load(file)
    pred_O = np.array(predicted['pred_O'])
    resid_O = np.array(params['resid_O'])

    ax_idx = np.unravel_index(i, subplot_size)
    axes[ax_idx].grid()
    axes[ax_idx].plot(pred_O, resid_O, '.', color=color, alpha=alpha)
    axes[ax_idx].title.set_text(str(userid))
    axes[ax_idx].set_xticks(np.arange(-2, 2 + xtick_width, xtick_width))
    axes[ax_idx].set_yticks(np.arange(-2, 2 + ytick_width, ytick_width))

fig.supxlabel('predicted O')
fig.supylabel('residual')
# fig.suptitle(r'Residual in the Model of O')
plt.savefig(folder_figure + 'testbed_resid_O_V' + aim + '.pdf')
plt.show()


# %% dropout


dropout_summary = pd.read_csv(folder_dat + 'dropout.csv', index_col='id')
# print(np.sum(dropout_summary['dropout'] > 0))
# print(dropout_summary.mean())


dropout_summary['before study'] = 0
dropout_summary['warm up'] = 0
for userid in userid_all:
    dat_daily = pd.read_csv(
        folder_dat + 'csv files/kpwhri.aim-' + aim + '.' + str(userid) 
        + '.daily-metrics.csv'
    )
    dat_steps = pd.read_csv(
        folder_dat + 'csv files/kpwhri.aim-' + aim + '.' + str(userid) 
        + '.fitbit-data-per-minute.csv'
    )
    dat_request = pd.read_csv(
        folder_dat + 'csv files/kpwhri.aim-' + aim + '.' + str(userid) 
        + '.walking-suggestion-service-requests.csv'
    )

    date = dat_daily['Date']
    date_study_start = dropout_summary.loc[userid, 'study start date']
    day_first_study = (date == date_study_start).idxmax()

    warm_up = dat_request.loc[
        dat_request['url'] == 'http://walking-suggestion:8080/initialize', 
        'request_data'
    ]
    warm_up = json.loads(warm_up.to_list()[-1])
    len_warm = len(warm_up['totalStepsArray'])
    date_first_warm = datetime.strptime(
        warm_up['date'], '%Y-%m-%d'
    ).date() - timedelta(days=len_warm - 1)
    date_study_start = datetime.strptime(
        date_study_start, '%Y-%m-%d'
    ).date()
    day_first_wear = day_first_study - (date_study_start - date_first_warm).days

    ## the 'daily-metrics' table missed to save the first day
    ## so the actual day_first_record may be negative
    day_first_record = 0
    if day_first_wear < 0:
        day_first_record = day_first_wear

    dropout_summary.loc[userid, 'before study'] = day_first_wear - day_first_record
    dropout_summary.loc[userid, 'warm up'] = len_warm


dropout_plot = dropout_summary.copy()
dropout_plot = dropout_plot[['dropout', 'in study', 'warm up', 'before study']]
dropout_plot.sum(axis=1)

## plot dropout
figsize = (8, 4)
colors = ['#A40407', '#53446B', '#F16C93', '#C0AECE']
alpha = 0.7
dropout_ax = dropout_plot.plot.bar(
    stacked=True, figsize=figsize, color=colors, alpha=alpha
)
dropout_ax.set_xlabel("User ID")
dropout_ax.set_ylabel("number of days")
dropout_bar = dropout_ax.get_figure()
dropout_bar.tight_layout()
dropout_bar.savefig(folder_figure + 'dropout_bar_V' + aim + '.pdf')


# %% cause of unavailability


check_avail = []
for userid in userid_all:
    dat_walking = pd.read_csv(
        folder_dat + 'csv files/kpwhri.aim-' + aim + '.' + str(userid) 
        + '.walking-suggestion-decisions.csv'
    )
    num_dec = dat_walking.shape[0]
    if 'Available' in dat_walking.columns:
        num_avail = np.sum(dat_walking['Available'])
    else:
        num_avail = np.sum(dat_walking['available'])
    num_unreachable = np.sum(dat_walking['Unavailable Unreachable'])
    check_avail.append([num_avail, num_unreachable, num_dec - num_avail - num_unreachable])

check_avail = pd.DataFrame(
    check_avail, 
    index=userid_all,
    columns=['Available', 'Unavailable Unreachable', 'Other Unavailable']
)

figsize = (8, 4)
colors = ['#53446B', '#F16C93', '#C0AECE']
alpha = 0.7
unavail_ax = check_avail.plot.bar(
    stacked=True, figsize=figsize, color=colors, alpha=alpha
)
unavail_ax.set_xlabel("User ID")
unavail_ax.set_ylabel("Number of decision points")
unavail_bar = unavail_ax.get_figure()
unavail_bar.tight_layout()
unavail_bar.savefig(folder_figure + 'unavailability_bar_V' + aim + '.pdf')


# %% histogram of variables


fig_size = (4, 2.8)
color = '#53446B'
alpha = 0.7

## plot E
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(dat_org['E'][dat_org['time'] == 1], color=color, alpha=alpha)
plt.title("square root of app view")
plt.ylabel("frequency")
plt.savefig(folder_figure + 'hist_E_V' + aim + '.pdf')
plt.show()
print(np.max(dat_org['E']))


## plot R
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(dat_org['R'][dat_org['time'] == 1], color=color, alpha=alpha)
plt.title("commitment to PA")
plt.ylabel("frequency")
plt.savefig(folder_figure + 'hist_R_V' + aim + '.pdf')
plt.show()
print(np.max(dat_org['R']))


## plot O
plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(dat_org['O'][dat_org['time'] == 1], color=color, alpha=alpha)
plt.title("bouts of unprompted PA")
plt.ylabel("frequency")
plt.savefig(folder_figure + 'hist_O_V' + aim + '.pdf')
plt.show()
print(np.max(dat_org['O']))


# %% missing data in C


org_C = []
C_all_users = []
for userid in userid_all:
    user_C = pd.DataFrame(np.zeros((0, 7)))
    user_C.columns = ['userid', 'day'] + ['C' + str(k) for k in range(1, 6)]
    dat_service = pd.read_csv(
        folder_dat + 'csv files/kpwhri.aim-' + aim + '.' + str(userid) 
        + '.walking-suggestion-service-requests.csv'
    )
    for j in range(dat_service.shape[0]):
        ## nightly-update rows contain the information about both the date and the day in study
        ## some users (10027, 10137, 10307) have multiple dates corresponding to the same day of study
        ## use later records to update the previous records
        if dat_service.loc[j, 'url'] == "http://walking-suggestion:8080/nightly":
            json_info = json.loads(dat_service.loc[j, 'request_data'])
            day = json_info['studyDay']
            if np.isnan(day):
                continue
            user_C.loc[day - 1, 'day'] = day
            user_C.loc[day - 1, 'C1':'C5'] = json_info['preStepsArray']
    user_C['userid'] = userid
    dropout_day = np.max(dat_org.loc[dat_org['id'] == userid, 'day'])
    user_C = user_C[user_C['day'] <= dropout_day]
    miss_point = np.sum(user_C.loc[:, 'C1':'C5'].isna().sum())
    miss_day = np.sum(user_C.loc[:, 'C1':'C5'].isna(), axis=1)
    miss_day = np.sum(miss_day > 0)
    org_C.append([userid, miss_point, miss_day])
    CC = np.log(user_C.loc[:, 'C1':'C5'].values.reshape(-1) + 0.5)
    C_all_users.extend(CC.tolist())

org_C = np.array(org_C)


plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(org_C[:, 1], color=color, alpha=alpha)
plt.title("decision times with missing C")
plt.ylabel("frequency")
plt.savefig(folder_figure + 'hist_C_missing_h_V' + aim + '.pdf')
plt.show()


plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(org_C[:, 2], color=color, alpha=alpha)
plt.title("days with missing C")
plt.ylabel("frequency")
plt.savefig(folder_figure + 'hist_C_missing_d_V' + aim + '.pdf')
plt.show()


plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(dat_org['C'], color=color, alpha=alpha)
plt.title("log prior 30-min step counts")
plt.ylabel("frequency")
# plt.title("missing data imputed")
plt.savefig(folder_figure + 'hist_C_V' + aim + '.pdf')
plt.show()


plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(C_all_users, color=color, alpha=alpha)
plt.title("log prior 30-min step counts")
plt.ylabel("frequency")
# plt.title("missing data removed")
plt.savefig(folder_figure + 'hist_C_missing_removed_V' + aim + '.pdf')
plt.show()


# %% plot M


org_M = []
for userid in userid_all:
    dat_user = dat_org[dat_org['id'] == userid]
    M = dat_user.pivot(index='day', columns='time', values='M')
    M = M.fillna(M.mean()).values.reshape(-1)
    org_M.extend(M.tolist())


plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(dat_org['M'], color=color, alpha=alpha)
plt.title("log post 30-min step counts")
plt.ylabel("frequency")
# plt.title("missing data removed")
plt.savefig(folder_figure + 'hist_M_V' + aim + '.pdf')
plt.show()


plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.hist(org_M, color=color, alpha=alpha)
plt.title("log post 30-min step counts")
plt.ylabel("frequency")
# plt.title("missing data imputed")
plt.savefig(folder_figure + 'hist_M_missing_imputed_V' + aim + '.pdf')
plt.show()


# %%
