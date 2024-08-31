# %% set up


import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


# %% homogeneous policy


exp = '1'
folder_figure = 'Figures/'
scalars = np.arange(6)

params_env_path = 'params_env_V2/'
userid_all = np.loadtxt(params_env_path + 'user_ids.txt', dtype=int)

idx_scalar = 0
idx_id = 1
idx_value_zero = 2
idx_value_opt = 3

path = 'results' + exp + '/'
stes = []
for scalar in scalars:
    user_es = []
    for userid in userid_all:
        filepath = path + 'res' + exp + '_' + str(scalar) + '_' + str(userid) + '.txt'
        reward = np.loadtxt(filepath)
        user_adv = np.mean(reward[:, idx_value_opt] - reward[:, idx_value_zero])
        user_std = np.std(reward[:, [idx_value_zero]])
        user_es.append([userid, user_adv, user_std])
    user_es = np.array(user_es)
    stes.append(np.mean(user_es[:, 1] / user_es[:, 2]))
    print(scalar, np.mean(user_es[:, 1]), np.mean(user_es[:, 2]))
stes = np.array(stes)

## standardized effect size
out = np.stack([scalars, stes], axis=1)
out = out[out[:, 0].argsort()]
out


# %% plots

fig_size = (4, 2.8)
color = '#53446B'
alpha = 0.7
xtick_step = 1
ytick_step = 0.1

plt.figure(figsize=fig_size)
plt.tight_layout()
plt.grid()
plt.plot(
    xtick_step * out[:, 0], out[:, 1], 
    'o-', color=color, alpha=alpha
)
xtick_min = xtick_step * np.min(out[:, 0])
xtick_max = xtick_step * np.max(out[:, 0])
plt.xticks(np.arange(xtick_min, xtick_max + 0.1 * xtick_step, xtick_step))
ytick_min = np.floor(np.min(out[:, 1] + 0.02) / ytick_step) * ytick_step
ytick_max = np.ceil(np.max(out[:, 1] - 0.02) / ytick_step) * ytick_step
plt.yticks(np.arange(ytick_min, ytick_max + 0.1 * ytick_step, ytick_step))
plt.xlabel(r'$\xi$')
plt.ylabel('STE')
plt.savefig(folder_figure + 'ste_' + exp + '.pdf', bbox_inches="tight")
plt.show()


# %%

