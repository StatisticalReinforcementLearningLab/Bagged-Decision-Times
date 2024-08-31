# %% set up


import numpy as np
import numpy.random as rd
import pandas as pd
from env_config_AR import EnvConfig
from env_testbed_AR import Env
from dataset import Dataset
from mrt_AR import MRT
from joblib import Parallel, delayed
import d3rlpy
import sys
jobid = int(sys.argv[1])


# %% specification


params_env_path = 'params_env_AR_V2/'
params_std_file = 'params_std_V2.json'
userid_all = np.loadtxt(params_env_path + 'user_ids.txt', dtype=int)

scalar_list = np.arange(1, 6)
seed_list = np.arange(len(scalar_list)) * 2024
N_train = 5000
n_jobs = 8 # parallel computing
P0 = 0.5

array_idx = np.unravel_index(jobid, (len(scalar_list), len(userid_all)))
scalar = scalar_list[array_idx[0]]
userid = userid_all[array_idx[1]]
seed = seed_list[array_idx[0]]

exp = '5'
logger_dir = 'd3rlpy_logs/exp_' + str(exp)
tensorboard_dir = 'tensorboard_logs/exp_' + str(exp)
experiment_name = 'job' + str(scalar) + '_user' + str(userid)
model_dir = logger_dir + '/' + experiment_name + '_model.d3'


# %% create a large offline dataset


## initialization
rd.seed(seed)

def generate_offline_data(n):
    # userid = rd.choice(userid_all, size=1)[0]
    config = EnvConfig(userid, params_env_path, params_std_file, scalar)
    env = Env(config, noise='random')
    mrt = MRT(config, env)
    K = config.K
    W = config.W
    D = config.D
    nweek = config.nweek
    dat = mrt.gen_data_fixed_prob(P0, W, D)

    ## action
    actions = dat.df.loc[1:, dat.col_A].values.reshape(-1)

    ## reward
    rewards = dat.df['R'].fillna(0).values.reshape(-1, 1)
    rewards = np.hstack([np.zeros((D + 1, K - 1)), rewards]).reshape(-1)
    rewards = rewards[K:] ## remove leading zeros

    ## state indicators
    state_mask_d = np.tile(np.repeat(np.arange(W), K), nweek)
    state_mask_d = pd.get_dummies(state_mask_d, drop_first=True, dtype=int).values
    state_mask_h = np.tile(np.arange(K), nweek * W)
    state_mask_h = pd.get_dummies(state_mask_h, drop_first=True, dtype=int).values

    ## state C
    state_C = dat.df.loc[1:, dat.col_C].values.reshape(-1, 1)

    ## state M
    shift_by_time = np.arange(1, W * K).tolist()  ## len=W*K-1
    state_M = pd.Series(dat.df.loc[1:, dat.col_M].values.reshape(-1))
    state_M = state_M.shift(periods=shift_by_time, fill_value=0).values
    for j in range(W * K):
        state_M[np.arange(j, nweek * W * K, W * K), j:] = 0

    ## state A
    state_A = pd.Series(dat.df.loc[1:, dat.col_A].values.reshape(-1))
    state_A = state_A.shift(periods=shift_by_time, fill_value=0).values
    for j in range(W * K):
        state_A[np.arange(j, nweek * W * K, W * K), j:] = 0

    ## state R
    state_R = dat.df.loc[np.arange(0, D, W), 'R_obs'].values
    state_R = np.repeat(state_R, W * K).reshape(-1, 1)

    ## state E
    shift_by_day = np.arange(0, W).tolist()  ## len=W
    state_E = dat.df.loc[:(D - 1), 'E']  ##.loc contains the end of indices
    state_E = state_E.shift(periods=shift_by_day, fill_value=0).values
    for j in range(W):
        state_E[np.arange(j, nweek * W, W), (j + 1):] = 0
    state_E = np.repeat(state_E, K, axis=0)

    ## combine states
    states = np.hstack([
        state_mask_d, state_mask_h, state_C, state_M, state_A, state_R, state_E
    ])
    terminals = np.zeros(D * K)
    timeouts = np.zeros(D * K)
    timeouts[-1] = 1
    out = {}
    out['states'] = states
    out['actions'] = actions.reshape(-1, 1)
    out['rewards'] = rewards.reshape(-1, 1)
    out['terminals'] = terminals.reshape(-1, 1)
    out['timeouts'] = timeouts.reshape(-1, 1)
    return out


results = Parallel(n_jobs=n_jobs)(
    delayed(generate_offline_data)(n) for n in range(N_train)
)
results = list(results)


## save the offline dataset for training WQN
dats = {
    'states': [],
    'rewards': [],
    'actions': [],
    'terminals': [],
    'timeouts': [],
}

## combine all the episodes from N users
for res in results:
    for key in dats.keys():
        dats[key].append(res[key])

for key, value in dats.items():
    dats[key] = np.vstack(value)

## in the d3rlpy package, 'terminals' and 'timeouts' must be 1-dim
dats['terminals'] = dats['terminals'].reshape(-1)
dats['timeouts'] = dats['timeouts'].reshape(-1)


# %% optimal policy


## create the dataset for DQN
dataset = d3rlpy.dataset.MDPDataset(
    observations=dats['states'],
    actions=dats['actions'],
    rewards=dats['rewards'],
    terminals=dats['terminals'],
    timeouts=dats['timeouts'],
    action_space=d3rlpy.constants.ActionSpace.DISCRETE,
    action_size=2,
)

test_episodes = dataset.episodes[:100]

# encoder factory
encoder_factory = d3rlpy.models.VectorEncoderFactory(
    hidden_units=[256, 128, 64, 32],
)
# logger_adapter
logger_adapter = d3rlpy.logging.CombineAdapterFactory([
   d3rlpy.logging.FileAdapterFactory(root_dir=logger_dir),
   d3rlpy.logging.TensorboardAdapterFactory(root_dir=tensorboard_dir),
])
evaluators={
    'td_error': d3rlpy.metrics.TDErrorEvaluator(test_episodes),
    'value_scale': d3rlpy.metrics.AverageValueEstimationEvaluator(test_episodes),
}

# setup algorithm
dqn = d3rlpy.algos.DQNConfig(
    batch_size=256, gamma=0.99, learning_rate=1e-04, target_update_interval=5000,
    encoder_factory=encoder_factory
).create(device=True)
dqn.fit(
    dataset, n_steps=100000, n_steps_per_epoch=1000, 
    experiment_name=experiment_name,
    logger_adapter=logger_adapter, evaluators=evaluators,
    save_interval=1000, show_progress=False,
)

dqn.save(model_dir)

