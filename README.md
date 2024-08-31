# Missing and Bagged Rewards

Code for the paper "Harnessing Causality in Reinforcement Learning to Address Missing and Bagged Rewards in mHealth".

## File Descriptions

- `/testbed`: Preprocess the HeartSteps data, build the testbed, and construct the prior. The numbered scripts need to be run sequentially.
  - `/0filter_combine.R`: Extracts necessary variables from the raw and processed data in HeartSteps V2 and V3.
  - `/1estimate_R.py`: Constructs the rewards using linear dynamical systems for HeartSteps V2.
  - `/2build_dag.py`: Builds the vanilla testbed. The files `2build_dag_RE.py`, `2build_dag_AR.py`, and `2build_dag_RC.py` construct testbed variants that violate the assumptions in the DAG by representing additional arrows $R_{d-1} \to E_d$, $A_{d, 1:K} \to R_d$, and $R_{d-1}, E_{d-1} \to C_{d, 1:K}$, respectively.
  - `/3diagnostics.py`: Performs model diagnostics for the vanilla testbed.
  - `/4check_trend.py`: Checks the trend of generated data.
  - `/5estimate_R_prior.py`: Constructs the rewards using linear dynamical systems for HeartSteps V3.
  - `/6build_dag_prior.py`: Constructs the prior for BSLSVI.
  - `/6build_dag_prior_TS.py`: Constructs the prior for TSM.
  - `/lds.py`: Modifies the script in the [SSM](https://github.com/lindermanlab/ssm/blob/master/ssm/lds.py) package to remove the direct influence from the input vector to the emission variable and fixes the bug in initializing $Var(v_d^{\prime})$ when the latent variable and the emission variable have the same dimension.
  - `/regression.py`: Implements linear regression with $L_2$ and Laplacian penalties, as well as Bayesian linear regression.

- `/base`: Contains scripts used to check the testbed, estimate the standardized treatment effect (STE), and run experiments.
  - `/dataset.py`: A container that stores the generated episodes.
  - `/env_testbed.py`: Implements the vanilla testbed. The files `env_testbed_RE.py`, `env_testbed_AR.py`, and `env_testbed_RC.py` are testbed variants that violate the assumptions in the DAG.
  - `/env_config_base.py`: Base environment configurator.
  - `/mrt.py`: Runs a micro-randomized trial (MRT) that selects actions with a fixed probability in the vanilla testbed. The files `mrt_RE.py`, `mrt_AR.py`, and `mrt_RC.py` run the MRT in different testbed variants.

- `/ste`: Estimates the STE for testbed variants.
  - `/opt_policy.py`: Finds the true optimal policy of the vanilla testbed with a very large dataset. The files `opt_policy_RE.py`, `opt_policy_AR.py`, and `opt_policy_RC.py` find the true optimal policy of testbed variants that violate the assumptions in the DAG.
  - `/eval_ste.py`: Generates episodes under the optimal policy of the vanilla testbed and the zero policy. The files `eval_ste_RE.py`, `eval_ste_AR.py`, and `eval_ste_RC.py` generate episodes under the optimal policy of different testbed variants.
  - `/env_config.py`: Configurator for a testbed variant that enhances the positive effects by increasing $A_{d, k} \to M_{d, k}$.
  - `/env_config2.py`: Configurator for a testbed variant that enhances the negative effects by increasing $E_d \to R_d$ and decreasing $A_{d, k} \to E_d$.
  - `/env_config3.py`: Configurator for a testbed variant that enhances both the positive and the negative effects.
  - `/env_config_AR.py`: Configurator for a testbed variant that violates the assumption $A_{d, 1:K} \to R_d$ in the DAG and reduces the positive effect in $A_{d, 1:K} \to R_d$.
  - `/ste_variants.py`: Calculates the STE and draws the figures.

- `/experiments`: Contains scripts to run experiments for BSLSVI, RLSVI, TSM, and RAND.
  - `/exp_BSLSVI.py`, `/exp_RLSVI.py`, `/exp_TSM.py`, and `/exp_RAND.py` run the experiments for the four algorithms.
  - `/env_config.py`, ..., `/env_config4.py` are the configurators for the four testbed variants in the paper. `env_config_AR.py` is the configurator for a testbed variant that violates the assumption $A_{d, 1:K} \to R_d$ in the DAG.
  - `artificial_data.py`: Generates artificial data in BSLSVI.
  - `predictor.py`: Imputes missing rewards in BSLSVI.
  - `SLSVI.py`: Updates the policy in BSLSVI.
  - `RLSVI.py`: Updates the policy in RLSVI.
  - `TS.py`: Updates the policy in TS.