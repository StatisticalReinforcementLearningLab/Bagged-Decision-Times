# Bagged Decision Times

Code for the paper "Harnessing Causality in Reinforcement Learning With Bagged Decision Times".

## File Descriptions

- Base files that will be used for both running experiments for the proposed algorithm and estimating standardized treatment effect (STE).
  - `/dataset.py`: A container that stores the generated episodes.
  - `/env_testbed.py`: Implements the vanilla testbed. The files `env_testbed_RE.py`, `env_testbed_AR.py`, and `env_testbed_RC.py` are testbed variants that violate the assumptions in the DAG. `env_testbed_MA.py` allows interaction effects between $M_{d, 1:k-1}$ and $A_{d, k}$ on $M_{d, k}$.
  - `/env_config_base.py`: Base environment configurator.
  - `/mrt.py`: Runs a micro-randomized trial (MRT) that selects actions with a fixed probability in the vanilla testbed. The files `mrt_RE.py`, `mrt_AR.py`, `mrt_RC.py`, and `mrt_MA.py` run the MRT in different testbed variants.

- `/experiments`: Contains scripts to run experiments for BRLSVI, RLSVI, SRLSVI, and RAND.
  - `/exp_BRLSVI.py`, `/exp_RLSVI.py`, `/exp_SRLSVI.py`, and `/exp_RAND.py` run the experiments for the four algorithms. 
  - `/env_config.py`, ..., `/env_config4.py` are the configurators for the four testbed variants in the paper. `env_config_AR.py` is the configurator for a testbed variant that violates the assumption $A_{d, 1:K} \to R_d$ in the DAG. `env_config_MA.py` is the configurator for a testbed variant that allows interaction effects between $M_{d, 1:k-1}$ and $A_{d, k}$.
  - `artificial_data.py`: Combines observed and artificial data (in the current experiments, no artificial data is used).
  - `BRLSVI.py`: Updates the policy in BRLSVI.
  - `RLSVI.py`: Updates the policy in RLSVI.
  - `SRLSVI.py`: Updates the policy in SRLSVI.
  - To compare different states, `/exp_BRLSVI_Sp.py`, `/exp_BRLSVI_Spp.py`, and `/exp_BRLSVI_Sppp.py` run the experiments with `/BRLSVI_Sp.py`, `/BRLSVI_Spp.py`, and `/BRLSVI_Sppp.py`, respectively.
  - `eval.py`: Compares different algorithms by drawing the figures.
  - `run_BRLSVI.sh`, `run_RLSVI.sh`, `run_SRLSVI.sh`, `run_RAND.sh`, and `run_BRLSVIS.sh` are scripts that submit their respective experiment code to the server.
  - `/params_env_V2`, `/params_env_RE_V2.py`, `/params_env_AR_V2.py`, `/params_env_RC_V2.py`, `/params_env_MA_V2.py`: Contain parameters for the vanilla testbed and other testbed variants. It preserves confidentiality via perturbations.
  - `params_std_V2.json`: Contains the standardization and truncation parameters.

- `/ste`: Estimates the STE for testbed variants.
  - `/env_config_base.py`: Base environment configurator. The parameter $W = 1$ means that all the bag-specific rewards are observed.
  - `/opt_policy.py`: Finds the true optimal policy of the vanilla testbed with a very large dataset. The files `opt_policy_RE.py`, `opt_policy_AR.py`, and `opt_policy_RC.py` find the true optimal policy of testbed variants that violate the assumptions in the DAG.
  - `/eval_ste.py`: Generates episodes under the optimal policy of the vanilla testbed and the zero policy. The files `eval_ste_RE.py`, `eval_ste_AR.py`, and `eval_ste_RC.py` generate episodes under the optimal policy of different testbed variants.
  - `/env_config.py`: Configurator for a testbed variant that enhances the positive effects by increasing $A_{d, k} \to M_{d, k}$.
  - `/env_config2.py`: Configurator for a testbed variant that enhances the negative effects by increasing $E_d \to R_d$ and decreasing $A_{d, k} \to E_d$.
  - `/env_config3.py`: Configurator for a testbed variant that enhances both the positive and the negative effects.
  - `/env_config_AR.py`: Configurator for a testbed variant that violates the assumption $A_{d, 1:K} \to R_d$ in the DAG and reduces the positive effect in $A_{d, 1:K} \to R_d$.
  - `/ste_variants.py`: Calculates the STE and draws the figures.
  - The testbed parameter files `/params_env_V2`, `/params_env_RE_V2.py`, `/params_env_AR_V2.py`, `/params_env_RC_V2.py`, `params_std_V2.json` need to be copied into this folder before running the code.
