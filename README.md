# Bagged Decision Times

Code for the paper "Harnessing Causality in Reinforcement Learning With Bagged Decision Times".

## File Descriptions

- Base files that will be used for both running experiments and estimating standardized treatment effect (STE).
  - `/dataset.py`: A container that stores the generated episodes.
  - `/env_testbed.py`: Implements the vanilla testbed. The files `env_testbed_RE.py`, `env_testbed_AR.py`, and `env_testbed_RC.py` are testbed variants that violate the assumptions in the DAG.
  - `/env_config_base.py`: Base environment configurator.
  - `/mrt.py`: Runs a micro-randomized trial (MRT) that selects actions with a fixed probability in the vanilla testbed. The files `mrt_RE.py`, `mrt_AR.py`, and `mrt_RC.py` run the MRT in different testbed variants.

- `/experiments`: Contains scripts to run experiments for BRLSVI, RLSVI, SRLSVI, and RAND.
  - `/exp_BRLSVI.py`, `/exp_RLSVI.py`, `/exp_SRLSVI.py`, and `/exp_RAND.py` run the experiments for the four algorithms. `/exp_BRLSVIS.py` runs the experiment for BRLSVI with $\check{S}$.
  - `/env_config.py`, ..., `/env_config4.py` are the configurators for the four testbed variants in the paper. `env_config_AR.py` is the configurator for a testbed variant that violates the assumption $A_{d, 1:K} \to R_d$ in the DAG.
  - `artificial_data.py`: Generates artificial data in BRLSVI.
  - `BRLSVI.py`: Updates the policy in BRLSVI.
  - `RLSVI.py`: Updates the policy in RLSVI.
  - `SRLSVI.py`: Updates the policy in SRLSVI.
  - `eval.py`: Compares different algorithms by drawing the figures.
  - `/params_env_V2`, `/params_env_RE_V2.py`, `/params_env_AR_V2.py`, `/params_env_RC_V2.py`: Contains the testbed parameters for the vanilla testbed and other testbed variants with misspecified DAGs. It preserves confidentiality via perturbations.
  - `params_std_V2.json`: Contains the standardization and truncation parameters.
  - `/test_perturbed_testbed`: Contains a replication of Figure 2(a) with the perturbed testbed parameters.

- `/ste`: Estimates the STE for testbed variants.
  - `/env_config_base.py`: Base environment configurator. The parameter $W = 1$ means that all the bag-specific rewards are observed.
  - `/opt_policy.py`: Finds the true optimal policy of the vanilla testbed with a very large dataset. The files `opt_policy_RE.py`, `opt_policy_AR.py`, and `opt_policy_RC.py` find the true optimal policy of testbed variants that violate the assumptions in the DAG.
  - `/eval_ste.py`: Generates episodes under the optimal policy of the vanilla testbed and the zero policy. The files `eval_ste_RE.py`, `eval_ste_AR.py`, and `eval_ste_RC.py` generate episodes under the optimal policy of different testbed variants.
  - `/env_config.py`: Configurator for a testbed variant that enhances the positive effects by increasing $A_{d, k} \to M_{d, k}$.
  - `/env_config2.py`: Configurator for a testbed variant that enhances the negative effects by increasing $E_d \to R_d$ and decreasing $A_{d, k} \to E_d$.
  - `/env_config3.py`: Configurator for a testbed variant that enhances both the positive and the negative effects.
  - `/env_config_AR.py`: Configurator for a testbed variant that violates the assumption $A_{d, 1:K} \to R_d$ in the DAG and reduces the positive effect in $A_{d, 1:K} \to R_d$.
  - `/ste_variants.py`: Calculates the STE and draws the figures.
  - The testbed parameters `/params_env_V2`, `/params_env_RE_V2.py`, `/params_env_AR_V2.py`, `/params_env_RC_V2.py`, `params_std_V2.json` need to be copied into this folder.
