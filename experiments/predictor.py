# %% set up


import numpy as np
from scipy.optimize import minimize
from numpy.linalg import norm


# %% mle


class Predictor:
    def __init__(self, K, sigma2s, idxes_theta, lam):
        self.K = K
        ## pre-fixed noise variance
        self.sigma2_R, self.sigma2_O, self.sigma2_M = sigma2s
        ## index of unknown parameters in the vector vars
        self.idx_theta_R, self.idx_theta_O, self.idx_theta_M = idxes_theta
        self.lam = lam


    def loglik_normal(self, Y, X, mu, sigma2):
        ## log-likelihood of a normally distributed variable
        loglik = - 1/(2*sigma2) * (Y - X @ mu).T @ (Y - X @ mu)
        ## the constant can be ignored
        # loglik += - 1/2 * np.log(2 * np.pi) - 1/2 * np.log(sigma2) 
        return loglik.item()


    def grad_theta_loglik_normal(self, Y, X, mu, sigma2):
        ## gradient of log-likelihood of a normally distributed variable
        grad = - 1/sigma2 * (Y - X @ mu).T @ (- X)
        return grad.reshape(-1)


    def neg_loglik_all(self, vars):
        ## copy the vectors R_{d-1} and R_d from the combined dataset
        RRdm1_var, RR_var = self.RRdm1.copy(), self.RR.copy()
        ## extract unknown parameters and unobserved rewards from the vars vector
        theta_R = vars[self.idx_theta_R].reshape(-1, 1)
        theta_O = vars[self.idx_theta_O].reshape(-1, 1)
        theta_M = vars[self.idx_theta_M].reshape(-1, 1)
        RRdm1_var[self.idx_RUdm1_dat] = vars[self.idx_RUdm1_vars]
        RR_var[self.idx_RUd_dat] = vars[self.idx_RUd_vars]

        ## for 1000 replications of calculating the sum of log-likelihood of R (D = 300)
        ## for loop takes 4.4; matrix operation takes 0.6s

        ## construct the predictors in the conditions
        cond_R = np.hstack([self.intercept, self.MM, self.EE, RRdm1_var])
        cond_O = np.hstack([self.intercept, RRdm1_var])
        cond_M = []
        for k in range(self.K):
            Ch = self.CC[:, [k]]
            Ah = self.AA[:, [k]]
            cond_M.append(np.hstack([
                self.intercept, self.EEdm1, RRdm1_var, Ch, 
                Ah, Ah * self.EEdm1, Ah * RRdm1_var, Ah * Ch
            ]))

        loglik = 0
        ## loglik of R
        loglik += self.loglik_normal(RR_var, cond_R, theta_R, self.sigma2_R)
        ## loglik of O
        loglik += self.loglik_normal(self.OO, cond_O, theta_O, self.sigma2_O)
        ## loglik of M
        for k in range(0, self.K):
            loglik += self.loglik_normal(self.MM[:, [k]], cond_M[k], theta_M, self.sigma2_M)
        ## L2 penalty
        penalty = self.lam * norm(vars, 2)**2
        return - loglik + penalty


    def grad_neg_loglik_all(self, vars):
        ## copy the vectors R_{d-1} and R_d from the combined dataset
        RRdm1_var, RR_var = self.RRdm1.copy(), self.RR.copy()
        ## extract unknown parameters and unobserved rewards from the vars vector
        theta_R = vars[self.idx_theta_R].reshape(-1, 1)
        theta_O = vars[self.idx_theta_O].reshape(-1, 1)
        theta_M = vars[self.idx_theta_M].reshape(-1, 1)
        RRdm1_var[self.idx_RUdm1_dat] = vars[self.idx_RUdm1_vars]
        RR_var[self.idx_RUd_dat] = vars[self.idx_RUd_vars]

        ## construct the predictors in the conditions
        cond_R = np.hstack([self.intercept, self.MM, self.EE, RRdm1_var])
        cond_O = np.hstack([self.intercept, RRdm1_var])
        cond_M = []
        for k in range(self.K):
            Ch = self.CC[:, [k]]
            Ah = self.AA[:, [k]]
            cond_M.append(np.hstack([
                self.intercept, self.EEdm1, RRdm1_var, Ch, 
                Ah, Ah * self.EEdm1, Ah * RRdm1_var, Ah*Ch
            ]))

        ## d(loglik_R) / d(theta_R)
        grad_theta_R = self.grad_theta_loglik_normal(RR_var, cond_R, theta_R, self.sigma2_R)
        ## d(loglik_O) / d(theta_O)
        grad_theta_O = self.grad_theta_loglik_normal(self.OO, cond_O, theta_O, self.sigma2_O)
        ## d(loglik_M) / d(theta_M)
        grad_theta_M = 0
        for k in range(0, self.K):
            grad_theta_M += self.grad_theta_loglik_normal(
                self.MM[:, [k]], cond_M[k], theta_M, self.sigma2_M
            )
        
        idx_start_RUall = self.idx_theta_M[-1] + 1
        ## index of unobserved R_{d-1} in the combined vector of unobserved R_{d-1} and R_{d} 
        idx_RUdm1_RUall = self.idx_RUdm1_vars - idx_start_RUall
        ## index of unobserved R_{d} in the combined vector of unobserved R_{d-1} and R_{d} 
        idx_RUd_RUall = self.idx_RUd_vars - idx_start_RUall

        ## d(loglik) / d(all unobserved R)
        grad_RU = np.zeros(len(vars) - idx_start_RUall)

        ## d(loglik_R) / d(R_{d})
        grad_Rd = - 1 / self.sigma2_R * (RR_var - cond_R @ theta_R)
        ## add grad_Rd to grad_RU 
        grad_RU[idx_RUd_RUall] += grad_Rd[self.idx_RUd_dat]

        ## d(loglik_R) / d(R_{d-1})
        grad_Rdm1 = - 1 / self.sigma2_R * (RR_var - cond_R @ theta_R) * (- theta_R[7, 0])
        ## d(loglik_O) / d(R_{d-1})
        grad_Rdm1 += - 1 / self.sigma2_O * (self.OO - cond_O @ theta_O) * (- theta_O[1, 0])
        ## d(loglik_M) / d(R_{d-1})
        for k in range(self.K):
            Ah = self.AA[:, [k]]
            grad_Rdm1 += - 1 / self.sigma2_M * (self.MM[:, [k]] - cond_M[k] @ theta_M) \
                * (- theta_M[2, 0] - Ah * theta_M[6, 0])
        ## add grad_Rdm1 to grad_RU 
        grad_RU[idx_RUdm1_RUall] += grad_Rdm1[self.idx_RUdm1_dat]

        grad = np.hstack([grad_theta_R, grad_theta_O, grad_theta_M, grad_RU])
        grad_penalty = self.lam * 2 * vars
        return - grad + grad_penalty


    def maximize_loglik(self, comb_dat, dat_RR_obs, init_theta):
        ## find the last observed reward for each unobserved reward
        mask = np.isnan(dat_RR_obs)
        idx_ffill = np.where(~mask, np.arange(len(dat_RR_obs)), 0)
        idx_ffill = np.maximum.accumulate(idx_ffill, axis=0)
        init_RR_mis = dat_RR_obs[idx_ffill][mask]
        ## combine the initial value of theta and initial value of unobserved rewards
        init = np.hstack([init_theta, init_RR_mis])

        ## start index of all unobserved R in the vector vars
        idx_start_RU = self.idx_theta_M[-1] + 1
        ## index of unobserved R_{d-1} in the vector vars
        self.idx_RUdm1_vars = idx_start_RU + np.arange(0, np.sum(mask[:-1]))
        ## index of unobserved R_{d} in the vector vars
        self.idx_RUd_vars = idx_start_RU + np.arange(mask[0].astype(int), np.sum(mask))

        ## extract variables from the combined dataset
        dd = comb_dat['comb_RR'].shape[0] * comb_dat['comb_RR'].shape[1]
        self.CC = comb_dat['comb_CC'].reshape(dd, self.K)
        self.AA = comb_dat['comb_AA'].reshape(dd, self.K)
        self.MM = comb_dat['comb_MM'].reshape(dd, self.K)
        self.EEdm1 = comb_dat['comb_EEdm1'].reshape(dd, 1)
        self.EE = comb_dat['comb_EE'].reshape(dd, 1)
        self.OO = comb_dat['comb_OO'].reshape(dd, 1)
        self.RRdm1 = comb_dat['comb_RRdm1'].reshape(dd, 1)
        self.RR = comb_dat['comb_RR'].reshape(dd, 1)
        self.intercept = np.ones((dd, 1))        

        ## index of unobserved R_{d-1} in the vector of all R_{d-1}
        self.idx_RUdm1_dat = np.isnan(self.RRdm1) 
        ## index of unobserved R_{d} in the vector of all R_{d}
        self.idx_RUd_dat = np.isnan(self.RR)

        results = minimize(
            self.neg_loglik_all, init, method='BFGS', 
            jac=self.grad_neg_loglik_all, tol=0.1
        )

        vars = results.x
        theta_R = vars[self.idx_theta_R]
        theta_O = vars[self.idx_theta_O]
        theta_M = vars[self.idx_theta_M]
        RUdm1 = vars[self.idx_RUdm1_vars]
        RUd = vars[self.idx_RUd_vars]

        return [theta_R, theta_O, theta_M, RUdm1, RUd]

