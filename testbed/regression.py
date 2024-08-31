# %% set up


import numpy as np
import numpy.random as rd
from numpy.linalg import inv
from scipy.optimize import minimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


# %% linear regression with L2 and Laplacian penalty


class LinearReg:
    def __init__(self, alpha_l2_list, alpha_lap_list, ncv):
        self.beta = None
        self.alpha_l2 = None
        self.alpha_lap = None
        self.alpha_l2_list = alpha_l2_list
        self.alpha_lap_list = alpha_lap_list
        self.ncv = ncv


    def obj(self, beta, X, Y, L, alpha_l2, alpha_lap):
        obj = (Y - X @ beta).T @ (Y - X @ beta) / X.shape[0]
        penalty1 = alpha_l2 * beta.T @ beta
        penalty2 = alpha_lap * beta.T @ L @ beta
        return obj + penalty1 + penalty2


    def obj_grad(self, beta, X, Y, L, alpha_l2, alpha_lap):
        obj_grad = 2 * (- X).T @ (Y - X @ beta) / X.shape[0]
        penalty_grad1 = 2 * alpha_l2 * beta
        penalty_grad2 = 2 * alpha_lap * L @ beta
        return obj_grad + penalty_grad1 + penalty_grad2


    def fit(self, X, Y, L, alpha_l2, alpha_lap):
        Y = Y.reshape(-1)
        obj = lambda beta: self.obj(
            beta, X, Y, L, alpha_l2, alpha_lap
        )
        obj_grad = lambda beta: self.obj_grad(
            beta, X, Y, L, alpha_l2, alpha_lap
        )
        beta_init = np.zeros(X.shape[1])
        mod_opt = minimize(
            obj, jac=obj_grad, x0 = beta_init, method='BFGS'
        )
        self.beta = mod_opt.x  ## shape=(X.shape[1],)


    def predict(self, X):
        return X @ self.beta
    

    def cv(self, X, Y, L, seed):
        rd.seed(seed)
        n = X.shape[0]
        sse_cv = np.zeros((self.ncv, len(self.alpha_l2_list), len(self.alpha_lap_list)))
        fold_idx = rd.choice(self.ncv, n, replace=True)
        for m in range(self.ncv):
            X_train = X[fold_idx != m, :]
            Y_train = Y[fold_idx != m]
            X_valid = X[fold_idx == m, :]
            Y_valid = Y[fold_idx == m]
            for l, alpha_l2 in enumerate(self.alpha_l2_list):
                for j, alpha_lap in enumerate(self.alpha_lap_list):
                    self.fit(X_train, Y_train, L, alpha_l2, alpha_lap)
                    sse_cv[m, l, j] = np.sum((Y_valid - self.predict(X_valid))**2)
        sse_cv_mean = np.mean(sse_cv, axis=0)
        best_idx = np.unravel_index(sse_cv_mean.argmin(), sse_cv_mean.shape)
        self.beta = None
        self.alpha_l2 = self.alpha_l2_list[best_idx[0]]
        self.alpha_lap = self.alpha_lap_list[best_idx[1]]


# %% posterior of Bayesian linear regression


class BayesianLinearReg:
    def __init__(self, d):
        self.prior_mu = np.zeros((d, 1))
        self.prior_Sigma = 1 * np.eye(d)
        self.post_mu = None
        self.post_Sigma = None
        self.sigma2 = None


    def get_sigma(self, X, Y, userid):
        dat = np.hstack([
            userid.reshape(-1, 1), Y.reshape(-1, 1), X[:, 1:]
        ])
        dat = pd.DataFrame(dat)
        X_columns = ['X' + str(j) for j in range(X.shape[1] - 1)]
        dat.columns = ['userid', 'Y'] + X_columns
        fam = sm.families.Gaussian()
        cov = sm.cov_struct.Exchangeable()
        formula = 'Y ~ ' + ' + '.join(X_columns)
        model_gee = smf.gee(
            formula, "userid", dat, cov_struct=cov, family=fam
        ).fit()
        print(model_gee.summary())
        return model_gee.scale
    

    def fit(self, X, Y, userid):
        self.sigma2 = self.get_sigma(X, Y, userid)
        Y = Y.reshape(-1, 1)
        self.post_Sigma = inv(X.T @ X / self.sigma2 + inv(self.prior_Sigma))
        self.post_mu = self.post_Sigma @ (X.T @ Y / self.sigma2 + inv(self.prior_Sigma) @ self.prior_mu)

