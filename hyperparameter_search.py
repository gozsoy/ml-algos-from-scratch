import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import KFold, ParameterGrid

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from bayesian_opt import GP_UCB


def find_best_hyperparams_bayesian(X_train,y_train,param_grid,categorical_feats,model,metric_fnc,tol): # Bayesian way
    
    scaler = StandardScaler()
    kf = KFold(n_splits=5)

    hyper_matrix = pd.DataFrame(list(map(lambda x: list(x.values()),list(ParameterGrid(param_grid)))))
    
    ord_encoder = OrdinalEncoder()
    hyper_scaler = StandardScaler()
    
    param_grid_keys = list(list(ParameterGrid(param_grid))[0].keys())
    cat_indices = list(map(lambda x: param_grid_keys.index(x),categorical_feats))

    if len(cat_indices) > 0:
        hyper_matrix[cat_indices] = ord_encoder.fit_transform(hyper_matrix[cat_indices])
        
    hyper_matrix = hyper_scaler.fit_transform(hyper_matrix)

    best_score = -1
    best_score_std = None
    best_config = None

    def compute_cv_score(temp_hyp_cfg):
        
        temp_hyp_cfg = hyper_scaler.inverse_transform(temp_hyp_cfg)
        
        if len(cat_indices) > 0:
            temp_hyp_cfg = pd.DataFrame([temp_hyp_cfg])
            temp_hyp_cfg[cat_indices] = ord_encoder.inverse_transform(temp_hyp_cfg[cat_indices])
            temp_hyp_cfg = temp_hyp_cfg.values[0]

        temp_hyp_cfg = dict((list(param_grid_keys)[idx],temp_hyp_cfg[idx]) for idx in range(len(temp_hyp_cfg)))


        scores = []
        for train_index, test_index in kf.split(X_train):   
            X_train_cv, X_valid_cv = X_train[train_index], X_train[test_index]
            y_train_cv, y_valid_cv = y_train[train_index], y_train[test_index]

            X_train_cv = scaler.fit_transform(X_train_cv)
            X_valid_cv = scaler.transform(X_valid_cv)

            model.set_params(**temp_hyp_cfg)

            model.fit(X_train_cv,y_train_cv)

            y_pred_cv = model.predict(X_valid_cv)
            scores.append(metric_fnc(y_valid_cv,y_pred_cv))

        nonlocal best_score,best_score_std,best_config
        if best_score < np.mean(scores):
            best_score = np.mean(scores)
            best_score_std = np.std(scores)
            best_config = temp_hyp_cfg

        return np.mean(scores)


    kernel = 0.8 * RBF(0.5,length_scale_bounds="fixed")
    gpr = GaussianProcessRegressor(kernel=kernel)

    ucb = GP_UCB(gpr,hyper_matrix,compute_cv_score,beta = 2.0)


    for _ in range(tol):
        
        ucb.query_new_point()

        ucb.update_gp()


    print(f'hyper_config: {best_config}, mean: {best_score:.4f}, std: {best_score_std:.4f}')

    return


def find_best_hyperparams_grid(X_train,y_train,param_grid,model,metric_fnc):
    print(f'size of hyperparameter space: {len(list(ParameterGrid(param_grid)))}')

    scaler = StandardScaler()
    kf = KFold(n_splits=5)

    best_score = -1
    best_score_std = None
    best_config = None

    for hyp_cfg in list(ParameterGrid(param_grid)):

        scores = []
        for train_index, test_index in kf.split(X_train):   
            X_train_cv, X_valid_cv = X_train[train_index], X_train[test_index]
            y_train_cv, y_valid_cv = y_train[train_index], y_train[test_index]

            X_train_cv = scaler.fit_transform(X_train_cv)
            X_valid_cv = scaler.transform(X_valid_cv)

            model.set_params(**hyp_cfg)

            model.fit(X_train_cv,y_train_cv)

            y_pred_cv = model.predict(X_valid_cv)
            scores.append(metric_fnc(y_valid_cv,y_pred_cv))

        if best_score < np.mean(scores):
            best_score = np.mean(scores)
            best_score_std = np.std(scores)
            best_config = hyp_cfg


    print(f'hyper_config: {best_config}, mean: {best_score:.4f}, std: {best_score_std:.4f}')
    return