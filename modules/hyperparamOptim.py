import optuna
from xgboost import XGBClassifier
from rich import print

from modules.utils import auc_from_eff, customROC_curve_AN
from modules.utils_logger import logger

def get_xgb_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"



def objective(trial, xp_train, xp_val, y_train, y_val, w_train, weight_nom_val, random_seed) -> float:
    verbosity=2
    device = get_xgb_device()
    print(f"\n\n====> device: {device}")
    clf = XGBClassifier(
        n_estimators= trial.suggest_int("n_estimators", 200, 2100),           # Number of trees
        max_depth=trial.suggest_int("max_depth", 3, 10),                 # Max depth
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),          # Shrinkage
        subsample=trial.suggest_float("subsample", 0.2, 1.0),               # Bagged sample fraction
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        
        tree_method='hist',          # Needed for max_bin
        device=device,
        max_bin=trial.suggest_int("max_bin", 64, 256),                  # Number of cuts
        
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0),

        gamma = trial.suggest_float("gamma", 0, 10),
        reg_alpha = trial.suggest_float("reg_alpha", 0, 10),
        reg_lambda = trial.suggest_float("reg_lambda", 0.1, 50, log=True),
        
        # objective='binary:logistic', # CrossEntropy (logloss)
        # use_label_encoder=False,     # Optional: suppress warning
        
        eval_metric=["logloss", "error", "auc"], # Ensures logloss used during training
        # eval_metric=["auc"], # Ensures logloss used during training
        n_jobs=30,                   # Use all CPU cores
        # scale_pos_weight=scale_pos_weight*0.005,
        # scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=15,#15
        verbosity=verbosity,
        random_state=random_seed,
    )

    eval_set = [(xp_train, y_train), (xp_val, y_val)]
    
    clf.fit(
        xp_train, y_train,
        eval_set=eval_set,
        sample_weight=w_train,
        verbose=False
    )

    proba = clf.predict_proba(xp_val)[:, 1]
    eff_bkg_val, eff_sig_val, thresholds_val, TpFpTnFn_df_val = customROC_curve_AN(y_val, proba, weight_nom_val, doClassBalance=False)
    
    auc  = auc_from_eff(eff_sig_val, eff_bkg_val)
    

    trial.report(auc, step=getattr(clf, "best_iteration", 0) or 0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return auc