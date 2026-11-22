"""Optuna search over LightGBM params on outputs/mats.pkl (5-fold GroupKFold by ZIP).
Run: python src/tune.py <n_trials>
"""
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna

OUT = Path(__file__).resolve().parents[1] / "outputs"
y, folds, mats, _ = pickle.loads((OUT / "mats.pkl").read_bytes())


def objective(trial):
    p = dict(objective="regression", learning_rate=0.03, verbose=-1, n_estimators=6000,
             num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
             min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.7),
             subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
             reg_lambda=trial.suggest_float("reg_lambda", 0.01, 50, log=True),
             reg_alpha=trial.suggest_float("reg_alpha", 0.001, 10, log=True),
             max_bin=trial.suggest_categorical("max_bin", [63, 255]),
             extra_trees=trial.suggest_categorical("extra_trees", [False, True]),
             random_state=trial.number)
    se, n, iters = 0.0, 0, []
    for k, ((a, b), (Xa, Xb)) in enumerate(zip(folds, mats)):
        m = lgb.LGBMRegressor(**p).fit(Xa, y[a], eval_set=[(Xb, y[b])],
                                       callbacks=[lgb.early_stopping(200, verbose=False)])
        pr = np.clip(m.predict(Xb, num_iteration=m.best_iteration_), 0, 100)
        se += ((y[b] - pr) ** 2).sum()
        n += len(b)
        iters.append(m.best_iteration_)
        trial.report(np.sqrt(se / n), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("iters", iters)
    return float(np.sqrt(se / n))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=6))
    st.enqueue_trial(dict(num_leaves=15, min_child_samples=40, colsample_bytree=0.4, subsample=0.8,
                          reg_lambda=5.0, reg_alpha=0.001, max_bin=255, extra_trees=False))
    st.optimize(objective, n_trials=int(sys.argv[1]),
                callbacks=[lambda s, t: print(t.number, t.state.name, t.value and round(t.value, 3),
                                              "best", round(s.best_value, 3), flush=True)])
    best = {**st.best_params, "iters": st.best_trial.user_attrs["iters"], "cv": st.best_value}
    (OUT / "tune_best.json").write_text(json.dumps(best, indent=1))
    print("BEST", best)
