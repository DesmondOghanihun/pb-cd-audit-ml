"""
04_train_compare.py
Six regressors under grouped CV, with honest train/test gap reporting
and a positive-test-R2 preference for the "best" model selection.

If no model reaches positive test R2, the best-by-RMSE model is still
saved (so downstream SHAP/screening can run) but a WARNING is printed.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.stats import loguniform, randint, uniform

from config import (TAB_DIR, MODEL_DIR, SHEETS, RANDOM_STATE,
                    N_RANDOM_SEARCH, CV_FOLDS, LOG_TARGET)


def make_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)])


def get_models():
    return {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": randint(200, 800),
             "model__max_depth": randint(4, 30),
             "model__min_samples_leaf": randint(1, 10),
             "model__max_features": uniform(0.3, 0.7)}),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"model__n_estimators": randint(100, 600),
             "model__learning_rate": loguniform(0.01, 0.3),
             "model__max_depth": randint(2, 8),
             "model__subsample": uniform(0.6, 0.4)}),
        "XGBoost": (
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=1,
                         objective="reg:squarederror"),
            {"model__n_estimators": randint(100, 800),
             "model__learning_rate": loguniform(0.01, 0.3),
             "model__max_depth": randint(3, 10),
             "model__subsample": uniform(0.6, 0.4),
             "model__colsample_bytree": uniform(0.5, 0.5)}),
        "CatBoost": (
            CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
            {"model__iterations": randint(200, 800),
             "model__learning_rate": loguniform(0.02, 0.2),
             "model__depth": randint(3, 9)}),
        "SVR": (
            SVR(),
            {"model__C": loguniform(0.1, 100),
             "model__epsilon": loguniform(1e-3, 1.0),
             "model__gamma": loguniform(1e-4, 1.0)}),
    }


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def drop_train_all_nan(X_tr, X_te):
    all_nan = X_tr.columns[X_tr.isna().all()].tolist()
    if all_nan:
        print(f"  dropping train-all-NaN features: {all_nan}")
    return X_tr.drop(columns=all_nan), X_te.drop(columns=all_nan)


def evaluate(name):
    df = pd.read_parquet(TAB_DIR / f"02_clean_{name}.parquet")
    y = df["__target__"].values.astype(float)
    groups = df["__group__"].values
    X = df.drop(columns=["__target__", "__group__"])

    tr_idx = np.load(TAB_DIR / f"03_train_idx_{name}.npy")
    te_idx = np.load(TAB_DIR / f"03_test_idx_{name}.npy")

    y_model = np.log1p(np.clip(y, a_min=0, a_max=None)) if LOG_TARGET else y

    X_tr_raw, X_te_raw = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
    X_tr_raw, X_te_raw = drop_train_all_nan(X_tr_raw, X_te_raw)

    y_tr, y_te = y_model[tr_idx], y_model[te_idx]
    y_te_raw = y[te_idx]

    pre = make_preprocessor(X_tr_raw)
    rows = []
    fitted = {}   # model_name -> (pipe, cv_rmse, test_r2, test_rmse)

    for mname, (estimator, params) in get_models().items():
        print(f"\n[{name}/{mname}] tuning...")
        pipe = Pipeline([("pre", pre), ("model", estimator)])
        gkf = GroupKFold(n_splits=CV_FOLDS)
        try:
            if params:
                search = RandomizedSearchCV(
                    pipe, params, n_iter=N_RANDOM_SEARCH,
                    scoring="neg_root_mean_squared_error",
                    cv=gkf, random_state=RANDOM_STATE, n_jobs=-1,
                    refit=True, error_score="raise")
                search.fit(X_tr_raw, y_tr, groups=groups[tr_idx])
                best_pipe = search.best_estimator_
                cv_rmse = -search.best_score_
            else:
                pipe.fit(X_tr_raw, y_tr)
                best_pipe = pipe
                cv_scores = cross_val_score(
                    pipe, X_tr_raw, y_tr, groups=groups[tr_idx], cv=gkf,
                    scoring="neg_root_mean_squared_error")
                cv_rmse = -cv_scores.mean()
        except Exception as e:
            print(f"  {mname} failed: {type(e).__name__}: {e}")
            continue

        pred_tr = best_pipe.predict(X_tr_raw)
        pred_te = best_pipe.predict(X_te_raw)
        tr_r2 = r2_score(y_tr, pred_tr)
        te_r2 = r2_score(y_te, pred_te)
        te_rmse = rmse(y_te, pred_te)
        pred_te_raw = np.expm1(pred_te) if LOG_TARGET else pred_te
        te_r2_raw = r2_score(y_te_raw, pred_te_raw)
        te_rmse_raw = rmse(y_te_raw, pred_te_raw)

        row = dict(model=mname, cv_rmse_log=cv_rmse,
                   train_r2_log=tr_r2, test_r2_log=te_r2,
                   test_rmse_log=te_rmse,
                   test_r2_mgg=te_r2_raw,
                   test_rmse_mgg=te_rmse_raw,
                   train_test_gap_log=tr_r2 - te_r2)
        rows.append(row)
        fitted[mname] = (best_pipe, cv_rmse, te_r2, te_rmse)
        print(f"  CV-RMSE(log)={cv_rmse:.4f}  train R2={tr_r2:.4f}  "
              f"test R2={te_r2:.4f}  gap={tr_r2-te_r2:.4f}")

    table = pd.DataFrame(rows).sort_values("test_rmse_log")
    table.to_csv(TAB_DIR / f"04_model_comparison_{name}.csv", index=False)
    print(f"\n[{name}] FINAL TABLE")
    print(table.to_string(index=False))

    if not fitted:
        print(f"[{name}] ALL models failed - nothing to save")
        return table, {"name": None, "pipeline": None}

    # -------- best-model selection: prefer positive test R2
    positive = [(m, v) for m, v in fitted.items() if v[2] > 0]
    if positive:
        best_name, (best_pipe, _, _, best_rmse) = min(
            positive, key=lambda kv: kv[1][3])
        print(f"[{name}] best model (positive test R2): "
              f"{best_name}  rmse={best_rmse:.4f}")
    else:
        best_name, (best_pipe, _, _, best_rmse) = min(
            fitted.items(), key=lambda kv: kv[1][3])
        print(f"[{name}] WARNING: no model has positive test R2. "
              f"Saving {best_name} by RMSE only -- DO NOT interpret "
              f"downstream SHAP/screening as generalisable.")

    joblib.dump(best_pipe, MODEL_DIR / f"best_{name}.joblib")
    with open(MODEL_DIR / f"best_{name}.txt", "w") as f:
        f.write(f"dataset={name}\nmodel={best_name}\n"
                f"test_rmse_log={best_rmse:.6f}\n"
                f"positive_test_r2={'yes' if positive else 'no'}\n"
                f"log_target={LOG_TARGET}\n"
                f"random_state={RANDOM_STATE}\n")
    return table, {"name": best_name, "pipeline": best_pipe}


if __name__ == "__main__":
    summary = {}
    for k in SHEETS:
        tbl, best = evaluate(k)
        summary[k] = best["name"]
    print("\nBest model per sheet:", summary)