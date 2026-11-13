"""
05_shap_analysis.py
SHAP analysis with the definitional / non-definitional split.
Handles the case where the categorical branch was fit on zero columns
(after the training-fold drop of all-NaN features).
"""
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from config import TAB_DIR, MODEL_DIR, FIG_DIR, SHEETS

DEFINITIONAL_KEYWORDS = [
    "loading", "biomass addition", "mass", "solid-liquid ratio",
    "liquid volume",
]


def is_definitional(feat):
    f = feat.lower()
    return any(k in f for k in DEFINITIONAL_KEYWORDS)


def recover_feature_names(pre):
    """Rebuild the output column names of a fitted ColumnTransformer,
    robust to the case where the cat branch has zero fitted columns."""
    names = []
    for spec_name, trans, cols in pre.transformers_:
        if spec_name == "num":
            names.extend(cols)
        elif spec_name == "cat":
            ohe = trans.named_steps["ohe"]
            if hasattr(ohe, "categories_"):
                names.extend(ohe.get_feature_names_out(cols).tolist())
            # else: the encoder was fit on 0 cols — contributes nothing
    return names


def run(name):
    pipe = joblib.load(MODEL_DIR / f"best_{name}.joblib")
    df = pd.read_parquet(TAB_DIR / f"02_clean_{name}.parquet")
    X = df.drop(columns=["__target__", "__group__"])
    te_idx = np.load(TAB_DIR / f"03_test_idx_{name}.npy")
    X_te = X.iloc[te_idx]

    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    X_te_t = pre.transform(X_te)

    feat_names = recover_feature_names(pre)
    if len(feat_names) != X_te_t.shape[1]:
        # last-ditch fallback: generic names
        print(f"[{name}] name mismatch ({len(feat_names)} vs "
              f"{X_te_t.shape[1]}); using generic names")
        feat_names = [f"f{i}" for i in range(X_te_t.shape[1])]

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_te_t)
    except Exception as e:
        print(f"[{name}] TreeExplainer failed ({e}); using KernelExplainer")
        bg = shap.sample(X_te_t, min(50, len(X_te_t)), random_state=0)
        explainer = shap.KernelExplainer(model.predict, bg)
        shap_vals = explainer.shap_values(X_te_t)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[..., 0]

    importance = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance["definitional"] = importance["feature"].apply(is_definitional)
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(TAB_DIR / f"05_shap_importance_{name}.csv", index=False)

    print(f"\n[{name}] TOP 20 SHAP features")
    print(importance.head(20).to_string(index=False))

    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#c0392b" if d else "#2e86c1" for d in top["definitional"]]
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"{name}: Top 20 predictors (red = definitional)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"05_shap_bar_{name}.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_vals, X_te_t, feature_names=feat_names,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"05_shap_beeswarm_{name}.png", dpi=150)
    plt.close()

    total = importance["mean_abs_shap"].sum()
    def_sum = importance.loc[importance["definitional"], "mean_abs_shap"].sum()
    print(f"\n[{name}] Definitional share of total |SHAP|: "
          f"{def_sum/total*100:.1f}%")
    with open(MODEL_DIR / f"05_shap_share_{name}.txt", "w") as f:
        f.write(f"definitional_share={def_sum/total:.4f}\n")
    return importance


if __name__ == "__main__":
    for k in SHEETS:
        run(k)