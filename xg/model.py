"""xG Day 5: LightGBM vs logistic regression vs StatsBomb, calibration, heatmap, and xG for every shot (for Project 2).
Run: .venv/Scripts/python xg/model.py"""
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import evaluate, load, split
from features import ALL_FEATURES, add_features
from statsbomb import DATA, shots

HERE = Path(__file__).parent
PENALTY_XG = 0.76  # long-run penalty conversion; 300/400 = 0.75 in this data
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def fit_lgbm(train, seed=0):
    """Hold out 20% of TRAIN matches for early stopping, so the test set is never touched while fitting."""
    fit, val = split(train, test_size=0.2, seed=seed)
    model = lgb.LGBMClassifier(n_estimators=3000, random_state=seed, **PARAMS)
    model.fit(fit[ALL_FEATURES].astype(float), fit.is_goal,
              eval_set=[(val[ALL_FEATURES].astype(float), val.is_goal)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def fit_logreg(train):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(train[ALL_FEATURES].astype(float), train.is_goal)


def predict(model, df):
    return model.predict_proba(df[ALL_FEATURES].astype(float))[:, 1]


df = add_features(load())
train, test = split(df)

lr, gbm = fit_logreg(train), fit_lgbm(train)
p_lr, p_gbm = predict(lr, test), predict(gbm, test)
print(f"LightGBM stopped at {gbm.best_iteration_} trees")
res = pd.DataFrame([
    evaluate("logistic regression (29 features)", test.is_goal, p_lr),
    evaluate("LightGBM (29 features)", test.is_goal, p_gbm),
    evaluate("average of both", test.is_goal, (p_lr + p_gbm) / 2),
    evaluate("StatsBomb xG (benchmark)", test.is_goal, test.sb_xg.to_numpy()),
])
res["gap_closed"] = (0.3109 - res.log_loss) / (0.3109 - res.log_loss.iloc[-1])
print(res.round(4).to_string(index=False))

# Is the remaining gap to StatsBomb real or noise? Paired bootstrap over test MATCHES (shots within a match are correlated).
rng = np.random.default_rng(0)
t = test.assign(ll_ours=-(test.is_goal * np.log((p_lr + p_gbm) / 2) + (1 - test.is_goal) * np.log(1 - (p_lr + p_gbm) / 2)),
                ll_sb=-(test.is_goal * np.log(test.sb_xg) + (1 - test.is_goal) * np.log(1 - test.sb_xg)))
per_match = t.groupby("match_id")[["ll_ours", "ll_sb"]].agg(["sum", "size"])
diffs = []
for _ in range(1000):
    m = per_match.iloc[rng.integers(0, len(per_match), len(per_match))]
    diffs.append(m[("ll_ours", "sum")].sum() / m[("ll_ours", "size")].sum() - m[("ll_sb", "sum")].sum() / m[("ll_sb", "size")].sum())
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"log-loss gap (ours - StatsBomb): {np.mean(diffs):+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]"
      f" -> {'StatsBomb genuinely better' if lo > 0 else 'not distinguishable'}")

imp = pd.Series(gbm.booster_.feature_importance("gain"), index=ALL_FEATURES)
print("\nLightGBM top features by gain share:")
print((imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_string())

# Calibration on the test set
fig, ax = plt.subplots(figsize=(6, 6))
for name, p in [("logistic", p_lr), ("LightGBM", p_gbm), ("StatsBomb xG", test.sb_xg)]:
    CalibrationDisplay.from_predictions(test.is_goal, p, n_bins=10, strategy="quantile", name=name, ax=ax)
ax.set_title("Calibration on held-out matches")
fig.savefig(HERE / "calibration_final.png", dpi=120, bbox_inches="tight")

# Honest xG for EVERY shot: 5-fold by match, each shot predicted by a model that never saw its match.
# This is what Project 2 uses, so player xG isn't inflated by training-set memorisation.
oof = np.zeros(len(df))
for k, (tr, te) in enumerate(GroupKFold(5).split(df, groups=df.match_id)):
    oof[te] = predict(fit_lgbm(df.iloc[tr], seed=k), df.iloc[te])
print(f"\nout-of-fold xG: {evaluate('LightGBM OOF (all shots)', df.is_goal, oof)}")

xg = pd.concat([
    pd.DataFrame({"id": df.id, "xg": oof}),
    shots().query("shot_type == 'Penalty'")[["id"]].assign(xg=PENALTY_XG),
])
xg.to_parquet(DATA / "shot_xg.parquet")  # direct corners (7 shots) have no xG
print(f"saved {DATA / 'shot_xg.parquet'} ({len(xg):,} shots)")

# Heatmap of average xG per zone, like the video
df["xg"] = oof
pitch = VerticalPitch(pitch_type="statsbomb", half=True, pad_bottom=-20)
fig, ax = pitch.draw(figsize=(8, 7))
hb = pitch.hexbin(df.x, df.y, C=df.xg, reduce_C_function=np.mean, gridsize=(18, 18), mincnt=10, cmap="viridis", ax=ax, edgecolors="none")
goals = df[df.is_goal]
pitch.scatter(goals.x, goals.y, marker="*", s=6, color="white", alpha=0.25, ax=ax)
fig.colorbar(hb, ax=ax, shrink=0.6, label="average xG (our LightGBM model)")
ax.set_title("Average xG per zone, non-penalty shots (stars = goals)")
fig.savefig(HERE / "xg_heatmap.png", dpi=120, bbox_inches="tight")
print("saved calibration_final.png, xg_heatmap.png")
