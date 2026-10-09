"""xG Day 2: logistic regression on distance + angle only. The bar every later model must beat.
Run: .venv/Scripts/python xg/baseline.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression

from common import angle, distance, evaluate, load, split

df = load()
train, test = split(df)
print(f"train {len(train):,} shots / {train.match_id.nunique()} matches | test {len(test):,} / {test.match_id.nunique()}")

FEATURES = ["distance", "angle"]
model = LogisticRegression().fit(train[FEATURES], train.is_goal)
p = model.predict_proba(test[FEATURES])[:, 1]

results = pd.DataFrame([
    evaluate("constant (train goal rate)", test.is_goal, np.full(len(test), train.is_goal.mean())),
    evaluate("baseline: distance + angle", test.is_goal, p),
    evaluate("StatsBomb xG (benchmark)", test.is_goal, test.sb_xg.to_numpy()),
])
print(results.round(4).to_string(index=False))
print("\ncoefficients:", dict(zip(FEATURES, model.coef_[0].round(3))), "intercept", model.intercept_.round(3))

for x, y, label in [(108, 40, "penalty spot"), (114, 40, "6-yard line, central"), (102, 40, "edge of box, central"),
                    (114, 22, "6 yards out, tight angle"), (95, 40, "25 yards, central")]:
    spot = pd.DataFrame({"distance": [distance(x, y)], "angle": [angle(x, y)]})
    print(f"  xG from {label:<26} {model.predict_proba(spot)[0, 1]:.3f}")

fig, ax = plt.subplots(figsize=(6, 6))
CalibrationDisplay.from_predictions(test.is_goal, p, n_bins=10, strategy="quantile", name="baseline", ax=ax)
CalibrationDisplay.from_predictions(test.is_goal, test.sb_xg, n_bins=10, strategy="quantile", name="StatsBomb xG", ax=ax)
ax.set_title("Calibration: predicted xG vs actual goal rate (test set)")
out = Path(__file__).parent / "calibration_baseline.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nsaved {out}")
