"""xG Lab data: the final xG model (trained on ALL 2015/16 top-4-league shots) applied to the most recent big
tournaments in StatsBomb's free data - an honest out-of-sample test (new players, new era, international football).
Writes xg/app_data/: shots.parquet (every tournament shot with our xG), model files, metrics.json.
Run: .venv/Scripts/python xg/tournaments.py"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[1]))
import statsbomb
from common import angle, distance, load
from features import ALL_FEATURES, add_features
# same settings as xg/model.py (copied: importing model.py would re-run its whole training script)
PENALTY_XG = 0.76
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)

HERE = Path(__file__).parent
OUT = HERE / "app_data"
TOURNAMENTS = {"World Cup 2022": (43, 106), "Euro 2024": (55, 282), "Copa America 2024": (223, 282)}
N_TREES = 208   # what early stopping chose in xg/model.py


def tournament_shots():
    frames = []
    for name, (cid, sid) in TOURNAMENTS.items():
        ms = requests.get(f"{statsbomb.RAW}/matches/{cid}/{sid}.json", timeout=30).json()
        statsbomb.download_events([m["match_id"] for m in ms])
        info = {m["match_id"]: f"{m['home_team']['home_team_name']} {m['home_score']}-{m['away_score']} "
                               f"{m['away_team']['away_team_name']}" for m in ms}
        rows = [r for m in ms for r in statsbomb._shot_rows(m["match_id"])]
        df = pd.DataFrame(rows)
        df["tournament"], df["match"] = name, df.match_id.map(info)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    OUT.mkdir(exist_ok=True)
    train = add_features(load())   # all 2015/16 non-penalty shots
    X = train[ALL_FEATURES].astype(float)
    gbm = lgb.LGBMClassifier(n_estimators=N_TREES, random_state=0, **PARAMS).fit(X, train.is_goal)
    mu, sd = X.mean(), X.std().replace(0, 1)
    lr = LogisticRegression(max_iter=3000).fit((X - mu) / sd, train.is_goal)

    shots = tournament_shots()
    open_play = shots.shot_type.isin(["Open Play", "Free Kick"])
    np_shots = shots[open_play].copy()
    np_shots["distance"], np_shots["angle"] = distance(np_shots.x, np_shots.y), angle(np_shots.x, np_shots.y)
    feats = add_features(np_shots)
    Xt = feats[ALL_FEATURES].astype(float)
    p = (gbm.predict_proba(Xt)[:, 1] + lr.predict_proba((Xt - mu) / sd)[:, 1]) / 2   # same blend as the final model
    shots["xg"] = np.nan
    shots.loc[np_shots.index, "xg"] = p
    shots.loc[shots.shot_type == "Penalty", "xg"] = PENALTY_XG

    metrics = {}
    for name, g in shots[open_play].groupby("tournament"):
        metrics[name] = {"shots": len(g), "goals": int(g.is_goal.sum()),
                         "our_xg": round(g.xg.sum(), 1), "statsbomb_xg": round(g.sb_xg.sum(), 1),
                         "our_log_loss": round(log_loss(g.is_goal, g.xg), 4),
                         "statsbomb_log_loss": round(log_loss(g.is_goal, g.sb_xg), 4),
                         "our_auc": round(roc_auc_score(g.is_goal, g.xg), 3),
                         "statsbomb_auc": round(roc_auc_score(g.is_goal, g.sb_xg), 3)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    keep = ["tournament", "match", "minute", "team", "player", "x", "y", "body_part", "shot_type", "technique",
            "play_pattern", "outcome", "is_goal", "xg", "sb_xg"]
    shots[keep].to_parquet(OUT / "shots.parquet")
    # for the click-the-pitch calculator: the interpretable half of the blend, as plain numbers
    (OUT / "logreg.json").write_text(json.dumps({"features": ALL_FEATURES, "coef": lr.coef_[0].tolist(),
                                                 "intercept": float(lr.intercept_[0]),
                                                 "mean": mu.tolist(), "std": sd.tolist()}))
    gbm.booster_.save_model(str(OUT / "lgbm.txt"))
    print(pd.DataFrame(metrics).T.to_string())
    print(f"saved {len(shots):,} shots to {OUT}")


if __name__ == "__main__":
    main()
