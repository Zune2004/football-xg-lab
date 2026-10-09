"""xG Days 3-4: feature engineering.
Day 3 = event features (how/where the shot happened). Day 4 = freeze-frame features (where everyone else was).
Run: .venv/Scripts/python xg/features.py   -> cumulative ablation table (each group added on top of the previous)."""
import json

import numpy as np
import pandas as pd

from common import GOAL_X, GOAL_Y, HALF_GOAL

POST_L, POST_R = (GOAL_X, GOAL_Y - HALF_GOAL), (GOAL_X, GOAL_Y + HALF_GOAL)


def in_triangle(p, a, b, c):
    """Is point p inside triangle abc? (sign-of-cross-product test; edges count as inside)"""
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def freeze_frame_features(x, y, frame_json):
    """Defenders between the ball and the goal, goalkeeper position, and close pressure on the shooter."""
    frame = json.loads(frame_json)
    shot = (x, y)
    opponents = [p for p in frame if not p["teammate"]]
    gk = next((p["location"] for p in opponents if p["position"]["name"] == "Goalkeeper"), None)
    outfield = [p["location"] for p in opponents if p["position"]["name"] != "Goalkeeper"]
    return {
        # outfield defenders inside the triangle shot -> left post -> right post: they can block it
        "defenders_in_cone": sum(in_triangle(loc, shot, POST_L, POST_R) for loc in outfield),
        # opponents within 2.5 units (~yards) of the shooter: physical pressure
        "opponents_close": sum(np.hypot(loc[0] - x, loc[1] - y) < 2.5 for loc in outfield),
        "gk_missing": gk is None,
        # GK off his line = more of the goal to aim at, or a lob; NaN-free defaults if no GK in frame
        "gk_dist_to_goal": np.hypot(GOAL_X - gk[0], GOAL_Y - gk[1]) if gk else 0.0,
        "gk_dist_to_shot": np.hypot(gk[0] - x, gk[1] - y) if gk else np.hypot(GOAL_X - x, GOAL_Y - y),
        "gk_in_cone": in_triangle(gk, shot, POST_L, POST_R) if gk else False,
    }


def add_features(df):
    df = df.copy()
    # --- Day 3: event features ---
    df["is_header"] = df.body_part == "Head"
    df["is_other_body"] = df.body_part == "Other"
    df["header_x_distance"] = df.is_header * df.distance  # headers lose power with distance much faster than kicks
    df["log_distance"] = np.log1p(df.distance)              # goal chance falls off non-linearly
    df["is_free_kick"] = df.shot_type == "Free Kick"
    df["from_counter"] = df.play_pattern == "From Counter"
    df["from_set_piece"] = df.play_pattern.isin(["From Corner", "From Free Kick", "From Throw In"])
    for t in ["Volley", "Half Volley", "Lob", "Overhead Kick", "Backheel", "Diving Header"]:
        df[f"tech_{t.lower().replace(' ', '_')}"] = df.technique == t
    df["assist_high"] = df.assist_height == "High Pass"
    # --- Day 4: freeze-frame features ---
    ff = pd.DataFrame([freeze_frame_features(x, y, f) for x, y, f in zip(df.x, df.y, df.freeze_frame)], index=df.index)
    df = pd.concat([df, ff], axis=1)
    return df


FEATURE_GROUPS = {
    "geometry (Day 2 baseline)": ["distance", "angle"],
    "+ non-linear distance": ["log_distance"],
    "+ body part": ["is_header", "is_other_body", "header_x_distance"],
    "+ shot technique": ["tech_volley", "tech_half_volley", "tech_lob", "tech_overhead_kick", "tech_backheel", "tech_diving_header"],
    "+ shot context": ["is_free_kick", "first_time", "under_pressure", "one_on_one"],
    "+ play pattern": ["from_counter", "from_set_piece"],
    "+ assist type": ["assisted", "assist_cross", "assist_through_ball", "assist_cut_back", "assist_high"],
    "+ freeze frame: defenders": ["defenders_in_cone", "opponents_close"],
    "+ freeze frame: goalkeeper": ["gk_dist_to_goal", "gk_dist_to_shot", "gk_in_cone", "gk_missing"],
}
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from common import evaluate, load, split

    # in_triangle self-check
    assert in_triangle((110, 40), (100, 40), POST_L, POST_R)
    assert not in_triangle((110, 30), (100, 40), POST_L, POST_R)

    train, test = split(add_features(load()))
    rows, used = [], []
    for name, group in FEATURE_GROUPS.items():
        used += group
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(train[used].astype(float), train.is_goal)
        rows.append(evaluate(name, test.is_goal, model.predict_proba(test[used].astype(float))[:, 1]))
    rows.append(evaluate("StatsBomb xG (benchmark)", test.is_goal, test.sb_xg.to_numpy()))
    res = pd.DataFrame(rows)
    res["gap_closed"] = (0.3109 - res.log_loss) / (0.3109 - res.log_loss.iloc[-1])  # 0.3109 = constant model (Day 2)
    print(res.round(4).to_string(index=False))

    coefs = pd.Series(model[-1].coef_[0], index=used).sort_values()
    print("\nstandardised coefficients, full model (+ = more likely goal):")
    print(coefs.round(3).to_string())
