"""Shared xG pieces: data prep, geometry, the fixed train/test split, and metrics."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parents[1]))
from statsbomb import shots

GOAL_X, GOAL_Y, HALF_GOAL = 120, 40, 4  # StatsBomb 120x80 pitch; posts at y=36 and y=44


def distance(x, y):
    return np.hypot(GOAL_X - x, GOAL_Y - y)


def angle(x, y):
    """Angle (radians) between the two posts as seen from the shot location. Bigger = more goal to aim at."""
    dx, dy = GOAL_X - x, y - GOAL_Y
    return np.arctan2(2 * HALF_GOAL * dx, dx**2 + dy**2 - HALF_GOAL**2)


def load():
    """Non-penalty shots (penalties get a constant xG; 7 direct corners dropped) with geometry added."""
    df = shots()
    df = df[df.shot_type.isin(["Open Play", "Free Kick"])].reset_index(drop=True)
    df["distance"] = distance(df.x, df.y)
    df["angle"] = angle(df.x, df.y)
    return df


def split(df, test_size=0.25, seed=42):
    """Split by MATCH, never by shot, so one game's shots never leak across train/test. Fixed seed = same split every day."""
    train_idx, test_idx = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(df, groups=df.match_id))
    return df.iloc[train_idx], df.iloc[test_idx]


def evaluate(name, y, p):
    return {"model": name, "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p), "pred_goals": p.sum(), "goals": y.sum()}


if __name__ == "__main__":  # geometry self-check
    assert distance(108, 40) == 12                                   # penalty spot
    assert abs(np.degrees(angle(108, 40)) - 36.87) < 0.01            # 2*atan(4/12)
    assert abs(np.degrees(angle(119.9, 40)) - 180) < 3               # on the line, dead centre ~ 180 deg
    assert angle(120 - 10, 10) < angle(120 - 10, 40)                 # wide is worse than central
    assert angle(110, 30) == angle(110, 50)                          # symmetric
    print("geometry ok")
