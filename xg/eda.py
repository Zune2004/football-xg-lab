"""xG Day 1: what do the shots look like?  Run: .venv/Scripts/python xg/eda.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

sys.path.insert(0, str(Path(__file__).parents[1]))
from statsbomb import shots

df = shots()
df["distance"] = np.hypot(120 - df.x, 40 - df.y)  # goal centre is (120, 40) on StatsBomb's 120x80 pitch
df["has_freeze_frame"] = df.freeze_frame != "[]"

print(f"{len(df):,} shots, {df.is_goal.sum():,} goals, {df.match_id.nunique():,} matches")
print(f"goal rate {df.is_goal.mean():.3f} | StatsBomb xG total {df.sb_xg.sum():.0f} vs goals {df.is_goal.sum()}")
print(f"freeze frame present: {df.has_freeze_frame.mean():.1%}")

for col in ["league", "shot_type", "body_part", "play_pattern", "outcome"]:
    t = df.groupby(col).agg(shots=("is_goal", "size"), goal_rate=("is_goal", "mean"), sb_xg=("sb_xg", "mean"))
    print("\n" + t.sort_values("shots", ascending=False).round(3).to_string())

print("\nGoal rate by distance (StatsBomb units ~ yards):")
dist_bins = pd.cut(df.distance, [0, 6, 12, 18, 25, 35, 200])
print(df.groupby(dist_bins, observed=True).is_goal.agg(["size", "mean"]).round(3).to_string())

# Shot map: non-penalty shots, goals highlighted
np_ = df[df.shot_type != "Penalty"]
pitch = VerticalPitch(pitch_type="statsbomb", half=True)
fig, ax = pitch.draw(figsize=(8, 7))
pitch.scatter(np_.x[~np_.is_goal], np_.y[~np_.is_goal], s=2, alpha=0.15, color="grey", ax=ax, label="no goal")
pitch.scatter(np_.x[np_.is_goal], np_.y[np_.is_goal], s=4, alpha=0.6, color="crimson", ax=ax, label="goal")
ax.legend(loc="lower left")
ax.set_title(f"{len(np_):,} non-penalty shots, 2015/16 top-4 leagues")
out = Path(__file__).parent / "shot_map.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nsaved {out}")
