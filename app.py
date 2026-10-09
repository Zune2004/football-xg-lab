"""xG Lab: an expected-goals model trained on 2015/16 club football, tested on Euro 2024, Copa America 2024 and the
2022 World Cup (StatsBomb Open Data). Run: streamlit run app.py"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "xg" / "app_data"
GOAL_X, GOAL_Y, HALF_GOAL = 120, 40, 4   # StatsBomb pitch: 120 x 80, posts at y = 36 and 44

st.set_page_config(page_title="xG Lab", page_icon="⚽", layout="wide")


@st.cache_data
def load_shots():
    return pd.read_parquet(DATA / "shots.parquet")


@st.cache_resource
def load_models():
    lr = json.loads((DATA / "logreg.json").read_text())
    return lgb.Booster(model_file=str(DATA / "lgbm.txt")), lr


def predict(features: dict) -> float:
    """Same blend as the trained model: average of LightGBM and the standardised logistic regression."""
    gbm, lr = load_models()
    x = np.array([[float(features[f]) for f in lr["features"]]])
    p_gbm = gbm.predict(x)[0]
    z = (x[0] - np.array(lr["mean"])) / np.array(lr["std"])
    p_lr = 1 / (1 + np.exp(-(z @ np.array(lr["coef"]) + lr["intercept"])))
    return float((p_gbm + p_lr) / 2)


def half_pitch(fig):
    line = dict(color="#ffffff", width=2)
    shapes = [dict(type="rect", x0=0, y0=60, x1=80, y1=120, line=line),
              dict(type="rect", x0=18, y0=102, x1=62, y1=120, line=line),
              dict(type="rect", x0=30, y0=114, x1=50, y1=120, line=line),
              dict(type="line", x0=36, y0=120, x1=44, y1=120, line=dict(color="#ffffff", width=6)),
              dict(type="circle", x0=30, y0=50, x1=50, y1=70, line=line)]
    fig.update_layout(shapes=shapes, plot_bgcolor="#2e8b57", xaxis=dict(range=[-1, 81], visible=False),
                      yaxis=dict(range=[58, 121], visible=False, scaleanchor="x"), height=560,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.04))
    return fig


shots = load_shots()
metrics = json.loads((DATA / "metrics.json").read_text())

st.title("xG Lab")
st.caption("An expected-goals model trained on every shot of the 2015/16 Premier League, La Liga, Serie A and Ligue 1, "
           "then tested on tournaments it never saw. Data: StatsBomb Open Data.")

tab_now, tab_test, tab_map, tab_calc, tab_perf = st.tabs(["This season", "How good is it?", "Shot maps",
                                                         "xG calculator", "Finishing"])

with tab_now:
    season_file = Path(__file__).parent / "xg" / "season" / "matches.csv"
    st.subheader("2026/27 so far: who's creating chances, and who's riding luck?")
    if not season_file.exists():
        st.info("Season data is being collected; check back soon.")
    else:
        sm = pd.read_csv(season_file)
        league = st.selectbox("League", sorted(sm.league.unique()))
        lg = sm[sm.league == league]
        rows = pd.concat([
            lg.assign(team=lg.home, gf=lg.home_goals, ga=lg.away_goals, xgf=lg.home_xg_lite, xga=lg.away_xg_lite),
            lg.assign(team=lg.away, gf=lg.away_goals, ga=lg.home_goals, xgf=lg.away_xg_lite, xga=lg.home_xg_lite)])
        full = rows[rows.complete == 1]   # like-for-like: only matches with the full inside/outside shot split
        tab = rows.groupby("team").agg(P=("gf", "size"), GF=("gf", "sum"), GA=("ga", "sum"))
        per = full.groupby("team").agg(n=("gf", "size"), gf=("gf", "sum"), ga=("ga", "sum"), xgf=("xgf", "sum"), xga=("xga", "sum"))
        tab["xGF/match"] = per.xgf / per.n
        tab["xGA/match"] = per.xga / per.n
        tab["xG diff/match"] = tab["xGF/match"] - tab["xGA/match"]
        tab["finishing (G - xG)"] = per.gf - per.xgf
        tab["keeping (xGA - GA)"] = per.xga - per.ga
        tab = tab.sort_values("xG diff/match", ascending=False).round(2)
        st.dataframe(tab, width="stretch")
        fig = go.Figure(go.Scatter(x=tab["xGF/match"], y=tab["xGA/match"], mode="markers+text", text=tab.index,
                                   textposition="top center", marker=dict(size=10, color="#37003c")))
        fig.update_layout(xaxis_title="Chances created (xG-lite for per match)", height=520,
                          yaxis=dict(title="Chances allowed (xG-lite against per match)", autorange="reversed"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Top right = creates a lot, allows little. {len(lg)} matches, "
                   f"{lg.complete.mean():.0%} with the full shot split (xG columns use those only). "
                   "Positive finishing = scoring more than the chances suggest (often regresses); positive keeping = "
                   "conceding less than the chances suggest. "
                   "xG-lite = 0.140 per shot inside the box + 0.034 per shot outside it, calibrated on 37,881 StatsBomb "
                   "shots; on recent tournaments its team totals track full xG at r = 0.96. "
                   "Match data: TheSportsDB (free API). Champions League has no free shot data, so it isn't included.")

with tab_test:
    st.subheader("Tested on recent tournaments, against StatsBomb's own commercial xG")
    m = pd.DataFrame(metrics).T
    m = m.rename(columns={"shots": "Shots", "goals": "Goals", "our_xg": "Our xG", "statsbomb_xg": "StatsBomb xG",
                          "our_log_loss": "Our log loss", "statsbomb_log_loss": "StatsBomb log loss",
                          "our_auc": "Our AUC", "statsbomb_auc": "StatsBomb AUC"})
    st.dataframe(m, width="stretch")
    st.markdown("Log loss (lower is better) and AUC (higher is better) are within about 0.005 of StatsBomb's model on all "
                "three tournaments, though ours was trained on a decade-old club season with free data only. "
                "Penalties are excluded from these numbers.")

with tab_map:
    c1, c2, c3 = st.columns(3)
    tourn = c1.selectbox("Tournament", sorted(shots.tournament.unique()))
    t = shots[shots.tournament == tourn]
    team = c2.selectbox("Team", ["All"] + sorted(t.team.unique()))
    if team != "All":
        t = t[t.team == team]
    player = c3.selectbox("Player", ["All"] + sorted(t.player.unique()))
    if player != "All":
        t = t[t.player == player]
    # vertical half pitch: StatsBomb x (length) -> plot y, StatsBomb y (width) -> plot x
    fig = go.Figure()
    for goal, colour, label in [(False, "rgba(255,255,255,0.55)", "No goal"), (True, "#ffd23f", "Goal")]:
        g = t[t.is_goal == goal]
        fig.add_trace(go.Scatter(
            x=g.y, y=g.x, mode="markers", name=label,
            marker=dict(size=6 + 40 * g.xg.fillna(0), color=colour, line=dict(width=1, color="#1b4332")),
            customdata=np.stack([g.player, g.match, g.minute, g.xg.round(2), g.sb_xg.round(2), g.outcome], axis=1),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}, %{customdata[2]}'<br>"
                          "our xG %{customdata[3]} | StatsBomb %{customdata[4]}<br>%{customdata[5]}<extra></extra>"))
    st.plotly_chart(half_pitch(fig), width="stretch")
    a, b, c = st.columns(3)
    a.metric("Shots", len(t))
    b.metric("Goals", int(t.is_goal.sum()))
    c.metric("xG (our model)", f"{t.xg.sum():.1f}")

with tab_calc:
    st.subheader("What's the chance this shot goes in?")
    left, right = st.columns([1, 1])
    with left:
        dist_out = st.slider("Distance from the goal line (yards)", 1, 40, 12)
        across = st.slider("Distance from the centre (yards, + = right)", -30, 30, 0)
        body = st.radio("Body part", ["Foot", "Head", "Other"], horizontal=True)
        technique = st.selectbox("Technique", ["Normal", "Volley", "Half Volley", "Lob", "Overhead Kick", "Backheel",
                                                "Diving Header"])
        situation = st.multiselect("Situation", ["First time", "Under pressure", "One-on-one", "Counter-attack",
                                                 "From a set piece", "Direct free kick"])
        assist = st.selectbox("Assist", ["None", "Ground pass", "Through ball", "Cross", "Cut-back", "High ball"])
        defenders = st.slider("Defenders between the ball and the goal", 0, 6, 1)
        close = st.slider("Opponents within ~2.5 yards of the shooter", 0, 4, 1)
        gk_off = st.slider("Goalkeeper's distance off his line (yards)", 0, 15, 2)
    sx, sy = GOAL_X - dist_out, GOAL_Y + across
    d = float(np.hypot(GOAL_X - sx, GOAL_Y - sy))
    ang = float(np.arctan2(2 * HALF_GOAL * (GOAL_X - sx), (GOAL_X - sx) ** 2 + (sy - GOAL_Y) ** 2 - HALF_GOAL ** 2))
    unit = np.array([sx - GOAL_X, sy - GOAL_Y]) / max(d, 1e-6)
    gk = np.array([GOAL_X, GOAL_Y]) + unit * max(gk_off, 0.5)   # the keeper steps out along the shot line
    is_header = body == "Head"
    f = {"distance": d, "angle": ang, "log_distance": np.log1p(d), "is_header": is_header, "is_other_body": body == "Other",
         "header_x_distance": d * is_header,
         **{f"tech_{k.lower().replace(' ', '_')}": technique == k
            for k in ["Volley", "Half Volley", "Lob", "Overhead Kick", "Backheel", "Diving Header"]},
         "is_free_kick": "Direct free kick" in situation, "first_time": "First time" in situation,
         "under_pressure": "Under pressure" in situation, "one_on_one": "One-on-one" in situation,
         "from_counter": "Counter-attack" in situation, "from_set_piece": "From a set piece" in situation,
         "assisted": assist != "None", "assist_cross": assist == "Cross", "assist_through_ball": assist == "Through ball",
         "assist_cut_back": assist == "Cut-back", "assist_high": assist in ("Cross", "High ball"),
         "defenders_in_cone": defenders, "opponents_close": close,
         "gk_dist_to_goal": float(np.hypot(*(gk - [GOAL_X, GOAL_Y]))), "gk_dist_to_shot": float(np.hypot(*(gk - [sx, sy]))),
         "gk_in_cone": True, "gk_missing": False}
    xg = predict(f)
    with right:
        st.metric("Expected goals (xG)", f"{xg:.2f}", help="Roughly: this shot goes in this often")
        st.write(f"About **{xg * 100:.0f} in 100** shots like this are scored.")
        fig = go.Figure(go.Scatter(x=[sy], y=[sx], mode="markers", marker=dict(size=18, color="#ffd23f"),
                                   hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[gk[1]], y=[gk[0]], mode="markers", marker=dict(size=14, color="#ff6b6b", symbol="square"),
                                 name="Goalkeeper", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[36, sy, 44], y=[120, sx, 120], fill="toself", mode="lines", line=dict(width=0),
                                 fillcolor="rgba(255,210,63,0.18)", hoverinfo="skip", name="Shot cone"))
        st.plotly_chart(half_pitch(fig), width="stretch")

with tab_perf:
    st.subheader("Who finished above or below their chances?")
    tourn = st.selectbox("Tournament ", sorted(shots.tournament.unique()))
    np_shots = shots[(shots.tournament == tourn) & (shots.shot_type != "Penalty")]
    perf = np_shots.groupby(["player", "team"]).agg(shots=("xg", "size"), goals=("is_goal", "sum"),
                                                    xg=("xg", "sum")).reset_index()
    perf = perf[perf.shots >= st.slider("Minimum shots", 3, 20, 6)]
    perf["goals - xG"] = perf.goals - perf.xg
    st.dataframe(perf.sort_values("goals - xG", ascending=False).round(2), width="stretch", hide_index=True)
    st.caption("Non-penalty goals and xG. Over a few matches, luck dominates: a big positive number is more often a "
               "hot streak than a new level.")

st.divider()
st.caption("Data: StatsBomb Open Data (github.com/statsbomb/open-data). Model and app: open source, see the repository.")
