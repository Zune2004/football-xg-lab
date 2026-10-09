"""xG Lab: an expected-goals model trained on 2015/16 club football, tested on recent tournaments; plus live
current-season team xG for the top-5 leagues and the Champions League. Run: streamlit run app.py"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

DATA = Path(__file__).parent / "xg" / "app_data"
SEASON = Path(__file__).parent / "xg" / "season"
GOAL_X, GOAL_Y, HALF_GOAL = 120, 40, 4   # StatsBomb pitch: 120 x 80, posts at y = 36 and 44
REAL_MIN = 0.9   # a league's table uses real xG once it covers 90% of its finished matches (sources never mixed)

ui.setup("xG Lab", "⚽")


@st.cache_data
def load_shots():
    return pd.read_parquet(DATA / "shots.parquet")


@st.cache_resource
def load_models():
    return lgb.Booster(model_file=str(DATA / "lgbm.txt")), json.loads((DATA / "logreg.json").read_text())


@st.cache_data(ttl=3600)
def load_season():
    lite = pd.read_csv(SEASON / "matches.csv") if (SEASON / "matches.csv").exists() else pd.DataFrame()
    real = pd.read_csv(SEASON / "hl_matches.csv") if (SEASON / "hl_matches.csv").exists() else pd.DataFrame()
    index = json.loads((SEASON / "hl_index.json").read_text()) if (SEASON / "hl_index.json").exists() else {}
    return lite, real, index


def predict(features):
    """Same blend as the trained model: average of LightGBM and the standardised logistic regression."""
    gbm, lr = load_models()
    x = np.array([[float(features[f]) for f in lr["features"]]])
    z = (x[0] - np.array(lr["mean"])) / np.array(lr["std"])
    p_lr = 1 / (1 + np.exp(-(z @ np.array(lr["coef"]) + lr["intercept"])))
    return float((gbm.predict(x)[0] + p_lr) / 2)


def pitch(fig, height=560):
    line = dict(color="rgba(229,233,240,.45)", width=1.5)
    shapes = [dict(type="rect", x0=0, y0=60, x1=80, y1=120, line=line, fillcolor="#0F2A1D", layer="below"),
              dict(type="rect", x0=18, y0=102, x1=62, y1=120, line=line), dict(type="rect", x0=30, y0=114, x1=50, y1=120, line=line),
              dict(type="line", x0=36, y0=120, x1=44, y1=120, line=dict(color=ui.TEXT, width=5)),
              dict(type="circle", x0=30, y0=50, x1=50, y1=70, line=line)]
    fig.update_layout(shapes=shapes, xaxis=dict(range=[-1, 81], visible=False),
                      yaxis=dict(range=[58, 121], visible=False, scaleanchor="x"), height=height,
                      margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.02))
    return fig


def team_table(lg, home_away_cols):
    """Per-team table from match rows: goals, xG for/against per match, finishing and keeping luck."""
    parts = []
    for home in (True, False):
        d = {"team": lg.home if home else lg.away, "gf": lg.home_goals if home else lg.away_goals,
             "ga": lg.away_goals if home else lg.home_goals}
        for k, (hc, ac) in home_away_cols.items():
            d[k] = lg[hc] if home else lg[ac]
            d[k + "_opp"] = lg[ac] if home else lg[hc]
        parts.append(pd.DataFrame(d))
    rows = pd.concat(parts)
    g = rows.groupby("team")
    t = g.agg(P=("gf", "size"), GF=("gf", "sum"), GA=("ga", "sum"))
    t["xg_for"], t["xg_against"] = g.xg.mean(), g.xg_opp.mean()
    t["xg_diff"] = t.xg_for - t.xg_against
    t["finishing"] = g.gf.sum() - g.xg.sum()
    t["keeping"] = g.xg_opp.sum() - g.ga.sum()
    if "poss" in home_away_cols:
        t["possession"] = g.poss.mean() * 100
        t["big_for"], t["big_against"] = g.big.sum(), g.big_opp.sum()
    return t.sort_values("xg_diff", ascending=False)


shots = load_shots()
metrics = json.loads((DATA / "metrics.json").read_text())
lite, real, index = load_season()
finished = {lg: sum(m["state"] == "Finished" for m in e["matches"]) for lg, e in index.items()}
have = real.dropna(subset=["home_xg"]).groupby("league").size().to_dict() if len(real) else {}

gap = np.mean([v["our_log_loss"] - v["statsbomb_log_loss"] for v in metrics.values()])
ui.hero("xG Lab", "How good was that chance? An expected-goals model built from free data that matches StatsBomb's "
                  "commercial xG, plus live chance-creation tables for this season.",
        tag="Live 2026/27 · Top-5 leagues + Champions League")
ui.kpis([("vs StatsBomb xG", f"+{gap:.3f}", "log loss gap on 3 recent tournaments (≈ identical)", "green"),
         ("Shots modelled", f"{37881 + len(shots):,}", "2015/16 club + WC 2022, Euro 2024, Copa 2024", "blue"),
         ("This season", f"{sum(have.values()) + (len(lite) if len(lite) else 0):,}", "match records collected so far", "amber"),
         ("Model", "LightGBM + LR", "29 features incl. defender & keeper positions", "")])

tab_now, tab_tourn, tab_calc, tab_test = st.tabs(["This season", "Tournaments", "xG calculator", "How good is it?"])

# ---------------------------------------------------------------- this season
with tab_now:
    leagues = [lg for lg in ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Champions League"]
               if lg in set(lite.league if len(lite) else []) | set(finished)]
    league = st.segmented_control("League", leagues, default=leagues[0] if leagues else None) or (leagues[0] if leagues else None)
    if league is None:
        st.info("Season data is being collected; check back soon.")
    else:
        cover = have.get(league, 0) / finished[league] if finished.get(league) else 0
        if cover >= REAL_MIN:
            lg = real[real.league == league].dropna(subset=["home_xg"])
            t = team_table(lg, {"xg": ("home_xg", "away_xg"), "poss": ("home_possession", "away_possession"),
                                "big": ("home_big_chances", "away_big_chances")})
            source, unit = f"Real match xG (Highlightly) · {len(lg)} matches", "xG"
        elif len(lite) and league in set(lite.league):
            lg = lite[(lite.league == league) & (lite.complete == 1)]
            t = team_table(lg, {"xg": ("home_xg_lite", "away_xg_lite")})
            source = f"xG-lite from shot locations · {len(lg)} matches · real xG {cover:.0%} collected, switches at 90%"
            unit = "xG-lite"
        else:
            t, source, unit = None, f"Real xG {cover:.0%} collected so far", "xG"
        if t is None or t.empty:
            st.info(f"{league}: {source}. The table appears once 90% of finished matches are in.")
        else:
            best, tight = t.xg_for.idxmax(), t.xg_against.idxmin()
            hot, cold = t.finishing.idxmax(), t.finishing.idxmin()
            ui.kpis([("Best attack", best, f"{t.xg_for.max():.2f} {unit} created per match", "green"),
                     ("Tightest defence", tight, f"{t.xg_against.min():.2f} {unit} allowed per match", "blue"),
                     ("Running hot", hot, f"{t.finishing.max():+.1f} goals above their chances", "amber"),
                     ("Wasteful", cold, f"{t.finishing.min():+.1f} goals below their chances", "rose")])
            mx, my = t.xg_for.mean(), t.xg_against.mean()
            colour = np.where(t.xg_diff > 0, ui.GREEN, ui.ROSE)
            fig = go.Figure(go.Scatter(
                x=t.xg_for, y=t.xg_against, mode="markers+text", text=t.index, textposition="top center",
                textfont=dict(size=11, color=ui.TEXT),
                marker=dict(size=12, color=colour, line=dict(width=1, color=ui.BG)),
                customdata=np.stack([t.GF, t.GA, t.finishing, t.keeping], axis=1),
                hovertemplate="<b>%{text}</b><br>" + unit + " for %{x:.2f} · against %{y:.2f} per match<br>"
                              "goals %{customdata[0]}-%{customdata[1]} · finishing %{customdata[2]:+.1f} · "
                              "keeping %{customdata[3]:+.1f}<extra></extra>"))
            fig.add_vline(x=mx, line=dict(color=ui.LINE, dash="dot"))
            fig.add_hline(y=my, line=dict(color=ui.LINE, dash="dot"))
            xr, yr = (t.xg_for.min(), t.xg_for.max()), (t.xg_against.min(), t.xg_against.max())
            for x, y, txt in [(xr[1], yr[0], "DOMINANT"), (xr[0], yr[0], "SOLID, BLUNT"), (xr[1], yr[1], "OPEN GAMES"),
                              (xr[0], yr[1], "STRUGGLING")]:
                fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(color=ui.MUTED, size=11),
                                   xanchor="right" if x == xr[1] else "left", yanchor="bottom" if y == yr[0] else "top")
            fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis=dict(title=f"Chances created ({unit} for per match) →"),
                              yaxis=dict(title=f"← Chances allowed ({unit} against per match)", autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
            show = t.rename(columns={"xg_for": f"{unit} for", "xg_against": f"{unit} against", "xg_diff": "difference"})
            cfg = {f"{unit} for": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=float(t.xg_for.max())),
                   f"{unit} against": st.column_config.NumberColumn(format="%.2f"),
                   "difference": st.column_config.NumberColumn(format="%+.2f"),
                   "finishing": st.column_config.NumberColumn("finishing (G − xG)", format="%+.1f"),
                   "keeping": st.column_config.NumberColumn("keeping (xGA − GA)", format="%+.1f")}
            if "possession" in show:
                cfg["possession"] = st.column_config.ProgressColumn("possession %", format="%.0f", min_value=0, max_value=100)
            st.dataframe(show, width="stretch", column_config=cfg)
            st.markdown(f"<div class='note'>{ui.esc(source)}. Finishing above zero = scoring more than the chances "
                        "suggest, which usually cools off; keeping above zero = conceding less than the chances suggest. "
                        "xG-lite: 0.140 per shot inside the box + 0.034 outside (calibrated on 37,881 StatsBomb shots; "
                        "team totals track full xG at r = 0.96). Data: Highlightly, TheSportsDB.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tournaments
with tab_tourn:
    c1, c2 = st.columns([1, 1])
    tourn = c1.segmented_control("Tournament", sorted(shots.tournament.unique()), default="Euro 2024") or "Euro 2024"
    t = shots[shots.tournament == tourn]
    team = c2.selectbox("Team", ["All teams"] + sorted(t.team.unique()))
    if team != "All teams":
        t = t[t.team == team]
    np_t = t[t.shot_type != "Penalty"]
    fin = t.is_goal.sum() - t.xg.sum()
    ui.kpis([("Shots", f"{len(t):,}", f"{(t.x >= 102).mean():.0%} from inside the box", "blue"),
             ("Goals", f"{int(t.is_goal.sum())}", f"{t.is_goal.mean():.1%} of shots", "green"),
             ("Expected goals", f"{t.xg.sum():.1f}", f"{np_t.xg.mean():.2f} xG per non-penalty shot", "amber"),
             ("Finishing", f"{fin:+.1f}", "goals above / below expected", "green" if fin >= 0 else "rose")])
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        for goal, colour, label in [(False, "rgba(229,233,240,.35)", "Missed"), (True, ui.AMBER, "Goal")]:
            g = t[t.is_goal == goal]
            fig.add_trace(go.Scatter(
                x=g.y, y=g.x, mode="markers", name=label,
                marker=dict(size=6 + 36 * g.xg.fillna(0), color=colour, line=dict(width=1, color=ui.BG)),
                customdata=np.stack([g.player, g.match, g.minute, g.xg.round(2), g.outcome], axis=1),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}, %{customdata[2]}'<br>"
                              "xG %{customdata[3]} · %{customdata[4]}<extra></extra>"))
        st.plotly_chart(pitch(fig), width="stretch")
        st.caption("Bigger dot = bigger chance. Hover any shot.")
    with right:
        perf = np_t.groupby("player").agg(shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"))
        perf = perf[perf.shots >= (3 if team != "All teams" else 6)]
        perf["diff"] = perf.goals - perf.xg
        show = pd.concat([perf.nlargest(6, "diff"), perf.nsmallest(6, "diff")])
        show = show[~show.index.duplicated()].sort_values("diff")
        fig = go.Figure(go.Bar(x=show["diff"], y=show.index, orientation="h",
                               marker=dict(color=np.where(show["diff"] > 0, ui.GREEN, ui.ROSE)),
                               customdata=np.stack([show.goals, show.xg, show.shots], axis=1),
                               hovertemplate="<b>%{y}</b><br>%{customdata[0]} goals from %{customdata[1]:.1f} xG "
                                             "(%{customdata[2]} shots)<extra></extra>"))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                          title=dict(text="Finishing vs chances", font=dict(size=15)), xaxis=dict(title="non-penalty goals − xG"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Over a few games, luck dominates: a big plus is more often a hot streak than a new level.")

# ---------------------------------------------------------------- calculator
with tab_calc:
    left, right = st.columns([1, 1])
    with left:
        dist_out = st.slider("Distance from the goal line (yards)", 1, 40, 12)
        across = st.slider("Distance from the centre (yards, + = right)", -30, 30, 0)
        body = st.segmented_control("Body part", ["Foot", "Head", "Other"], default="Foot") or "Foot"
        technique = st.selectbox("Technique", ["Normal", "Volley", "Half Volley", "Lob", "Overhead Kick", "Backheel",
                                                "Diving Header"])
        situation = st.pills("Situation", ["First time", "Under pressure", "1-on-1", "Counter", "Set piece",
                                           "Free kick"], selection_mode="multi") or []
        assist = st.selectbox("Assist", ["None", "Ground pass", "Through ball", "Cross", "Cut-back", "High ball"])
        defenders = st.slider("Defenders between the ball and the goal", 0, 6, 1)
        close = st.slider("Opponents within ~2.5 yards of the shooter", 0, 4, 1)
        gk_off = st.slider("Goalkeeper's distance off his line (yards)", 0, 15, 2)
    sx, sy = GOAL_X - dist_out, GOAL_Y + across
    d = float(np.hypot(GOAL_X - sx, GOAL_Y - sy))
    ang = float(np.arctan2(2 * HALF_GOAL * (GOAL_X - sx), (GOAL_X - sx) ** 2 + (sy - GOAL_Y) ** 2 - HALF_GOAL ** 2))
    unit_v = np.array([sx - GOAL_X, sy - GOAL_Y]) / max(d, 1e-6)
    gk = np.array([GOAL_X, GOAL_Y]) + unit_v * max(gk_off, 0.5)   # the keeper steps out along the shot line
    is_header = body == "Head"
    f = {"distance": d, "angle": ang, "log_distance": np.log1p(d), "is_header": is_header, "is_other_body": body == "Other",
         "header_x_distance": d * is_header,
         **{f"tech_{k.lower().replace(' ', '_')}": technique == k
            for k in ["Volley", "Half Volley", "Lob", "Overhead Kick", "Backheel", "Diving Header"]},
         "is_free_kick": "Free kick" in situation, "first_time": "First time" in situation,
         "under_pressure": "Under pressure" in situation, "one_on_one": "1-on-1" in situation,
         "from_counter": "Counter" in situation, "from_set_piece": "Set piece" in situation,
         "assisted": assist != "None", "assist_cross": assist == "Cross", "assist_through_ball": assist == "Through ball",
         "assist_cut_back": assist == "Cut-back", "assist_high": assist in ("Cross", "High ball"),
         "defenders_in_cone": defenders, "opponents_close": close,
         "gk_dist_to_goal": float(np.hypot(*(gk - [GOAL_X, GOAL_Y]))), "gk_dist_to_shot": float(np.hypot(*(gk - [sx, sy]))),
         "gk_in_cone": True, "gk_missing": False}
    xg = predict(f)
    with right:
        tone = "green" if xg >= 0.3 else "amber" if xg >= 0.1 else "rose"
        ui.kpis([("Expected goals", f"{xg:.2f}", f"about {xg * 100:.0f} in 100 shots like this go in", tone)])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[36, sy, 44], y=[120, sx, 120], fill="toself", mode="lines", line=dict(width=0),
                                 fillcolor="rgba(245,158,11,.18)", hoverinfo="skip", name="Shot cone"))
        fig.add_trace(go.Scatter(x=[sy], y=[sx], mode="markers", marker=dict(size=18, color=ui.AMBER), name="Shot", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[gk[1]], y=[gk[0]], mode="markers", marker=dict(size=14, color=ui.ROSE, symbol="square"),
                                 name="Goalkeeper", hoverinfo="skip"))
        st.plotly_chart(pitch(fig, 520), width="stretch")

# ---------------------------------------------------------------- how good is it
with tab_test:
    st.markdown("### Trained on 2015/16 club football. Tested on tournaments it never saw.")
    m = pd.DataFrame(metrics).T
    fig = go.Figure()
    fig.add_trace(go.Bar(name="This model", x=m.index, y=m.our_log_loss, marker_color=ui.GREEN,
                         text=m.our_log_loss.round(3), textposition="outside"))
    fig.add_trace(go.Bar(name="StatsBomb (commercial)", x=m.index, y=m.statsbomb_log_loss, marker_color=ui.BLUE,
                         text=m.statsbomb_log_loss.round(3), textposition="outside"))
    fig.update_layout(barmode="group", height=420, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(title="log loss (lower is better)", range=[0, float(m.our_log_loss.max()) * 1.25]),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(m.rename(columns={"shots": "Shots", "goals": "Goals", "our_xg": "Our xG", "statsbomb_xg": "StatsBomb xG",
                                   "our_log_loss": "Our log loss", "statsbomb_log_loss": "StatsBomb log loss",
                                   "our_auc": "Our AUC", "statsbomb_auc": "StatsBomb AUC"}), width="stretch")
    st.markdown("<div class='note'>Non-penalty shots. Training data: StatsBomb Open Data, 37,881 shots from the 2015/16 "
                "Premier League, La Liga, Serie A and Ligue 1. Features include where every defender and the goalkeeper "
                "stood at the moment of the shot. Open source; see the repository.</div>", unsafe_allow_html=True)
