"""Current-season team "xG-lite" for the top-5 leagues, from TheSportsDB's free API (public key, no signup).
Its free match stats include shots inside/outside the box (lists are capped at 5 entries, so lineups/timelines are
unusable and UCL has no stats). xG-lite = 0.140 x inside-box shots + 0.034 x outside-box shots: the average non-penalty
shot value per zone from 37,881 StatsBomb shots (2015/16), which held on WC 2022/Euro 2024/Copa 2024 (0.135 / 0.032).
Validation on those tournaments: team totals correlate 0.96 with full xG; vs goals 0.754 (full xG: 0.784).
Writes xg/season/matches.csv; resumable and incremental (only new finished matches cost requests).
Run: python xg/season_fetch.py"""
import csv
import json
import time
import urllib.request
from pathlib import Path

API = "https://www.thesportsdb.com/api/v1/json/123"   # TheSportsDB's public free key
LEAGUES = {4328: "Premier League", 4335: "La Liga", 4332: "Serie A", 4331: "Bundesliga", 4334: "Ligue 1"}
SEASON = "2026-2027"
XG_IN, XG_OUT = 0.140, 0.034
OUT = Path(__file__).parent / "season"
FIELDS = ["event_id", "league", "round", "date", "home", "away", "home_goals", "away_goals",
          "home_in", "home_out", "away_in", "away_out", "home_xg_lite", "away_xg_lite", "complete"]


def get(path):
    req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "football-xg-lab (github.com/Zune2004/football-xg-lab)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
    time.sleep(2.5)   # stay well under the free rate limit
    return out


def shots(stats):
    """{'home': (inside, outside), 'away': (...)} from whatever 5 stats the free tier returned."""
    s = {x["strStat"]: (x["intHome"], x["intAway"]) for x in stats}
    out = {}
    for k, side in ((0, "home"), (1, "away")):
        inside = s.get("Shots insidebox", (None, None))[k]
        outside = s.get("Shots outsidebox", (None, None))[k]
        if outside is None and s.get("Total Shots", (None, None))[k] is not None and inside is not None:
            outside = int(s["Total Shots"][k]) - int(inside)          # La Liga often gives totals instead
        out[side] = (None if inside is None else int(inside), None if outside is None else int(outside))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    path = OUT / "matches.csv"
    have = {r["event_id"]: r for r in csv.DictReader(path.open(encoding="utf-8"))} if path.exists() else {}
    for lid, league in LEAGUES.items():
        empty_rounds = 0
        for rnd in range(1, 39):
            events = get(f"eventsround.php?id={lid}&r={rnd}&s={SEASON}").get("events") or []
            finished = [e for e in events if e.get("intHomeScore") not in (None, "") and e.get("strStatus") in ("FT", "AET", "PEN", "Match Finished")]
            if not finished:
                empty_rounds += 1
                if empty_rounds >= 2:   # past the last played round
                    break
                continue
            empty_rounds = 0
            for e in finished:
                if e["idEvent"] in have and have[e["idEvent"]]["complete"] == "1":
                    continue
                st = shots(get(f"lookupeventstats.php?id={e['idEvent']}").get("eventstats") or [])
                (hi, ho), (ai, ao) = st["home"], st["away"]
                xg = lambda i, o: round(XG_IN * i + XG_OUT * o, 2) if i is not None and o is not None else ""
                have[e["idEvent"]] = {"event_id": e["idEvent"], "league": league, "round": rnd, "date": e["dateEvent"],
                                      "home": e["strHomeTeam"], "away": e["strAwayTeam"],
                                      "home_goals": e["intHomeScore"], "away_goals": e["intAwayScore"],
                                      "home_in": hi, "home_out": ho, "away_in": ai, "away_out": ao,
                                      "home_xg_lite": xg(hi, ho), "away_xg_lite": xg(ai, ao),
                                      "complete": "1" if None not in (hi, ho, ai, ao) else "0"}
            with path.open("w", newline="", encoding="utf-8") as f:   # save after every round: resumable
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(sorted(have.values(), key=lambda r: (r["league"], int(r["round"]), r["date"])))
        print(f"{league}: {sum(1 for r in have.values() if r['league'] == league)} matches")
    done = sum(r["complete"] == "1" for r in have.values())
    print(f"total {len(have)} finished matches, {done} with full shot split ({done / max(len(have), 1):.0%})")


if __name__ == "__main__":
    main()
