"""Current-season REAL team xG (and more) for the top-5 leagues + Champions League from Highlightly's free plan
(100 requests/day; its match statistics include Expected Goals / Expected Assists / big chances / possession...).
Terms (highlightly.net/terms, 6.1): data may be used, stored and distributed in your apps; no API resale/proxy.
Resumable within the daily quota: fixture lists are cached and refreshed every 2 days; each finished match costs
1 request, once. Writes xg/season/hl_matches.csv. Run: HIGHLIGHTLY_KEY=... python xg/hl_fetch.py"""
import csv
import datetime as dt
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

API = "https://soccer.highlightly.net"
LEAGUES = {"Premier League": 33973, "La Liga": 119924, "Serie A": 115669, "Bundesliga": 67162, "Ligue 1": 52695,
           "Champions League": 2486}
SEASON = 2026
KEEP_SPARE = 3   # leave a few of the 100 daily requests unused
OUT = Path(__file__).parent / "season"
STATS = {"Expected Goals": "xg", "Expected Assists": "xa", "Big Chances Created": "big_chances",
         "Shots on target": "shots_on", "Shots off target": "shots_off", "Blocked shots": "shots_blocked",
         "Possession": "possession", "Corners": "corners", "Attacks": "attacks", "Crosses": "crosses",
         "Passes": "passes", "Successful Passes": "passes_ok"}
FIELDS = ["match_id", "league", "round", "date", "home", "away", "home_goals", "away_goals"] + \
         [f"{side}_{v}" for side in ("home", "away") for v in STATS.values()]


class Quota(Exception):
    pass


remaining = 100


def get(path):
    global remaining
    if remaining <= KEEP_SPARE:
        raise Quota()
    req = urllib.request.Request(API + path, headers={"x-rapidapi-key": os.environ["HIGHLIGHTLY_KEY"]})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            remaining = int(r.headers.get("x-ratelimit-requests-remaining", remaining - 1))
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise Quota()
        raise


def main():
    OUT.mkdir(exist_ok=True)
    idx_file, out_file = OUT / "hl_index.json", OUT / "hl_matches.csv"
    index = json.loads(idx_file.read_text()) if idx_file.exists() else {}
    rows = {r["match_id"]: r for r in csv.DictReader(out_file.open(encoding="utf-8"))} if out_file.exists() else {}
    today = dt.date.today().isoformat()
    try:
        for league, lid in LEAGUES.items():   # 1) fixture lists, refreshed every 2 days
            entry = index.get(league)
            if entry and (dt.date.fromisoformat(today) - dt.date.fromisoformat(entry["listed"])).days < 2:
                continue
            matches, off = [], 0
            while True:
                d = get(f"/matches?leagueId={lid}&season={SEASON}&limit=100&offset={off}")
                matches += d.get("data", [])
                off += 100
                if off >= (d.get("pagination") or {}).get("totalCount", 0):
                    break
            index[league] = {"listed": today, "matches": [
                {"id": m["id"], "round": m.get("round"), "date": m["date"][:10], "home": m["homeTeam"]["name"],
                 "away": m["awayTeam"]["name"], "state": (m.get("state") or {}).get("description"),
                 "score": ((m.get("state") or {}).get("score") or {}).get("current")} for m in matches]}
            idx_file.write_text(json.dumps(index))
        for league in LEAGUES:   # 2) stats for each finished match, once
            for m in index.get(league, {}).get("matches", []):
                if m["state"] != "Finished" or str(m["id"]) in rows or not m["score"]:
                    continue
                try:
                    st = get(f"/statistics/{m['id']}")
                except urllib.error.HTTPError as e:   # their side fails now and then: skip, retry on a later run
                    print(f"  {league} {m['home']} v {m['away']}: HTTP {e.code}, will retry")
                    continue
                vals = {}
                for side, team in zip(("home", "away"), st if isinstance(st, list) else []):
                    for x in team.get("statistics", []):
                        if x["displayName"] in STATS:
                            vals[f"{side}_{STATS[x['displayName']]}"] = x["value"]
                hg, ag = (s.strip() for s in m["score"].split("-"))
                rows[str(m["id"])] = {"match_id": m["id"], "league": league, "round": m["round"], "date": m["date"],
                                      "home": m["home"], "away": m["away"], "home_goals": hg, "away_goals": ag, **vals}
    except Quota:
        print(f"daily quota reached ({remaining} left); resuming next run")
    except Exception as e:   # anything else: keep what we have
        print(f"stopped early: {type(e).__name__}: {e}")
    with out_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: (r["league"], r["date"])))
    fin = {lg: sum(m["state"] == "Finished" for m in e["matches"]) for lg, e in index.items()}
    got = {lg: sum(r["league"] == lg for r in rows.values()) for lg in LEAGUES}
    with_xg = sum(1 for r in rows.values() if r.get("home_xg") not in (None, ""))
    print("finished matches:", fin, "| with stats:", got, f"| with xG: {with_xg}/{len(rows)} | requests left: {remaining}")


if __name__ == "__main__":
    main()
