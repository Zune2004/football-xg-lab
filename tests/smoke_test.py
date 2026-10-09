"""Smoke test: the app loads, every tab and league renders, and the calculator gives sane numbers.
Run: python tests/smoke_test.py"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parents[1]

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
assert not at.exception, [e.value for e in at.exception]
league = at.segmented_control[0]
for lg in league.options:                      # every league, whichever data source it uses
    league.set_value(lg).run()
    assert not at.exception, (lg, [e.value for e in at.exception])


def calculator_xg():
    card = next(m.value for m in at.markdown if "shots like this go in" in m.value)
    return float(card.split("class='value")[1].split(">")[1].split("<")[0])


xg = calculator_xg()
assert 0.05 < xg < 0.5, f"12-yard central shot should be a decent chance, got {xg}"
at.slider[0].set_value(35).run()               # move the shot far out: xG must drop
far = calculator_xg()
assert far < xg, (far, xg)
print(f"xg-lab smoke test passed (12 yd: {xg:.2f}, 35 yd: {far:.2f})")
