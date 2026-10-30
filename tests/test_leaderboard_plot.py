"""The README chart is generated from the leaderboard submissions and must stay well-formed and deterministic."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import leaderboard  # noqa: E402


def test_plots_are_wellformed_deterministic_and_list_every_entry():
    entries = leaderboard.load_entries()
    for theme in leaderboard.THEMES:
        svg = leaderboard.plot("2q1c", entries, theme)
        assert svg == leaderboard.plot("2q1c", entries, theme)
        root = ET.fromstring(svg)
        texts = {t.text for t in root.iter("{http://www.w3.org/2000/svg}text")}
        for meta, _, _ in entries:
            if meta["tier"] == "2q1c":
                assert meta["name"] in texts


def test_committed_outputs_are_current():
    stale = [p.name for p, text in leaderboard.outputs().items() if not p.exists() or p.read_text() != text]
    assert not stale, f"run python scripts/leaderboard.py render: {stale}"
