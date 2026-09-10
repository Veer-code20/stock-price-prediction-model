"""Bundled membership metadata, independent of the market-price connection."""
import json
from pathlib import Path

SNAPSHOT = json.loads(Path(__file__).with_name('sp500.json').read_text())
MEMBERS = {row['symbol']: row for row in SNAPSHOT['items']}
