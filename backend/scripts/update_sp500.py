"""Refresh the bundled S&P 500 membership snapshot: python3 backend/scripts/update_sp500.py."""
import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
TARGET = Path(__file__).resolve().parents[1] / 'app' / 'sp500.json'


def snapshot(text):
    rows = [{'symbol': row['Symbol'], 'name': row['Security'], 'sector': row['GICS Sector']}
            for row in csv.DictReader(io.StringIO(text))]
    if not 490 <= len(rows) <= 520 or len({r['symbol'] for r in rows}) != len(rows):
        raise ValueError('Unexpected constituent count or duplicate symbols')
    if any(not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}', r['symbol']) or not r['name'] or not r['sector'] for r in rows):
        raise ValueError('Invalid constituent metadata')
    return dict(source=SOURCE, retrieved_at=datetime.now(timezone.utc).isoformat(), items=sorted(rows, key=lambda r: r['symbol']))


if __name__ == '__main__':
    with urlopen(Request(SOURCE, headers={'User-Agent': 'SPPM/1.0'}), timeout=20) as response:
        result = snapshot(response.read().decode('utf-8-sig'))
    temporary = TARGET.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(TARGET)
    print(f"Saved {len(result['items'])} S&P 500 securities. Restart the backend to load the new membership.")
