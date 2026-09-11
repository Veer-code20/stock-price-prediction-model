# Stock Price Prediction Model

This project began as a machine learning model for predicting Apple stock prices with historical Yahoo Finance data. The original experiment used a **Random Forest Regressor** to compare predicted prices against actual closing prices and generate a prediction chart.

It has since been upgraded into a full S&P 500 dashboard with live quote fetching, historical charts, search, watchlists, and a cleaner black/gray interface.

## Current Version

The active app is a React + FastAPI stock dashboard. The frontend is split into small components for display and hooks for catalog loading, quote refreshes, stock detail, forecasts, pagination, and watchlist storage.

- Browse the S&P 500 instead of a small hardcoded stock list.
- Search by symbol, company name, or sector.
- View current prices and historical chart ranges.
- Track a local browser watchlist.
- Refresh visible prices quietly in the background.
- Use green/red up-and-down price styling inspired by trading apps.
- Estimate future prices with a weighted momentum and volatility baseline.
- Run without activating a virtual environment.

## Project Layout

```text
.
├── backend/      FastAPI API, S&P 500 snapshot, market data, tests
├── frontend/     React dashboard, modular components/hooks, styling, tests
├── specs/        Specs for major dashboard and S&P 500 features
├── archive/      Original ML experiment files and old project history
├── README.md
└── .gitignore
```

## Run the Dashboard

Install dependencies:

```sh
cd /Users/veerpatel/proj/sppm
python3 -m pip install --user -r backend/requirements.txt

cd frontend
npm ci
```

Start the backend:

```sh
cd /Users/veerpatel/proj/sppm
python3 -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001
```

Start the frontend in a second terminal:

```sh
cd /Users/veerpatel/proj/sppm/frontend
SPPM_API_URL=http://127.0.0.1:8001 npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Data

The dashboard includes a bundled S&P 500 snapshot with 503 securities, stored in `backend/app/sp500.json`. The count is above 500 because the index includes multiple share classes.

Refresh the S&P 500 snapshot:

```sh
python3 backend/scripts/update_sp500.py
```

Live prices and history use Yahoo Finance's public chart endpoint. The dashboard loads the full S&P 500 company list first, then fetches prices for the visible page, highlight row, and selected stock. Visible prices refresh quietly about every 5 minutes. Prices may be delayed, rate-limited, or temporarily unavailable.

The price estimate uses weighted recent returns, longer trend context, and volatility to project 1 day, 1 week, 1 month, 3 months, 6 months, or 1 year ahead. It is a research estimate, not trading advice.

## Original Model

The original model files were moved out of the root and into `archive/` so the active codebase stays clean.

- `archive/original-root/`: original local experiment files
- `archive/github-root/`: files that used to sit at the GitHub repo root
- `archive/original-project/`: older project folder

The old experiment downloaded Apple stock data, trained a Random Forest model, generated predictions, and saved a plot. It is kept as project history, while the current dashboard lives in `backend/` and `frontend/`.

## Verify

Backend tests:

```sh
cd /Users/veerpatel/proj/sppm/backend
python3 -m unittest discover -s tests -v
```

Frontend tests and build:

```sh
cd /Users/veerpatel/proj/sppm/frontend
npm test
npm run build
```

## Specs

Specs are only used for major feature work:

- `specs/001-dashboard/`
- `specs/002-sp500/`

Routine fixes, styling updates, and small cleanup changes do not need new specs.
