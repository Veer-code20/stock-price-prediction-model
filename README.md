# Stock Price Prediction Model

A stock research dashboard built from an older price-prediction experiment. The current app uses a React frontend and FastAPI backend to browse the S&P 500, fetch live market prices, track a local watchlist, and show simple forecast scenarios.

The original Random Forest scripts are still here, but they live in `archive/` now. The active application code is only in `frontend/` and `backend/`.

## What it does

- Browse a bundled S&P 500 universe instead of a small demo list.
- Search companies by symbol, name, or sector.
- Fetch current quotes and historical chart data from Yahoo Finance's public chart endpoint.
- Refresh visible prices automatically every 60 seconds.
- Save a watchlist locally in the browser.
- View 1D, 1W, 1M, 3M, and 1Y chart ranges.
- Run without activating a virtual environment.

## Project layout

```text
.
├── backend/      FastAPI app, S&P 500 universe, market-data provider, tests
├── frontend/     React dashboard, styling, frontend tests
├── specs/        SDD docs for major features only
├── archive/      Original ML scripts, plots, CSV files, and old project notes
├── README.md
└── .gitignore
```

## Run locally

Requires Python 3.9+ and Node.js 22+ with npm.

Install dependencies:

```sh
cd /Users/veerpatel/proj/sppm
python3 -m pip install --user -r backend/requirements.txt

cd frontend
npm ci
```

Start the backend in one terminal:

```sh
cd /Users/veerpatel/proj/sppm
python3 -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001
```

Start the frontend in another terminal:

```sh
cd /Users/veerpatel/proj/sppm/frontend
SPPM_API_URL=http://127.0.0.1:8001 npm run dev
```

Open the Vite URL, usually:

```text
http://127.0.0.1:5173
```

If `npm` is not found on this Mac, use the temporary Node runtime that was downloaded earlier:

```sh
export PATH="/private/tmp/node-v22.14.0-darwin-arm64/bin:$PATH"
```

Run that before `npm ci` or `npm run dev`, or install Node.js normally.

## Data notes

The dashboard ships with a 503-security S&P 500 snapshot retrieved on September 10, 2026. The count is above 500 because the index includes multiple share classes. The snapshot is stored in `backend/app/sp500.json`.

Refresh the S&P 500 snapshot:

```sh
python3 backend/scripts/update_sp500.py
```

Then restart the backend.

Live quotes use Yahoo Finance's public, unofficial chart endpoint. Prices may be delayed, rate-limited, or temporarily unavailable. The backend caches each symbol for 55 seconds and does not replace failed market requests with fake prices.

Forecasts are simple trend illustrations. They are not trained trading signals or calibrated predictions.

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

This repo uses spec-driven development only for major feature work.

- `specs/001-dashboard/`: dashboard behavior and UI
- `specs/002-sp500/`: S&P 500 universe and live market-data features

Routine fixes, styling polish, dependency updates, and small cleanup work do not need new specs.

## Archive

The old stock-prediction work is preserved under `archive/`.

- `archive/original-root/`: original top-level files from this workspace
- `archive/github-root/`: files that used to live at the GitHub repo root
- `archive/original-project/`: older project folder, with nested Git metadata removed

The archived model trained a Random Forest Regressor on Apple stock data and generated a CSV plus plot. It is kept as project history and a learning artifact.
