# SPPM — S&P 500 Market Dashboard

[![React](https://img.shields.io/badge/React-19-20232a?logo=react&logoColor=61DAFB)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6-646CFF?logo=vite&logoColor=white)](https://vite.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](https://www.python.org/)

SPPM started as a stock price prediction project and grew into a full-stack S&P 500 market dashboard. It lets users explore stocks, view historical price charts, build a personal watchlist, and experiment with research-focused price scenarios using Yahoo Finance market data.

## From Prediction Model to Price Estimator

The original version of SPPM used a Random Forest model to experiment with stock price prediction. As the project grew, I shifted the focus from trying to predict an exact future price to building a larger market-data application.

The current Price Estimator uses recent returns, longer-term price movement, and volatility to generate illustrative price scenarios across different time periods. It is meant for research and experimentation rather than as a guaranteed prediction or trading signal. The original Random Forest work is preserved in `archive/` as part of the project's history.

## Tech Stack

| Area | Technologies |
| --- | --- |
| Frontend | React 19, Vite 6, Recharts, Lucide React, JavaScript, CSS |
| Backend | Python 3, FastAPI, Uvicorn, HTTPX |
| Market Data | Yahoo Finance |
| Testing | Vitest, Testing Library, Python `unittest` |

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py       # FastAPI routes and validation
│   │   ├── market.py     # Yahoo Finance requests, caching, and parsing
│   │   ├── data.py       # Demo data and Price Estimator logic
│   │   ├── universe.py   # S&P 500 universe loading
│   │   └── sp500.json    # Bundled S&P 500 list
│   ├── scripts/          # S&P 500 update utility
│   └── tests/            # Backend tests
├── frontend/
│   ├── src/
│   │   ├── components/   # Reusable UI components
│   │   ├── hooks/        # Data, watchlist, pagination, and estimator logic
│   │   ├── utils/        # Formatting helpers
│   │   ├── constants.js  # Shared frontend constants
│   │   └── App.jsx       # Main application layout and state
│   └── package.json
├── specs/                # Project specifications
└── archive/              # Original prediction-model work
```

The frontend is split into reusable components and focused hooks, while the backend separates the API, market-data logic, estimator, and S&P 500 universe. This keeps the codebase easier to test, debug, and build on.

## Run Locally

### Prerequisites

- Python 3 with `pip`
- Node.js with `npm`
- Git

### 1. Clone and install

```sh
git clone https://github.com/Veer-code20/stock-price-prediction-model.git
cd stock-price-prediction-model

python3 -m pip install --user -r backend/requirements.txt
npm --prefix frontend ci
```

### 2. Start the backend

From the repository root:

```sh
python3 -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001
```

The API runs at `http://127.0.0.1:8001`.

### 3. Start the frontend

In a second terminal from the repository root:

```sh
npm --prefix frontend run dev
```

Open `http://127.0.0.1:5173` in your browser.

## Disclaimer

SPPM is a research and educational project. Yahoo Finance market data may be delayed, incomplete, rate-limited, or temporarily unavailable. Price Estimator outputs are illustrative scenarios only and are not financial advice or recommendations to buy or sell securities.