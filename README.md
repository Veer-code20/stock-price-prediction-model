# SPPM — S&P 500 Market Dashboard
[![React](https://img.shields.io/badge/React-19-20232a?logo=react&logoColor=61DAFB)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6-646CFF?logo=vite&logoColor=white)](https://vite.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](https://www.python.org/)

**[Live Demo](https://sppm.vpatel2006.chatgpt.site)**

SPPM started as a stock price prediction project and grew into a full-stack S&P 500 market dashboard. It lets users explore stocks, view historical price charts, build a personal watchlist, and generate experimental trained price forecasts using Yahoo Finance market data.

## Trained Price Forecasts

In market mode, the Price Estimator now trains a Random Forest to predict the stock's future log return for the selected horizon (1, 5, 20, 60, 126, or 252 trading observations). It uses five years of Yahoo Finance split-adjusted daily closes, excludes today's bar, and builds features from past returns, moving-average distance, and volatility. Forecasts start from the latest completed daily close, rather than an intraday quote. Stocks with insufficient history return an explicit error.

Each response reports price MAE and RMSE against an unchanged-price baseline using three expanding chronological test folds. A horizon-sized gap keeps training outcomes before each test period. The model is then refit on all available labeled examples to predict the selected horizon. The API includes whether the model beat the baseline; these historical scores do not establish future accuracy. Test outcomes overlap, especially for longer horizons, and are not independent observations.

The endpoint returns one horizon-end prediction, not a fabricated daily path. Its range comes from the 10th–90th percentiles of historical log-return residuals; it is not a guaranteed confidence interval and need not contain the point prediction. Future dates count weekdays and do not account for exchange holidays. Hyperparameters are fixed rather than tuned on the evaluation folds. Training results are cached in a bounded in-memory cache.

Demo mode retains the synthetic formula-based estimator and does not claim trained-model performance. The original archived Random Forest included its target closing price among its input features and randomly split dates, so its scores did not evaluate future-price prediction. It is retained only as project history.

## Tech Stack

| Area | Technologies |
| --- | --- |
| Frontend | React 19, Vite 6, Recharts, Lucide React, JavaScript, CSS |
| Backend | Python 3, FastAPI, Uvicorn, HTTPX, scikit-learn |
| Market Data | Yahoo Finance |
| Testing | Vitest, Testing Library, Python `unittest` |

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py       # FastAPI routes and validation
│   │   ├── market.py     # Yahoo Finance requests, caching, and parsing
│   │   ├── data.py       # Synthetic demo estimator
│   │   ├── prediction.py # Trained forecasts and chronological evaluation
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

SPPM is a research and educational project. Yahoo Finance market data may be delayed, incomplete, rate-limited, or temporarily unavailable. Price forecasts are experimental model outputs and are not financial advice or recommendations to buy or sell securities.
