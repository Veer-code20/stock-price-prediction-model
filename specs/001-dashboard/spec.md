# 001 — Dashboard

Status: implemented; automated verification passed. Manual browser QA pending.

## Intent
A local stock research dashboard built with React, plain CSS, and Python FastAPI. This spec covers the original workspace and its dashboard redesign. S&P 500 coverage and quote batching are defined in [002 — S&P 500](../002-sp500/spec.md).

## Scope
Black and grey surfaces, system fonts, restrained copy, a chart-first layout, and green/red price movement. Search by company or ticker, select a stock, inspect history and statistics including open, today's high/low, volume, and 52-week high/low, save a browser-local watchlist, and explore illustrative price scenarios.

Yahoo Finance market data is the default, fetched server-side without credentials. This unofficial source may be delayed or rate-limited. Explicit `SPPM_DATA_MODE=demo` retains reproducible synthetic fixtures. Refresh visible market data quietly about every 5 minutes using a shared backend cache. The Price Estimator is a sidebar navigation item under Watchlist. It starts out of view, opens on user action, and uses a weighted momentum and volatility baseline with a clear non-advice warning.

## Acceptance criteria
- Search matches ticker/company case-insensitively and shows an explicit empty state.
- Selecting a stock updates identity, quote, chart, and statistics. History supports 1D, 1W, 1M, 3M, 1Y, and YTD, with 1D selected initially. All-time history is excluded because the app fetches bounded public Yahoo chart ranges.
- Price Estimator requests support 1 day, 1 week, 1 month, 3 months, 6 months, and 1 year, backed by 1/5/20/60/126/252 trading-day horizons. Changing either clears superseded results. Bounds reflect recent volatility. The UI must clearly state that the estimate is for learning/comparison only, is not financial advice, is not a buy/sell recommendation, and should not be used for real investing decisions.
- The header shows the regular NYSE core session window, 9:30 AM–4:00 PM ET, beside the market overview label instead of claiming live streaming data. Market source and refresh failures are visible without duplicating source/timestamp text under the chart. Synthetic prices are labeled and never silently substituted for failed market requests.
- When the Price Estimator navigation item is active, the right rail shows only the estimator; clicking the estimator header or returning to Discover/Watchlist restores the stock list.
- Loading, failure/retry, and empty states are supported. Superseded requests cannot overwrite newer selections.
- Watchlists persist in browser storage; malformed or unavailable storage does not break the interface.
- Layout adapts to narrow screens. Inputs have labels, controls support keyboard use, focus is visible, and charts have a text alternative.
- Original scripts/results remain in archive; startup and limitations are documented.
- Unsupported symbols/ranges/horizons are rejected. API tests, frontend interaction tests, and production build pass.

## Exclusions
Trading, authentication, all-exchange coverage, streaming exchange feeds, trained ML forecasts, and profit guarantees. S&P 500 membership and pagination belong to spec 002.
