# 001 — Dashboard

Status: implemented; automated verification passed. Manual browser QA pending.

## Intent
A local stock research dashboard built with React, plain CSS, and Python FastAPI. This spec covers the original workspace and its dashboard redesign. S&P 500 coverage and quote batching are defined in [002 — S&P 500](../002-sp500/spec.md).

## Scope
Black and grey surfaces, system fonts, restrained copy, a chart-first layout, and green/red price movement. Search by company or ticker, select a stock, inspect history and statistics, save a browser-local watchlist, and explore illustrative price scenarios.

Yahoo Finance market data is the default, fetched server-side without credentials. This unofficial source may be delayed or rate-limited. Explicit `SPPM_DATA_MODE=demo` retains reproducible synthetic fixtures. Refresh visible market data every 60 seconds using a shared backend cache. Forecasts remain simple trend illustrations.

## Acceptance criteria
- Search matches ticker/company case-insensitively and shows an explicit empty state.
- Selecting a stock updates identity, quote, chart, and statistics. History supports 1D, 1W, 1M, 3M, and 1Y.
- Forecast requests support roughly 1 week, 1 month, and 3 months, backed by 5/20/60 trading-day horizons. Changing either clears superseded results. Bounds never claim calibrated confidence or trained prediction accuracy.
- Market source, quote timestamp, and refresh failures are visible. Synthetic prices are labeled and never silently substituted for failed market requests.
- Loading, failure/retry, and empty states are supported. Superseded requests cannot overwrite newer selections.
- Watchlists persist in browser storage; malformed or unavailable storage does not break the interface.
- Layout adapts to narrow screens. Inputs have labels, controls support keyboard use, focus is visible, and charts have a text alternative.
- Original scripts/results remain in archive; startup and limitations are documented.
- Unsupported symbols/ranges/horizons are rejected. API tests, frontend interaction tests, and production build pass.

## Exclusions
Trading, authentication, all-exchange coverage, streaming exchange feeds, trained ML forecasts, and profit guarantees. S&P 500 membership and pagination belong to spec 002.
