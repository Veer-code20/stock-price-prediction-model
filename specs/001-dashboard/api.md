# Dashboard API contract

Currency is USD. `mode` is `market` by default or `demo` when explicitly configured. Market dates are ISO dates/timestamps; forecast horizons are trading days.

- `GET /api/health` → `{status, mode}`.
- `GET /api/stocks?q=` → searchable catalog. Membership metadata and quote batching are specified in [002 — S&P 500](../002-sp500/spec.md).
- `GET /api/stocks/{symbol}?range=1D|1W|1M|3M|1Y|YTD` → `{mode, stock, history: [{date, close}], stats: {open, high, low, volume, fifty_two_week_high, fifty_two_week_low}}`. Market responses also include `source`, `fetched_at`, and `quote_at`; unavailable statistics may be null. Market 1D/1W ranges use intraday Yahoo chart history; longer ranges use daily Yahoo chart history. 52-week high/low come from Yahoo chart metadata when available, with a best-effort Yahoo quoteSummary fallback when chart metadata omits fundamentals. Fallback failures do not fail the price/history response. Demo ranges use fixture suffixes.
- `POST /api/forecasts`, body `{symbol, horizon: 1|5|20|60|126|252}` → `{mode, symbol, horizon, method, last_price, projected_price, points: [{date, price, lower, upper}], note}`. Horizon values represent 1 day, 1 week, 1 month, 3 months, 6 months, and 1 year of trading days. Bounds are illustrative, not statistical confidence intervals.

Unknown symbol: 404. Invalid range/horizon/body: 422. Unavailable market source or insufficient forecast history: 503. The client displays errors rather than substituting synthetic success responses.
