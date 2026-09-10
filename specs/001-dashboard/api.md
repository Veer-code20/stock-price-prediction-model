# Dashboard API contract

Currency is USD. `mode` is `market` by default or `demo` when explicitly configured. Market dates are ISO dates/timestamps; demo sessions are weekdays, not an exchange holiday calendar.

- `GET /api/health` → `{status, mode}`.
- `GET /api/stocks?q=` → searchable catalog. Membership metadata and quote batching are specified in [002 — S&P 500](../002-sp500/spec.md).
- `GET /api/stocks/{symbol}?range=1D|1W|1M|3M|1Y` → `{mode, stock, history: [{date, close}], stats: {open, high, low, volume}}`. Market responses also include `source`, `fetched_at`, and `quote_at`; unavailable statistics may be null. Market 1D/1W ranges use intraday history; longer ranges use daily history. Demo ranges use fixture suffixes.
- `POST /api/forecasts`, body `{symbol, horizon: 5|20|60}` → `{mode, symbol, horizon, method, last_price, projected_price, points: [{date, price, lower, upper}], note}`. Bounds are illustrative, not statistical confidence intervals.

Unknown symbol: 404. Invalid range/horizon/body: 422. Unavailable market source or insufficient forecast history: 503. The client displays errors rather than substituting synthetic success responses.
