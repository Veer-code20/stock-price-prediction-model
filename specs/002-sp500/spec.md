# S&P 500 universe

Default market mode lists all members from a bundled, dated S&P 500 constituent snapshot. Include every listed share class, retain display symbols such as BRK.B, and translate dots to hyphens for Yahoo stock requests. Provide a reproducible snapshot refresh command with validation and source attribution.

GET /api/stocks returns searchable metadata for the full S&P 500 company list immediately without quote fan-out. GET /api/quotes accepts up to 20 validated, deduplicated member symbols. Quote calls retain the existing cache. Detail and forecast endpoints reject nonmembers before contacting Yahoo. Demo mode remains explicit with its existing fixture subset.

Frontend searches all S&P 500 stocks locally by symbol, company name, or sector, displays 12 results per page, and refreshes quotes for the current page, four highlights, and selected symbol about every 5 minutes. Cancel superseded batches; display loading/unavailable states accurately without calling refreshed quotes live streaming data. Preserve watchlists, detail, ranges, and forecasts.
