# Dashboard implementation plan

- React, Vite, and plain CSS; access the Python API through the development proxy. Keep preferences in local state and localStorage.
- FastAPI with separate market and deterministic demo providers, request validation, and JSON responses.
- Fetch Yahoo price history server-side, cache per symbol for 55 seconds, and poll visible frontend data every 60 seconds.
- Cancel superseded requests, invalidate forecasts when stock/horizon changes, and show failures without fabricated fallback prices.
- Preserve archived projects and provenance notes.
- Verify API contracts and frontend state behavior with focused tests, then build. Browser QA is recorded separately and only claimed when performed.
- Use system python3 directly; virtual environment activation is not required. Setup commands are in the root README.
