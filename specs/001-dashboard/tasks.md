# Tasks and verification

- [x] Write scope, acceptance criteria, plan, and API contract before implementation.
- [x] Preserve and reorganize original project.
- [x] Implement demo API and contract tests.
- [x] Implement responsive React/CSS workspace and state tests.
- [x] Run automated checks and record results.
- [x] Document local startup and known limitations.

Manual browser checklist: search/no matches; stock/range switch; forecast/horizon switch; watchlist persistence; API offline/retry; keyboard traversal; 375px and desktop layout. Record only checks actually performed.


## Verification — 2026-09-09

- Backend: 4 unittest contract tests passed (search/empty, consistent history ranges, all forecast horizons, invalid requests).
- Frontend: 5 Vitest/Testing Library tests passed (search/empty, watchlist persistence and horizon reset, superseded forecast handling, catalog error/retry, malformed storage).
- Frontend production build passed. Nonblocking warning: main bundle is approximately 599 kB before gzip, primarily charting dependencies; optimize loading in a later performance slice.
- Responsive CSS, focus styling and chart text alternative implemented. Manual browser visual/keyboard QA not performed; AC7 remains manually unverified.
- Preserved original assets and scripts under archive. No commits, pushes, or deployment.
- Project moved externally from Documents/proj to ~/proj during work. Renamed #1-SPPM to sppm with approval because Vite failed on the # path.


## Dashboard redesign and market integration

Verification — 2026-09-10

- [x] Black/grey surfaces, system fonts, restrained copy, chart-first desktop/mobile layout.
- [x] Green/red directional movement; chart reflects selected history movement.
- [x] Search, persisted watchlist, history selection, illustrative scenarios preserved.
- [x] Server-side Yahoo chart integration, 55-second per-symbol cache and concurrent catalog requests.
- [x] 60-second visible-page polling, cancellation, refresh failure labels and recovery.
- [x] Unavailable quotes are not substituted with synthetic values. Demo is explicit.
- [x] 9 backend tests and 7 frontend tests pass; production build passes (existing chart-bundle size warning).
- [x] Real AAPL 1D response verified, including current quote timestamp and 5-minute history. All 12 catalog quotes verified through the Vite API proxy.
- [x] README updated with startup commands, alternate port, and demo mode.
- [ ] Screenshot verification: browser runtime reported no connected browsers; desktop/mobile CSS has not been visually inspected.
