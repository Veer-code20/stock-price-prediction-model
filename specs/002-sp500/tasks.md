# Verification

- Bundled 503 securities from the DataHub constituent CSV retrieved 2026-09-10, with source and timestamp; validated refresh script included.
- Catalog listing/search performs no market-price requests. Quotes are requested in batches of at most 20 validated index members.
- BRK.B display symbol maps to Yahoo BRK-B; nonmembers rejected from quotes, detail, and forecasts.
- UI pages 12 results at a time, searches full membership, and refreshes the page, highlights, and selection.
- Backend tests run with system python3; no virtual environment activation or interpreter path needed.
- README documents direct Python setup, install commands, server commands, membership refresh, and explicit demo subset.
- Verification complete: 12 backend tests, 8 frontend tests, and production build pass (chart bundle size warning remains).
- Live BRK.B and COST quotes succeeded through the backend running under system python3. Test server stopped before handing over startup commands.
