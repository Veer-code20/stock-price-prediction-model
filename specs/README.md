# Spec-driven development

Use SDD for major features or substantial changes in product behavior. Do not create a new numbered spec for routine bug fixes, styling adjustments, refactors, dependency updates, or small improvements to an existing feature. Make those changes directly and update the existing spec only if its documented behavior needs to change.

## Current specs

1. [Dashboard](001-dashboard/spec.md): the original workspace, visual redesign, charts, watchlists, market-data integration, and illustrative forecasts.
2. [S&P 500](002-sp500/spec.md): index membership, full-catalog search, pagination, and bounded quote fetching.

## Future major features

For a new major feature, define scope and acceptance criteria before implementation in the next numbered folder. Add plans or API contracts only when they help; verify relevant criteria and record the results. Extend an existing spec when the work belongs to its scope instead of creating a separate spec for every iteration.

Keep only these two specs for the current product. Add future specs when major new features are actually undertaken, not as placeholder roadmap folders. Never describe planned capabilities or unperformed verification as completed.
