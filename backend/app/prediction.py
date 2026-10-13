"""Direct horizon return prediction with expanding-window, purged evaluation."""
from functools import lru_cache
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from .data import weekdays
from datetime import date, timedelta

LOOKBACK = 60


def features(closes, index):
    """Only observations available at the forecast origin are used."""
    window = np.asarray(closes[index - LOOKBACK:index + 1])
    returns = np.diff(np.log(window))
    return [math.log(window[-1] / window[-1 - lag]) for lag in (1, 5, 10, 20, 60)] + [
        float(np.std(returns[-20:])), float(np.std(returns)),
        float(window[-1] / np.mean(window[-20:]) - 1),
    ]


def model():
    return RandomForestRegressor(n_estimators=80, max_depth=6, min_samples_leaf=10,
                                 random_state=42, n_jobs=1)


@lru_cache(maxsize=32)
def evaluate_and_predict(closes, horizon):
    if len(closes) < LOOKBACK + 2 * horizon + 300:
        raise ValueError('Not enough history for training and chronological evaluation at this timeframe.')
    origins = np.arange(LOOKBACK, len(closes) - horizon)
    x = np.asarray([features(closes, int(i)) for i in origins])
    y = np.asarray([math.log(closes[i + horizon] / closes[i]) for i in origins])
    predicted, actual, baseline, residuals = [], [], [], []
    folds = []
    # Purge horizon observations so training targets end before each test starts.
    splitter = TimeSeriesSplit(n_splits=3, gap=horizon, test_size=min(100, (len(x) - horizon - 100) // 3))
    for train, test in splitter.split(x):
        fitted = model().fit(x[train], y[train])
        estimates = fitted.predict(x[test])
        base = np.asarray([closes[i] for i in origins[test]])
        truth = np.asarray([closes[i + horizon] for i in origins[test]])
        prices = base * np.exp(estimates)
        predicted.extend(prices.tolist())
        actual.extend(truth.tolist())
        baseline.extend(base.tolist())
        residuals.extend((y[test] - estimates).tolist())
        folds.append({'train_target_end_index': int(origins[train[-1]] + horizon),
                      'test_start_index': int(origins[test[0]]), 'samples': len(test)})
    actual = np.asarray(actual)
    mae = float(np.mean(np.abs(actual - predicted)))
    baseline_mae = float(np.mean(np.abs(actual - baseline)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((actual - baseline) ** 2)))
    fitted = model().fit(x, y)
    estimate = float(fitted.predict([features(closes, len(closes) - 1)])[0])
    low, high = np.quantile(residuals, [.1, .9])
    return estimate, float(low), float(high), {
        'model_mae': round(mae, 2), 'baseline_mae': round(baseline_mae, 2),
        'model_rmse': round(rmse, 2), 'baseline_rmse': round(baseline_rmse, 2),
        'beats_baseline': mae < baseline_mae,
        'improvement_pct': round((1 - mae / baseline_mae) * 100, 2) if baseline_mae else None,
        'samples': len(actual), 'folds': folds,
        'evaluation': '3 expanding chronological folds; horizon-sized purge; overlapping test outcomes',
    }


def forecast(symbol, horizon, past):
    if any(not math.isfinite(p['close']) or p['close'] <= 0 for p in past):
        raise ValueError('Invalid training prices.')
    dates = [p['date'] for p in past]
    if dates != sorted(set(dates)):
        raise ValueError('Training history must have unique chronological dates.')
    closes = tuple(float(p['close']) for p in past)
    estimate, low, high, validation = evaluate_and_predict(closes, horizon)
    last = closes[-1]
    price = round(last * math.exp(estimate), 2)
    end = weekdays(date.fromisoformat(dates[-1]) + timedelta(days=1), horizon)[-1]
    return dict(mode='market', symbol=symbol, horizon=horizon,
                method='Random Forest future-return prediction', last_price=last, projected_price=price,
                points=[dict(date=end, price=price, lower=round(last * math.exp(estimate + low), 2),
                             upper=round(last * math.exp(estimate + high), 2))],
                validation=validation, training_start=dates[0], training_end=dates[-1],
                training_observations=len(closes),
                note='Trained on split-adjusted closes. Range uses historical 10th–90th percentile forecast errors; coverage is not guaranteed. Dates count weekdays, excluding no exchange holidays. Historical results do not establish future accuracy.')
