import math
import unittest
from unittest.mock import patch, Mock
from datetime import date
from fastapi.testclient import TestClient
from app import prediction, data, market
from app.main import app


def history(count=1100):
    return [{'date': day, 'close': 100 * math.exp(.0002 * i + .025 * math.sin(i / 12))}
            for i, day in enumerate(data.weekdays(date(2020, 1, 1), count))]


class PredictionTests(unittest.TestCase):
    def test_features_cannot_see_future(self):
        closes = [p['close'] for p in history(100)]
        expected = prediction.features(closes, 60)
        closes[61:] = [999999] * 39
        self.assertEqual(expected, prediction.features(closes, 60))

    def test_purged_evaluation_and_price_baseline(self):
        result = prediction.forecast('AAPL', 20, history())
        stats = result['validation']
        self.assertEqual(stats['samples'], 300)
        self.assertTrue(all(f['train_target_end_index'] < f['test_start_index'] for f in stats['folds']))
        self.assertEqual(stats['beats_baseline'], stats['model_mae'] < stats['baseline_mae'])
        self.assertGreater(stats['baseline_mae'], 0)
        self.assertTrue(math.isfinite(result['projected_price']))
        self.assertEqual(result['points'][-1]['price'], result['projected_price'])
        self.assertLess(result['points'][0]['lower'], result['points'][0]['upper'])

    def test_all_supported_horizons(self):
        for horizon in (1, 5, 20, 60, 126, 252):
            result = prediction.forecast('AAPL', horizon, history())
            self.assertEqual(result['horizon'], horizon)
            self.assertGreater(result['validation']['samples'], 0)

    def test_insufficient_history_and_invalid_prices(self):
        with self.assertRaisesRegex(ValueError, 'Not enough history'):
            prediction.forecast('AAPL', 252, history(252))
        past = history()
        past[-1]['close'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            prediction.forecast('AAPL', 20, past)

    @patch.dict('os.environ', {'SPPM_DATA_MODE': 'market'})
    @patch('app.market.chart', return_value={})
    @patch('app.market.training_history', return_value=history())
    def test_market_route_uses_trained_model(self, training, chart):
        result = TestClient(app).post('/api/forecasts', json={'symbol': 'AAPL', 'horizon': 5})
        self.assertEqual(result.status_code, 200)
        self.assertIn('Random Forest', result.json()['method'])
        self.assertIn('validation', result.json())

    @patch.dict('os.environ', {'SPPM_DATA_MODE': 'market'})
    @patch('app.market.chart', return_value={})
    @patch('app.market.training_history', return_value=history(100))
    def test_short_history_returns_error_not_estimator(self, training, chart):
        result = TestClient(app).post('/api/forecasts', json={'symbol': 'AAPL', 'horizon': 252})
        self.assertEqual(result.status_code, 503)

    @patch('app.market.httpx.get')
    def test_training_history_excludes_incomplete_bar_and_caches(self, get):
        market._cache.clear()
        from datetime import datetime, timezone
        now = int(datetime.now(timezone.utc).timestamp())
        get.return_value = Mock(json=lambda: {'chart': {'result': [{'timestamp': [now - 86400, now],
            'indicators': {'quote': [{'close': [100, 101]}]}}]}})
        self.assertEqual(len(market.training_history('AAPL')), 1)
        market.training_history('AAPL')
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.kwargs['params']['range'], '5y')
