import unittest
from unittest.mock import patch, Mock
import httpx
from fastapi.testclient import TestClient
from app import market
from app.main import app


def payload():
    return {'chart': {'result': [{'meta': {'regularMarketPrice': 110, 'regularMarketTime': 1735833600,
        'chartPreviousClose': 80, 'regularMarketDayHigh': 112, 'regularMarketDayLow': 99,
        'regularMarketVolume': 12345}, 'timestamp': [1735660800, 1735747200, 1735833600],
        'indicators': {'quote': [{'close': [90, 100, 110], 'open': [89, 91, 101]}]}}]}}


class MarketTests(unittest.TestCase):
    def setUp(self):
        market._cache.clear()

    def test_daily_change_uses_previous_session_not_range_start(self):
        result = market.parse(payload(), 'AAPL')
        self.assertEqual(result['stock']['change_pct'], 10)
        self.assertEqual(result['stats']['open'], 101)
        self.assertEqual(result['mode'], 'market')
        self.assertIn('quote_at', result)

    def test_null_candles_are_filtered_and_invalid_prices_rejected(self):
        raw = payload()
        raw['chart']['result'][0]['indicators']['quote'][0]['close'][0] = None
        self.assertEqual(len(market.parse(raw, 'AAPL')['history']), 2)
        raw['chart']['result'][0]['meta']['regularMarketPrice'] = float('nan')
        with self.assertRaises(ValueError):
            market.parse(raw, 'AAPL')

    @patch('app.market.httpx.get')
    def test_cache_coalesces_quotes_and_history(self, get):
        get.return_value = Mock(json=payload)
        market.chart('AAPL')
        market.chart('AAPL')
        self.assertEqual(get.call_count, 1)

    @patch.dict('os.environ', {'SPPM_DATA_MODE': 'market'})
    @patch('app.market.httpx.get', side_effect=httpx.ConnectError('offline'))
    def test_provider_failure_is_503_never_synthetic(self, get):
        client = TestClient(app)
        self.assertEqual(client.get('/api/stocks/AAPL').status_code, 503)
        self.assertEqual(client.post('/api/forecasts', json={'symbol': 'AAPL', 'horizon': 5}).status_code, 503)
        self.assertEqual(get.call_count, 1)

    @patch('app.market.chart', side_effect=ValueError('offline'))
    def test_catalog_marks_unavailable_quotes(self, chart):
        items = market.quotes(['AAPL', 'MSFT'])
        self.assertEqual(len(items), 2)
        self.assertTrue(all(item['price'] is None for item in items))
