import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.universe import MEMBERS


@patch.dict('os.environ', {'SPPM_DATA_MODE': 'market'})
class UniverseTests(unittest.TestCase):
    @patch('app.market.httpx.get')
    def test_full_catalog_does_not_fetch_prices(self, get):
        result = TestClient(app).get('/api/stocks').json()
        self.assertGreaterEqual(len(result['items']), 500)
        self.assertEqual(len(result['items']), len(MEMBERS))
        get.assert_not_called()
        self.assertTrue(all(s['quote_status'] == 'pending' for s in result['items']))
        self.assertEqual(TestClient(app).get('/api/stocks?q=berkshire').json()['items'][0]['symbol'], 'BRK.B')
        self.assertEqual(TestClient(app).get('/api/stocks?q=technology').json()['items'][0]['sector'], 'Information Technology')

    @patch('app.market.quotes', return_value=[])
    def test_batch_is_bounded_and_rejects_nonmembers(self, quotes):
        client = TestClient(app)
        self.assertEqual(client.get('/api/quotes?symbols=AAPL,AAPL,MSFT').status_code, 200)
        quotes.assert_called_once_with(['AAPL', 'MSFT'])
        self.assertEqual(client.get('/api/quotes?symbols=NOTREAL').status_code, 404)
        self.assertEqual(client.get('/api/quotes?symbols=' + ','.join(list(MEMBERS)[:21])).status_code, 422)
        self.assertEqual(client.get('/api/stocks/NOTREAL').status_code, 404)
        self.assertEqual(client.post('/api/forecasts', json={'symbol': 'NOTREAL', 'horizon': 5}).status_code, 404)

    @patch('app.market.parse', return_value={})
    @patch('app.market.httpx.get')
    def test_share_class_symbol_translation(self, get, parse):
        from app import market
        market._cache.clear()
        market.chart('BRK.B')
        self.assertTrue(get.call_args.args[0].endswith('/BRK-B'))
        self.assertEqual(parse.call_args.args[1], 'BRK.B')
