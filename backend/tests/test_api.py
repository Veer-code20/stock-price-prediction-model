import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.mode = patch.dict('os.environ', {'SPPM_DATA_MODE': 'demo'})
        self.mode.start()
        self.addCleanup(self.mode.stop)

    def test_search_and_empty(self):
        self.assertEqual(client.get('/api/stocks?q=aPpLe').json()['items'][0]['symbol'], 'AAPL')
        self.assertEqual(client.get('/api/stocks?q=zzzz').json()['items'], [])

    def test_ranges_are_consistent(self):
        short = client.get('/api/stocks/aapl?range=1M').json()
        long = client.get('/api/stocks/AAPL?range=1Y').json()
        self.assertEqual(len(short['history']), 22)
        self.assertEqual(len(long['history']), 252)
        self.assertEqual(short['history'], long['history'][-22:])
        self.assertEqual(short['history'][-1]['close'], short['stock']['price'])
        self.assertEqual(short['mode'], 'demo')
        self.assertIn('fifty_two_week_high', short['stats'])
        self.assertIn('fifty_two_week_low', short['stats'])

    def test_ytd_range(self):
        result = client.get('/api/stocks/AAPL?range=YTD').json()
        self.assertGreaterEqual(len(result['history']), 1)
        self.assertTrue(all(point['date'][:4] == result['history'][-1]['date'][:4] for point in result['history']))

    def test_forecast_contract(self):
        for horizon in (1, 5, 20, 60, 126, 252):
            result = client.post('/api/forecasts', json={'symbol':'MSFT', 'horizon':horizon}).json()
            self.assertEqual(len(result['points']), horizon)
            self.assertEqual(result['symbol'], 'MSFT')
            self.assertEqual(result['mode'], 'demo')
            self.assertIn('not a trained forecast', result['note'])
            self.assertEqual(result['method'], 'Weighted momentum and volatility estimate')
            self.assertTrue(all(p['lower'] < p['price'] < p['upper'] for p in result['points']))
            dates = [p['date'] for p in result['points']]
            self.assertEqual(dates, sorted(set(dates)))

    def test_invalid_requests(self):
        self.assertEqual(client.get('/api/stocks/UNKNOWN').status_code, 404)
        self.assertEqual(client.get('/api/stocks/AAPL?range=2Y').status_code, 422)
        self.assertEqual(client.post('/api/forecasts', json={'symbol':'AAPL','horizon':7}).status_code, 422)
        self.assertEqual(client.post('/api/forecasts', json={'symbol':'BAD','horizon':5}).status_code, 404)
        self.assertEqual(client.post('/api/forecasts', json={}).status_code, 422)

if __name__ == '__main__':
    unittest.main()
