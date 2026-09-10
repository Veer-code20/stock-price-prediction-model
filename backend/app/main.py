import os
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from . import data, market
from .universe import MEMBERS, SNAPSHOT

app = FastAPI(title='SPPM API', version='0.1.0', description='Stock research workspace')


def demo():
    return os.getenv('SPPM_DATA_MODE', 'market') == 'demo'


def live_detail(symbol):
    try:
        return market.chart(symbol)
    except ValueError:
        raise HTTPException(status_code=503, detail='Market source unavailable. Try again shortly.')


def get_stock(symbol):
    stock = next((s for s in data.stocks() if s['symbol'] == symbol.upper()), None) if demo() else MEMBERS.get(symbol.upper())
    if stock is None:
        raise HTTPException(status_code=404, detail='Stock is not in the catalog.')
    return stock


@app.get('/api/health')
def health():
    return {'status': 'ok', 'mode': 'demo' if demo() else 'market'}


@app.get('/api/stocks')
def list_stocks(q: str = ''):
    query = q.strip().lower()
    items = data.stocks() if demo() else [dict(row, price=None, change_pct=None, quote_status='pending') for row in MEMBERS.values()]
    return {'mode': 'demo' if demo() else 'market', 'source': 'Synthetic data' if demo() else 'Yahoo Finance',
            'universe': 'demo' if demo() else 'S&P 500', 'membership_updated_at': SNAPSHOT['retrieved_at'],
            'items': [s for s in items if query in s['symbol'].lower() or query in s['name'].lower()]}


@app.get('/api/quotes')
def stock_quotes(symbols: str = Query(..., min_length=1, max_length=220)):
    selected = list(dict.fromkeys(s.strip().upper() for s in symbols.split(',')))
    if not 1 <= len(selected) <= 20:
        raise HTTPException(status_code=422, detail='Request between 1 and 20 symbols.')
    rows = [get_stock(symbol) for symbol in selected]
    return {'items': rows if demo() else market.quotes(selected)}


@app.get('/api/stocks/{symbol}')
def stock_detail(symbol: str, range: Literal['1D', '1W', '1M', '3M', '1Y'] = '3M'):
    stock = get_stock(symbol)
    if not demo():
        result = live_detail(stock['symbol'])
        if range in ('1D', '1W'):
            try:
                intraday = market.chart(stock['symbol'], True)
            except ValueError:
                raise HTTPException(status_code=503, detail='Intraday history unavailable. Try again shortly.')
            points = intraday['history']
            if range == '1D':
                points = [p for p in points if p['date'][:10] == points[-1]['date'][:10]]
        else:
            points = result['history'][-{'1M': 22, '3M': 66, '1Y': 252}[range]:]
        return dict(result, history=points)
    series = data.history(stock['symbol'])
    price = stock['price']
    return dict(mode='demo', stock=stock, history=series[-{'1D': 2, '1W': 5, '1M': 22, '3M': 66, '1Y': 252}[range]:],
                stats=dict(open=series[-2]['close'], high=round(price*1.012, 2), low=round(price*.986, 2), volume=48720000))


class ForecastRequest(BaseModel):
    symbol: str
    horizon: Literal[5, 20, 60]


@app.post('/api/forecasts')
def create_forecast(request: ForecastRequest):
    stock = get_stock(request.symbol)
    if demo():
        return data.forecast(stock['symbol'], request.horizon)
    result = live_detail(stock['symbol'])
    if len(result['history']) < 21:
        raise HTTPException(status_code=503, detail='Not enough history for a price scenario.')
    return data.forecast(stock['symbol'], request.horizon, past=result['history'], mode='market')
