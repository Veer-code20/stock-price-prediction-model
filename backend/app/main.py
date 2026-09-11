import os
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from . import data, market
from .universe import MEMBERS, SNAPSHOT

Range = Literal['1D', '1W', '1M', '3M', '1Y', 'YTD']
Horizon = Literal[1, 5, 20, 60, 126, 252]
DAILY_RANGE_LENGTHS = {'1M': 22, '3M': 66, '1Y': 252}
DEMO_RANGE_LENGTHS = {'1D': 2, '1W': 5, **DAILY_RANGE_LENGTHS}

app = FastAPI(title='SPPM API', version='0.1.0', description='Stock research workspace')


def demo():
    return os.getenv('SPPM_DATA_MODE', 'market') == 'demo'


def live_detail(symbol):
    try:
        return market.chart(symbol)
    except ValueError:
        raise HTTPException(status_code=503, detail='Market source unavailable. Try again shortly.')


def get_stock(symbol):
    normalized = symbol.upper()
    stock = next((s for s in data.stocks() if s['symbol'] == normalized), None) if demo() else MEMBERS.get(normalized)
    if stock is None:
        raise HTTPException(status_code=404, detail='Stock is not in the catalog.')
    return stock


def matches(stock, query):
    values = [stock['symbol'], stock['name'], stock['sector']]
    return any(query in value.lower() for value in values)


def ytd_points(points):
    return [point for point in points if point['date'][:4] == points[-1]['date'][:4]]


def daily_points(points, history_range):
    return ytd_points(points) if history_range == 'YTD' else points[-DAILY_RANGE_LENGTHS[history_range]:]


def demo_points(points, history_range):
    return ytd_points(points) if history_range == 'YTD' else points[-DEMO_RANGE_LENGTHS[history_range]:]


@app.get('/api/health')
def health():
    return {'status': 'ok', 'mode': 'demo' if demo() else 'market'}


@app.get('/api/stocks')
def list_stocks(q: str = ''):
    query = q.strip().lower()
    items = data.stocks() if demo() else [dict(row, price=None, change_pct=None, quote_status='pending') for row in MEMBERS.values()]
    return {'mode': 'demo' if demo() else 'market', 'source': 'Synthetic data' if demo() else 'Yahoo Finance',
            'universe': 'demo' if demo() else 'S&P 500', 'membership_updated_at': SNAPSHOT['retrieved_at'],
            'items': [s for s in items if matches(s, query)]}


@app.get('/api/quotes')
def stock_quotes(symbols: str = Query(..., min_length=1, max_length=220)):
    selected = list(dict.fromkeys(s.strip().upper() for s in symbols.split(',') if s.strip()))
    if not 1 <= len(selected) <= 20:
        raise HTTPException(status_code=422, detail='Request between 1 and 20 symbols.')
    rows = [get_stock(symbol) for symbol in selected]
    return {'items': rows if demo() else market.quotes(selected)}


@app.get('/api/stocks/{symbol}')
def stock_detail(symbol: str, history_range: Range = Query('1D', alias='range')):
    stock = get_stock(symbol)
    if not demo():
        result = live_detail(stock['symbol'])
        if history_range in ('1D', '1W'):
            try:
                intraday = market.chart(stock['symbol'], True)
            except ValueError:
                raise HTTPException(status_code=503, detail='Intraday history unavailable. Try again shortly.')
            points = intraday['history']
            if not points:
                raise HTTPException(status_code=503, detail='Intraday history unavailable. Try again shortly.')
            if history_range == '1D':
                points = [point for point in points if point['date'][:10] == points[-1]['date'][:10]]
        else:
            points = daily_points(result['history'], history_range)
        return dict(result, history=points)

    series = data.history(stock['symbol'])
    price = stock['price']
    return dict(mode='demo', stock=stock, history=demo_points(series, history_range),
                stats=dict(open=series[-2]['close'], high=round(price * 1.012, 2), low=round(price * .986, 2),
                           volume=48720000,
                           fifty_two_week_high=round(price * 1.18, 2), fifty_two_week_low=round(price * .72, 2)))


class ForecastRequest(BaseModel):
    symbol: str
    horizon: Horizon


@app.post('/api/forecasts')
def create_forecast(request: ForecastRequest):
    stock = get_stock(request.symbol)
    if demo():
        return data.forecast(stock['symbol'], request.horizon)
    result = live_detail(stock['symbol'])
    if len(result['history']) < 21:
        raise HTTPException(status_code=503, detail='Not enough history for a price scenario.')
    return data.forecast(stock['symbol'], request.horizon, past=result['history'], mode='market')
