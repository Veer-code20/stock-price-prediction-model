"""Cached public Yahoo chart data for local research; never falls back to fixtures."""
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
import httpx
from .universe import MEMBERS

_cache = {}
_locks = {(symbol, intraday): Lock() for symbol in MEMBERS for intraday in (False, True)}
CACHE_TTL_SECONDS = 295


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def chart(symbol, intraday=False):
    if symbol not in MEMBERS:
        raise ValueError('Not an S&P 500 member')
    key = (symbol, intraday)
    # Single-flight cache also bounds concurrent requests to the public provider.
    with _locks[key]:
        cached = _cache.get(key)
        if cached and time.monotonic() - cached[0] < CACHE_TTL_SECONDS:
            if isinstance(cached[1], Exception):
                raise ValueError('Market source unavailable')
            return cached[1]
        try:
            response = httpx.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol.replace(chr(46), chr(45))}',
                params={'range': '5d' if intraday else '1y', 'interval': '5m' if intraday else '1d'},
                headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
            response.raise_for_status()
            result = parse(response.json(), symbol, intraday)
            if not intraday:
                result = enrich_stats(symbol, result)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            _cache[key] = (time.monotonic(), error)
            raise ValueError('Market source unavailable') from error
        _cache[key] = (time.monotonic(), result)
        return result


def parse(payload, symbol, intraday=False):
    raw = payload['chart']['result'][0]
    meta = raw['meta']
    quote = raw['indicators']['quote'][0]
    points = [{'date': iso(t) if intraday else iso(t)[:10], 'close': round(c, 2)}
              for t, c in zip(raw['timestamp'], quote['close']) if number(c) and c > 0]
    price, previous = meta['regularMarketPrice'], meta.get('previousClose', meta.get('chartPreviousClose'))
    # chartPreviousClose is the start of the requested range, not yesterday.
    if not intraday and len(points) > 1:
        quote_day = iso(meta['regularMarketTime'])[:10]
        previous = points[-2]['close'] if points[-1]['date'] == quote_day else points[-1]['close']
    if not points or not number(price) or price <= 0 or not number(previous) or previous <= 0:
        raise ValueError('Invalid market prices')
    row = MEMBERS[symbol]
    return {'mode': 'market', 'source': 'Yahoo Finance', 'fetched_at': iso(time.time()),
            'quote_at': iso(meta['regularMarketTime']),
            'stock': dict(symbol=symbol, name=row['name'], sector=row['sector'], price=price,
                          change_pct=round((price / previous - 1) * 100, 2)),
            'history': points,
            'stats': dict(open=next((v for v in reversed(quote.get('open', [])) if number(v)), None),
                          high=meta.get('regularMarketDayHigh'), low=meta.get('regularMarketDayLow'),
                          volume=meta.get('regularMarketVolume'),
                          fifty_two_week_high=meta.get('fiftyTwoWeekHigh'),
                          fifty_two_week_low=meta.get('fiftyTwoWeekLow'))}


def raw_value(value):
    return value.get('raw') if isinstance(value, dict) and number(value.get('raw')) else None

def quote_fundamentals(symbol):
    response = httpx.get('https://query1.finance.yahoo.com/v7/finance/quote',
        params={'symbols': symbol.replace(chr(46), chr(45))}, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
    response.raise_for_status()
    quote = response.json()['quoteResponse']['result'][0]
    return {
        'fifty_two_week_high': quote.get('fiftyTwoWeekHigh'),
        'fifty_two_week_low': quote.get('fiftyTwoWeekLow'),
    }

def summary_fundamentals(symbol):
    response = httpx.get(f'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol.replace(chr(46), chr(45))}',
        params={'modules': 'price,summaryDetail'}, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
    response.raise_for_status()
    result = response.json()['quoteSummary']['result'][0]
    summary = result.get('summaryDetail', {})
    return {
        'fifty_two_week_high': raw_value(summary.get('fiftyTwoWeekHigh')),
        'fifty_two_week_low': raw_value(summary.get('fiftyTwoWeekLow')),
    }

def fundamentals(symbol):
    for fetch in (quote_fundamentals, summary_fundamentals):
        try:
            values = fetch(symbol)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            continue
        cleaned = {key: value for key, value in values.items() if number(value)}
        if cleaned:
            return cleaned
    return {}

def enrich_stats(symbol, result):
    stats = result.get('stats')
    if not isinstance(stats, dict):
        return result
    if all(number(stats.get(key)) for key in ('fifty_two_week_high', 'fifty_two_week_low')):
        return result
    try:
        extra = fundamentals(symbol)
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        # Fundamentals are nice-to-have. A failed quoteSummary call should not
        # turn a valid price/history response into "market unavailable."
        return result
    for key, value in extra.items():
        if stats.get(key) is None and value is not None:
            stats[key] = value
    return result


def quotes(symbols):
    def fetch(symbol):
        try:
            result = chart(symbol)
            return dict(result['stock'], quote_at=result['quote_at'], quote_status='ready')
        except ValueError:
            return dict(MEMBERS[symbol], price=None, change_pct=None, quote_status='unavailable')
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(fetch, symbols))
