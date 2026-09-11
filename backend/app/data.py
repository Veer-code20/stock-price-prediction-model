"""Reproducible synthetic fixtures. No market-data or ML claims."""
import math
from datetime import date, timedelta

CATALOG = [
    ('AAPL', 'Apple Inc.', 'Technology', 228.40),
    ('MSFT', 'Microsoft Corporation', 'Technology', 421.65),
    ('NVDA', 'NVIDIA Corporation', 'Technology', 132.80),
    ('GOOGL', 'Alphabet Inc.', 'Communication', 175.20),
    ('AMZN', 'Amazon.com Inc.', 'Consumer discretionary', 214.10),
    ('TSLA', 'Tesla Inc.', 'Consumer discretionary', 248.50),
    ('JPM', 'JPMorgan Chase & Co.', 'Financials', 231.80),
    ('V', 'Visa Inc.', 'Financials', 312.40),
    ('JNJ', 'Johnson & Johnson', 'Healthcare', 158.70),
    ('XOM', 'Exxon Mobil Corporation', 'Energy', 113.20),
    ('WMT', 'Walmart Inc.', 'Consumer staples', 92.60),
    ('CAT', 'Caterpillar Inc.', 'Industrials', 352.90),
]

def weekdays(start, count):
    days = []
    while len(days) < count:
        if start.weekday() < 5:
            days.append(start.isoformat())
        start += timedelta(days=1)
    return days


def history(symbol):
    row = next(row for row in CATALOG if row[0] == symbol)
    seed = sum(ord(c) for c in symbol)
    values = [1 + .0007*i + .025*math.sin(i*.15+seed) + .012*math.sin(i*.47+seed) for i in range(252)]
    return [{'date': day, 'close': round(row[3]*value/values[-1], 2)}
            for day, value in zip(weekdays(date(2024, 10, 1), 252), values)]


def stocks():
    result = []
    for symbol, name, sector, price in CATALOG:
        points = history(symbol)
        change = (points[-1]['close']/points[-2]['close']-1)*100
        result.append(dict(symbol=symbol, name=name, sector=sector, price=price, change_pct=round(change, 2)))
    return result


def forecast(symbol, horizon, past=None, mode='demo'):
    past = history(symbol) if past is None else past
    closes = [point['close'] for point in past if point.get('close') and point['close'] > 0]
    last = closes[-1]
    returns = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
    recent = returns[-60:] or returns
    short = returns[-20:] or recent
    long = returns[-126:] or recent
    weights = [i + 1 for i in range(len(short))]
    short_mean = sum(value * weight for value, weight in zip(short, weights)) / sum(weights)
    recent_mean = sum(recent) / len(recent)
    long_mean = sum(long) / len(long)
    blended = .55 * short_mean + .30 * recent_mean + .15 * long_mean
    variance = sum((value - recent_mean) ** 2 for value in recent) / max(1, len(recent) - 1)
    volatility = math.sqrt(variance)
    annualized_volatility = volatility * math.sqrt(252)
    trend_cap = max(.0015, min(.012, annualized_volatility / 252))
    drift = max(min(blended * .55, trend_cap), -trend_cap)
    days = weekdays(date.fromisoformat(past[-1]['date'])+timedelta(days=1), horizon)
    points = []
    for i, day in enumerate(days, 1):
        mean_reversion = .985 ** i
        price = last * math.exp(drift * i * mean_reversion)
        width = price * max(.015, volatility * math.sqrt(i) * 1.35)
        points.append(dict(date=day, price=round(price, 2), lower=round(max(.01, price-width), 2), upper=round(price+width, 2)))
    return dict(mode=mode, symbol=symbol, horizon=horizon, method='Weighted momentum and volatility estimate',
                last_price=last, projected_price=points[-1]['price'], points=points,
                note='Uses weighted recent returns, longer trend context, and volatility. This is not a trained forecast or a calibrated confidence interval.')
