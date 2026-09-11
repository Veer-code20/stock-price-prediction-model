import { Star } from 'lucide-react';
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import Change from './Change';
import Failure from './Failure';
import { compactNumber, dateLabel, intradayLabel, money } from '../utils/format';

export default function StockDetail({ detailError, retryDetail, detail, symbol, watchlist, toggleWatchlist, range, setRange, refreshError }) {
  if (detailError) {
    return <section className="panel stock-detail" aria-label="Selected stock"><Failure text={detailError} retry={retryDetail}/></section>;
  }

  if (!detail) {
    return <section className="panel stock-detail" aria-label="Selected stock"><div className="chart-loading" role="status">Loading {symbol} price history…</div></section>;
  }

  const stock = detail.stock;
  const chartColor = detail.history.at(-1).close < detail.history[0].close ? '#ff5c5c' : '#00c805';
  const rangeChange = detail.history.length > 1 ? (detail.history.at(-1).close / detail.history[0].close - 1) * 100 : stock.change_pct;
  const saved = watchlist.includes(symbol);

  return <section className="panel stock-detail" aria-label="Selected stock">
    <div className="stock-heading">
      <div className="stock-identity"><span className="company-icon large">{stock.symbol.slice(0, 1)}</span><div><h2>{stock.name}</h2><p>{stock.symbol} <span>·</span> {stock.sector}</p></div></div>
      <button className={`save-button ${saved ? 'saved' : ''}`} aria-label={saved ? 'Remove from watchlist' : 'Add to watchlist'} aria-pressed={saved} onClick={toggleWatchlist}>
        <Star size={17} fill={saved ? 'currentColor' : 'none'}/><span>{saved ? 'Saved' : 'Watch'}</span>
      </button>
    </div>
    <div className="price-heading">
      <div><strong>{money(stock.price)}</strong><Change value={rangeChange}/><small>{detail.mode === 'demo' ? `${range} sample change` : `${range} range change`}</small></div>
      <div className="range-control" aria-label="History range">{['1D', '1W', '1M', '3M', '1Y', 'YTD'].map(item => <button key={item} aria-pressed={range === item} className={range === item ? 'chosen' : ''} onClick={() => setRange(item)}>{item}</button>)}</div>
    </div>
    <div className="chart" role="img" aria-label={`${stock.symbol} ${detail.mode === 'demo' ? 'synthetic' : 'market'} ${range} closing prices, from ${money(detail.history[0].close)} to ${money(stock.price)}.`}>
      <ResponsiveContainer width="100%" height="100%"><AreaChart data={detail.history} margin={{ top: 15, right: 0, left: 0, bottom: 0 }}>
        <defs><linearGradient id="price-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={chartColor} stopOpacity={0.07}/><stop offset="100%" stopColor={chartColor} stopOpacity={0}/></linearGradient></defs>
        <XAxis dataKey="date" tickFormatter={range === '1D' && detail.mode !== 'demo' ? intradayLabel : dateLabel} minTickGap={65} axisLine={false} tickLine={false} tick={{ fill: '#777777', fontSize: 11 }} dy={10}/>
        <YAxis hide domain={['auto', 'auto']} tickFormatter={value => `$${Math.round(value)}`} axisLine={false} tickLine={false} tick={{ fill: '#777777', fontSize: 11 }}/>
        <Tooltip labelFormatter={value => value.includes('T') ? `${dateLabel(value)} · ${intradayLabel(value)} ET` : dateLabel(value)} formatter={value => [money(value), detail.mode === 'demo' ? 'Sample close' : 'Price']} contentStyle={{ background: '#171717', border: '1px solid #333', borderRadius: 4, color: '#eee' }}/>
        <Area type="linear" dataKey="close" stroke={chartColor} strokeWidth={2} fill="url(#price-fill)" isAnimationActive={false}/>
      </AreaChart></ResponsiveContainer>
    </div>
    <div aria-live="polite">{refreshError && <p className="refresh-warning" role="status">{refreshError}</p>}</div>
    <div className="stats">{[
      ['Open', money(detail.stats.open)],
      ["Today's High", money(detail.stats.high)],
      ["Today's Low", money(detail.stats.low)],
      ['Volume', compactNumber(detail.stats.volume)],
      ['52-Week High', money(detail.stats.fifty_two_week_high)],
      ['52-Week Low', money(detail.stats.fifty_two_week_low)],
    ].map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
  </section>;
}
