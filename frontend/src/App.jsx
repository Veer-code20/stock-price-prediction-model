import { useEffect, useRef, useState } from 'react';
import { Activity, ArrowDownRight, ArrowUpRight, ArrowRight, Bookmark, ChartNoAxesCombined, ChevronRight, FlaskConical, Globe2, Search, Star, X } from 'lucide-react';
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { loadWatchlist, request } from './api';

const money = value => value == null ? '—' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
const intradayLabel = value => new Date(value).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/New_York' });
const dateLabel = value => new Date(value.includes('T') ? value : `${value}T12:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
const horizonLabel = value => ({5: 'about 1 week', 20: 'about 1 month', 60: 'about 3 months'}[value]);
function Change({ value, status }) { if (value == null) return <span className="unavailable">{status === 'pending' ? 'Loading…' : 'Unavailable'}</span>; return <span className={value >= 0 ? 'positive change' : 'negative change'}>{value >= 0 ? <ArrowUpRight size={14}/> : <ArrowDownRight size={14}/>} {Math.abs(value).toFixed(2)}%</span>; }
function Failure({ text, retry }) { return <div className="failure" role="alert"><p>{text}</p><button className="secondary" onClick={retry}>Try again</button></div>; }

export default function App() {
  const [page, setPage] = useState(0);
  const [quotes, setQuotes] = useState({});
  const [feed, setFeed] = useState('market');
  const [refreshError, setRefreshError] = useState('');
  const [catalog, setCatalog] = useState([]);
  const [catalogStatus, setCatalogStatus] = useState('loading');
  const [catalogRetry, setCatalogRetry] = useState(0);
  const [query, setQuery] = useState('');
  const [tab, setTab] = useState('explore');
  const [symbol, setSymbol] = useState('AAPL');
  const [range, setRange] = useState('3M');
  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState('');
  const [retry, setRetry] = useState(0);
  const [watchlist, setWatchlist] = useState(loadWatchlist);
  const [storageError, setStorageError] = useState(false);
  const [horizon, setHorizon] = useState(20);
  const [forecast, setForecast] = useState(null);
  const [forecastStatus, setForecastStatus] = useState('idle');
  const [forecastError, setForecastError] = useState('');
  const forecastRequest = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    let busy = false;
    setCatalogStatus('loading');
    async function refresh() {
      if (busy) return;
      busy = true;
      try {
        const data = await request('/stocks', { signal: controller.signal });
        if (!controller.signal.aborted) { setCatalog(data.items); setFeed(data.mode); setCatalogStatus('ready'); }
      } catch (error) { if (error.name !== 'AbortError') setCatalogStatus('error'); }
      finally { busy = false; }
    }
    refresh();
    return () => controller.abort();
  }, [catalogRetry]);
  useEffect(() => {
    const controller = new AbortController();
    let busy = false;
    setDetail(null); setDetailError(''); setRefreshError('');
    async function refresh(initial = false) {
      if (busy) return;
      busy = true;
      try {
        const data = await request(`/stocks/${symbol}?range=${range}`, { signal: controller.signal });
        if (!controller.signal.aborted) { setDetail(data); setDetailError(''); setRefreshError(''); }
      } catch (error) {
        if (error.name !== 'AbortError') initial ? setDetailError(error.message) : setRefreshError('Refresh failed. Showing the last received prices.');
      } finally { busy = false; }
    }
    refresh(true);
    const timer = setInterval(() => { if (!document.hidden) refresh(); }, 60000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [symbol, range, retry]);
  useEffect(() => {
    try { localStorage.setItem('sppm.watchlist', JSON.stringify(watchlist)); setStorageError(false); }
    catch { setStorageError(true); }
  }, [watchlist]);
  useEffect(() => () => forecastRequest.current?.abort(), []);

  function resetForecast() { forecastRequest.current?.abort(); forecastRequest.current = null; setForecast(null); setForecastStatus('idle'); setForecastError(''); }
  function selectStock(next) { if (next !== symbol) { resetForecast(); setDetail(null); setSymbol(next); } }
  function toggleWatchlist() { setWatchlist(current => current.includes(symbol) ? current.filter(item => item !== symbol) : [...current, symbol]); }
  async function generateForecast() {
    forecastRequest.current?.abort();
    const controller = new AbortController(); forecastRequest.current = controller;
    setForecastStatus('loading'); setForecastError(''); setForecast(null);
    try {
      const result = await request('/forecasts', { method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol, horizon }) });
      if (forecastRequest.current === controller) { setForecast(result); setForecastStatus('ready'); }
    } catch (error) { if (error.name !== 'AbortError' && forecastRequest.current === controller) { setForecastError(error.message); setForecastStatus('error'); } }
  }
  const enriched = catalog.map(item => ({ ...item, ...quotes[item.symbol] }));
  const visible = enriched.filter(stock => `${stock.symbol} ${stock.name} ${stock.sector}`.toLowerCase().includes(query.trim().toLowerCase()) && (tab !== 'watchlist' || watchlist.includes(stock.symbol)));
  const pageCount = Math.max(1, Math.ceil(visible.length / 12));
  const currentPage = Math.min(page, pageCount - 1);
  const pageItems = visible.slice(currentPage * 12, (currentPage + 1) * 12);
  const highlights = ['AAPL', 'MSFT', 'NVDA', 'GOOGL'].map(ticker => enriched.find(item => item.symbol === ticker)).filter(Boolean);
  const quoteSymbols = [...new Set([...pageItems.map(item => item.symbol), ...highlights.map(item => item.symbol), symbol])].filter(ticker => catalog.some(item => item.symbol === ticker)).sort().join(',');
  useEffect(() => { setPage(0); }, [query, tab]);
  useEffect(() => {
    if (feed === 'demo' || !quoteSymbols) return;
    const controller = new AbortController();
    let busy = false;
    async function refresh() {
      if (busy) return;
      busy = true;
      try {
        const result = await request(`/quotes?symbols=${encodeURIComponent(quoteSymbols)}`, { signal: controller.signal });
        if (!controller.signal.aborted) setQuotes(current => ({ ...current, ...Object.fromEntries(result.items.map(item => [item.symbol, item])) }));
      } catch (error) {
        if (error.name !== 'AbortError' && !controller.signal.aborted) setQuotes(current => ({ ...current, ...Object.fromEntries(quoteSymbols.split(',').map(ticker => [ticker, {price: null, change_pct: null, quote_status: 'unavailable'}])) }));
      } finally { busy = false; }
    }
    refresh();
    const timer = setInterval(() => { if (!document.hidden) refresh(); }, 60000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [quoteSymbols, feed]);
  const stock = detail?.stock;
  const chartColor = detail && detail.history.at(-1).close < detail.history[0].close ? '#ff5c5c' : '#00c805';
  const last = forecast?.points.at(-1);

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#main" aria-label="SPPM market workspace"><span className="brand-icon"><Activity size={23}/></span>SPPM<span className="brand-dot">.</span></a>
      <span className="sidebar-label">WORKSPACE</span>
      <nav aria-label="Main navigation">
        <button className={tab === 'explore' ? 'nav-item active' : 'nav-item'} onClick={() => setTab('explore')}><Globe2 size={19}/>Explore markets<ChevronRight size={14}/></button>
        <button className={tab === 'watchlist' ? 'nav-item active' : 'nav-item'} onClick={() => setTab('watchlist')}><Bookmark size={19}/>Watchlist<span className="count">{watchlist.length}</span></button>
      </nav>
      <div className="sidebar-note"><span className="status-dot"/><strong>Market workspace</strong><p>Prices, watchlists, and research.<br/>All in one place.</p><span>S&P 500 / USD</span></div>
      <div className="profile"><span className="avatar">Y</span><div>Your workspace<small>Saved on this browser</small></div><span className="status-dot"/></div>
    </aside>
    <main id="main">
      <header className="topbar"><div>Workspace <ChevronRight size={13}/> <span>{tab === 'explore' ? 'Explore markets' : 'Watchlist'}</span></div><span className="demo-pill"><span className="status-dot"/>{feed === 'demo' ? 'Demo data' : 'Auto-refresh · 60s'}</span></header>
      <div className="content">
        <div className="page-heading"><div><div className="eyebrow">MARKET OVERVIEW</div><h1>{tab === 'explore' ? 'Markets' : 'Watchlist'}</h1><p>The S&P 500, in one place.</p></div><span className="market-tag"><Globe2 size={16}/> S&P 500 <span>·</span> USD</span></div>
        <section className="market-strip" aria-label="Market highlights">{highlights.map(item => <button onClick={() => selectStock(item.symbol)} key={item.symbol}><div><span>{item.symbol}</span><Change value={item.change_pct} status={item.quote_status}/></div><strong>{money(item.price)}</strong><small>{item.name}</small></button>)}{catalogStatus === 'loading' && <p role="status">Loading market…</p>}</section>
        <div className="workspace-grid">
          <section className="panel explorer"><div className="panel-title"><h2>{tab === 'explore' ? 'Discover' : 'Your watchlist'}</h2><span>{visible.length} stocks</span></div>
            <div className="search"><Search size={17}/><input aria-label="Search stocks" placeholder="Search name or symbol" value={query} onChange={event => setQuery(event.target.value)}/>{query && <button aria-label="Clear search" onClick={() => setQuery('')}><X size={14}/></button>}</div>
            <div className="list-label"><span>COMPANY</span><span>PRICE / CHANGE</span></div>
            {catalogStatus === 'error' ? <Failure text="Could not load the stock catalog." retry={() => setCatalogRetry(n => n+1)}/> : catalogStatus === 'loading' ? <p className="empty" role="status">Loading companies…</p> : visible.length === 0 ? <div className="empty"><Search size={24}/><h3>{query ? 'No stocks found' : 'Your list starts here'}</h3><p>{query ? 'Try another company or ticker.' : 'Open Explore markets and save a stock to follow it here.'}</p></div> : <div className="stock-list">{pageItems.map(item => <button key={item.symbol} className={`stock-row ${symbol === item.symbol ? 'selected' : ''}`} aria-pressed={symbol === item.symbol} onClick={() => selectStock(item.symbol)}><span className={`company-icon icon-${item.symbol.charCodeAt(0)%4}`}>{item.symbol.slice(0,1)}</span><span className="company-text"><strong>{item.symbol}</strong><small>{item.name}</small></span><span className="row-price"><strong>{money(item.price)}</strong><Change value={item.change_pct} status={item.quote_status}/></span></button>)}</div>}
            {catalogStatus === 'ready' && pageItems.some(item => item.quote_status === 'unavailable') && <p className="catalog-warning" role="status">Some quotes are unavailable. Retrying every 60 seconds.</p>}{visible.length > 12 && <nav className="pagination" aria-label="Stock pages"><button disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous</button><span>{currentPage + 1} / {pageCount}</span><button disabled={currentPage + 1 === pageCount} onClick={() => setPage(currentPage + 1)}>Next</button></nav>}<div className="catalog-note">{catalog.length} {feed === 'demo' ? 'sample stocks' : 'S&P 500 stocks'} · {feed === 'demo' ? 'Sample prices' : 'Yahoo Finance'}</div>
          </section>
          <div className="detail-column">
            <section className="panel stock-detail" aria-label="Selected stock">
              {detailError ? <Failure text={detailError} retry={() => setRetry(n => n+1)}/> : !detail ? <div className="chart-loading" role="status">Loading {symbol} price history…</div> : <>
                <div className="stock-heading"><div className="stock-identity"><span className="company-icon large">{stock.symbol.slice(0,1)}</span><div><h2>{stock.name}</h2><p>{stock.symbol} <span>·</span> {stock.sector}</p></div></div><button className={`save-button ${watchlist.includes(symbol) ? 'saved' : ''}`} aria-label={watchlist.includes(symbol) ? 'Remove from watchlist' : 'Add to watchlist'} aria-pressed={watchlist.includes(symbol)} onClick={toggleWatchlist}><Star size={17} fill={watchlist.includes(symbol) ? 'currentColor' : 'none'}/><span>{watchlist.includes(symbol) ? 'Saved' : 'Watch'}</span></button></div>
                <div className="price-heading"><div><strong>{money(stock.price)}</strong><Change value={stock.change_pct}/><small>{detail.mode === 'demo' ? 'Sample daily change' : 'Today · regular session'}</small></div><div className="range-control" aria-label="History range">{['1D','1W','1M','3M','1Y'].map(item => <button key={item} aria-pressed={range === item} className={range === item ? 'chosen' : ''} onClick={() => setRange(item)}>{item}</button>)}</div></div>
                <div className="chart" role="img" aria-label={`${stock.symbol} ${detail.mode === 'demo' ? 'synthetic' : 'market'} ${range} closing prices, from ${money(detail.history[0].close)} to ${money(stock.price)}.`}><ResponsiveContainer width="100%" height="100%"><AreaChart data={detail.history} margin={{top:15,right:0,left:0,bottom:0}}><defs><linearGradient id="price-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={chartColor} stopOpacity={.07}/><stop offset="100%" stopColor={chartColor} stopOpacity={0}/></linearGradient></defs><XAxis dataKey="date" tickFormatter={range === '1D' && detail.mode !== 'demo' ? intradayLabel : dateLabel} minTickGap={65} axisLine={false} tickLine={false} tick={{fill:'#777777',fontSize:11}} dy={10}/><YAxis hide domain={['auto','auto']} tickFormatter={value => `$${Math.round(value)}`} axisLine={false} tickLine={false} tick={{fill:'#777777',fontSize:11}}/><Tooltip labelFormatter={value => value.includes('T') ? `${dateLabel(value)} · ${intradayLabel(value)} ET` : dateLabel(value)} formatter={value => [money(value), detail.mode === 'demo' ? 'Sample close' : 'Price']} contentStyle={{background:'#171717',border:'1px solid #333',borderRadius:4,color:'#eee'}}/><Area type="linear" dataKey="close" stroke={chartColor} strokeWidth={2} fill="url(#price-fill)" isAnimationActive={false}/></AreaChart></ResponsiveContainer></div>
                <div className="chart-caption"><span>{detail.mode === 'demo' ? 'Synthetic price history' : 'Yahoo Finance · May be delayed'}</span><span>{detail.quote_at ? `Quote ${new Date(detail.quote_at).toLocaleString()}` : `As of ${dateLabel(detail.history.at(-1).date)}`}</span></div>
                <div aria-live="polite">{refreshError && <p className="refresh-warning" role="status">{refreshError}</p>}</div><div className="stats">{[['Open',money(detail.stats.open)],['Day high',money(detail.stats.high)],['Day low',money(detail.stats.low)],['Volume',detail.stats.volume == null ? '—' : `${(detail.stats.volume/1e6).toFixed(2)}M`]].map(([label,value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
              </>}
            </section>
            <section className="panel forecast-panel"><div className="forecast-title"><span className="forecast-icon"><ChartNoAxesCombined size={21}/></span><div><h2>Price estimate</h2><p>A rough trend-based estimate for the next few trading days.</p></div><span className="outline-pill">EXPERIMENTAL</span></div>
              <div className="forecast-controls"><div><label htmlFor="horizon">Timeframe</label><select id="horizon" value={horizon} onChange={event => {resetForecast(); setHorizon(Number(event.target.value));}}><option value={5}>About 1 week</option><option value={20}>About 1 month</option><option value={60}>About 3 months</option></select></div><button className="primary" onClick={generateForecast} disabled={!detail || forecastStatus === 'loading'}>{forecastStatus === 'loading' ? 'Estimating…' : 'Estimate price'}<ArrowRight size={17}/></button></div>
              <div aria-live="polite">{forecastError && <Failure text={forecastError} retry={generateForecast}/>} {forecast && <div className="forecast-result"><div><span>{forecast.symbol} · {horizonLabel(forecast.horizon)}</span><strong>{money(forecast.projected_price)}</strong><small>Estimated price</small></div><div><span>Possible range</span><strong className="bounds">{money(last.lower)}–{money(last.upper)}</strong><small>Rough trend range</small></div></div>}</div>
              <div className="forecast-note"><FlaskConical size={15}/><p>Uses recent price history to make a rough estimate. A trading day means a weekday market day, so 5 is about a week, 20 about a month, and 60 about three months.</p></div>
            </section>
          </div>
        </div>
        {storageError && <p role="status">Browser storage is unavailable. Your watchlist will last only for this session.</p>}
        <footer><span><Activity size={13}/> SPPM</span><span>{feed === 'demo' ? 'Demo mode · Synthetic prices' : 'Yahoo Finance · Prices may be delayed · USD'}</span></footer>
      </div>
    </main>
  </div>;
}
