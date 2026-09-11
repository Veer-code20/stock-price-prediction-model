import { ChevronDown, ChevronRight, Search, X } from 'lucide-react';
import Change from './Change';
import Failure from './Failure';
import { money } from '../utils/format';

export default function StockListPanel({
  tab,
  open,
  onToggle,
  listSummary,
  query,
  setQuery,
  catalogStatus,
  visible,
  pageItems,
  symbol,
  selectStock,
  currentPage,
  pageCount,
  setPage,
  setCatalogRetry,
  catalogLength,
  feed,
}) {
  const title = tab === 'explore' ? 'Discover' : 'Your watchlist';
  const changePage = (event, nextPage) => {
    const list = event.currentTarget.closest('.explorer')?.querySelector('.stock-list');
    if (list?.scrollTo) list.scrollTo({ top: 0 });
    else if (list) list.scrollTop = 0;
    setPage(nextPage);
  };

  return <section className={`panel explorer ${open ? '' : 'collapsed'}`}>
    <button className="collapse-title panel-title" aria-expanded={open} onClick={onToggle}>
      <h2>{title}</h2>
      <span className="panel-title-meta"><span>{listSummary}</span>{open ? <ChevronDown size={16}/> : <ChevronRight size={16}/>}</span>
    </button>
    {open && <div className="collapse-body">
      <div className="search">
        <Search size={17}/>
        <input aria-label="Search stocks" placeholder="Search S&P 500 stocks" value={query} onChange={event => setQuery(event.target.value)}/>
        {query && <button aria-label="Clear search" onClick={() => setQuery('')}><X size={14}/></button>}
      </div>
      <div className="list-label"><span>COMPANY</span><span>PRICE / CHANGE</span></div>
      {catalogStatus === 'error'
        ? <Failure text="Could not load the stock catalog." retry={() => setCatalogRetry(n => n + 1)}/>
        : catalogStatus === 'loading'
          ? <p className="empty" role="status">Loading companies…</p>
          : visible.length === 0
            ? <EmptyState query={query}/>
            : <div className="stock-list">{pageItems.map(item => <StockRow key={item.symbol} item={item} selected={symbol === item.symbol} selectStock={selectStock}/>)}</div>}
      {catalogStatus === 'ready' && pageItems.some(item => item.quote_status === 'unavailable') && <p className="catalog-warning" role="status">Some quotes are unavailable. Retrying in the background.</p>}
      {visible.length > 12 && <nav className="pagination" aria-label="Stock pages">
        <button disabled={currentPage === 0} onClick={event => changePage(event, currentPage - 1)}>Previous</button>
        <span>{currentPage + 1} / {pageCount}</span>
        <button disabled={currentPage + 1 === pageCount} onClick={event => changePage(event, currentPage + 1)}>Next</button>
      </nav>}
      <div className="catalog-note">{catalogLength} {feed === 'demo' ? 'sample stocks' : 'S&P 500 stocks'} · {feed === 'demo' ? 'Sample prices' : 'Yahoo Finance'}</div>
    </div>}
  </section>;
}

function StockRow({ item, selected, selectStock }) {
  return <button className={`stock-row ${selected ? 'selected' : ''}`} aria-pressed={selected} onClick={() => selectStock(item.symbol)}>
    <span className={`company-icon icon-${item.symbol.charCodeAt(0) % 4}`}>{item.symbol.slice(0, 1)}</span>
    <span className="company-text"><strong>{item.symbol}</strong><small>{item.name}</small></span>
    <span className="row-price"><strong>{money(item.price)}</strong><Change value={item.change_pct} status={item.quote_status}/></span>
  </button>;
}

function EmptyState({ query }) {
  return <div className="empty">
    <Search size={24}/>
    <h3>{query ? 'No stocks found' : 'Your list starts here'}</h3>
    <p>{query ? 'Try another company or ticker.' : 'Open Explore markets and save a stock to follow it here.'}</p>
  </div>;
}
