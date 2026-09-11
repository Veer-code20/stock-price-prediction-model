import { useMemo, useState } from 'react';
import { Activity } from 'lucide-react';
import { marketSessionLabel } from './utils/format';
import { useCatalog } from './hooks/useCatalog';
import { useForecast } from './hooks/useForecast';
import { useQuotes } from './hooks/useQuotes';
import { useStockBrowser } from './hooks/useStockBrowser';
import { useStockDetail } from './hooks/useStockDetail';
import { useWatchlist } from './hooks/useWatchlist';
import MarketHighlights from './components/MarketHighlights';
import PriceEstimatePanel from './components/PriceEstimatePanel';
import Sidebar from './components/Sidebar';
import StockDetail from './components/StockDetail';
import StockListPanel from './components/StockListPanel';

const titleFor = tab => ({ explore: 'Discover', watchlist: 'Watchlist', estimator: 'Price Estimator' })[tab];

export default function App() {
  const [query, setQuery] = useState('');
  const [tab, setTab] = useState('explore');
  const [symbol, setSymbol] = useState('AAPL');
  const [range, setRange] = useState('1D');
  const [horizon, setHorizon] = useState(20);
  const [listOpen, setListOpen] = useState(true);

  const { catalog, feed, catalogStatus, setCatalogRetry } = useCatalog();
  const { watchlist, storageError, toggleWatchlist } = useWatchlist();
  const { detail, setDetail, detailError, refreshError, retryDetail } = useStockDetail(symbol, range);
  const { forecast, forecastStatus, forecastError, resetForecast, generateForecast } = useForecast(symbol, horizon);
  const browser = useStockBrowser(catalog, tab, watchlist, query, symbol);
  const quotes = useQuotes(feed, browser.quoteSymbols);
  const visible = useMemo(() => browser.visible.map(stock => ({ ...stock, ...quotes[stock.symbol] })), [browser.visible, quotes]);
  const pageItems = useMemo(() => browser.pageItems.map(stock => ({ ...stock, ...quotes[stock.symbol] })), [browser.pageItems, quotes]);
  const highlights = useMemo(() => browser.highlights.map(stock => ({ ...stock, ...quotes[stock.symbol] })), [browser.highlights, quotes]);
  const { pageCount, currentPage, setPage, listSummary } = browser;

  function selectStock(nextSymbol) {
    if (nextSymbol !== symbol) {
      resetForecast();
      setDetail(null);
      setSymbol(nextSymbol);
    }
  }

  function toggleCurrentStock() {
    toggleWatchlist(symbol);
  }

  return <div className="app-shell">
    <Sidebar tab={tab} setTab={setTab} watchlistCount={watchlist.length}/>
    <main id="main">
      <div className="content">
        <div className="page-heading">
          <div className="heading-copy">
            <div className="overview-row">
              <div className="eyebrow overview-hover" tabIndex={0}>MARKET OVERVIEW<span>Prices, watchlists, and research.</span></div>
              <div className="eyebrow session-inline"><span>{marketSessionLabel()}</span></div>
            </div>
            <h1>{titleFor(tab)}</h1>
            {tab === 'explore' && <p>Discover the S&P 500 in one place.</p>}
          </div>
        </div>

        <MarketHighlights highlights={highlights} catalogStatus={catalogStatus} selectStock={selectStock}/>

        <div className="workspace-grid">
          <aside className="right-rail" aria-label="Research controls">
            {tab === 'estimator'
              ? <PriceEstimatePanel
                  open={true}
                  onToggle={() => setTab('explore')}
                  horizon={horizon}
                  setHorizon={setHorizon}
                  resetForecast={resetForecast}
                  generateForecast={generateForecast}
                  detail={detail}
                  forecastStatus={forecastStatus}
                  forecastError={forecastError}
                  forecast={forecast}
                />
              : <StockListPanel
                  tab={tab}
                  open={listOpen}
                  onToggle={() => setListOpen(open => !open)}
                  listSummary={listSummary}
                  query={query}
                  setQuery={setQuery}
                  catalogStatus={catalogStatus}
                  visible={visible}
                  pageItems={pageItems}
                  symbol={symbol}
                  selectStock={selectStock}
                  currentPage={currentPage}
                  pageCount={pageCount}
                  setPage={setPage}
                  setCatalogRetry={setCatalogRetry}
                  catalogLength={catalog.length}
                  feed={feed}
                />}
          </aside>

          <div className="detail-column">
            <StockDetail
              detailError={detailError}
              retryDetail={retryDetail}
              detail={detail}
              symbol={symbol}
              watchlist={watchlist}
              toggleWatchlist={toggleCurrentStock}
              range={range}
              setRange={setRange}
              refreshError={refreshError}
            />
          </div>
        </div>

        {storageError && <p className="storage-warning" role="status">Browser storage is unavailable. Your watchlist will last only for this session.</p>}
        <footer>
          <a href="https://www.spglobal.com/spdji/en/indices/equity/sp-500/" target="_blank" rel="noreferrer"><Activity size={13}/> SPPM</a>
          <span>{feed === 'demo' ? 'Demo mode · Synthetic prices' : <><a href="https://finance.yahoo.com/" target="_blank" rel="noreferrer">Yahoo Finance</a> · Prices may be delayed · USD</>}</span>
        </footer>
      </div>
    </main>
  </div>;
}
