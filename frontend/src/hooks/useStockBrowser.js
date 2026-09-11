import { useEffect, useMemo, useState } from 'react';
import { HIGHLIGHT_SYMBOLS, PAGE_SIZE } from '../constants';

export function useStockBrowser(catalog, tab, watchlist, query, symbol) {
  const [page, setPage] = useState(0);

  const visible = useMemo(() => {
    const search = query.trim().toLowerCase();
    return catalog.filter(stock => {
      const matchesSearch = `${stock.symbol} ${stock.name} ${stock.sector}`.toLowerCase().includes(search);
      const matchesTab = tab !== 'watchlist' || watchlist.includes(stock.symbol);
      return matchesSearch && matchesTab;
    });
  }, [catalog, query, tab, watchlist]);

  const pageCount = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount - 1);
  const pageItems = visible.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE);
  const highlights = HIGHLIGHT_SYMBOLS.map(ticker => catalog.find(item => item.symbol === ticker)).filter(Boolean);
  const quoteSymbols = [...new Set([...pageItems.map(item => item.symbol), ...highlights.map(item => item.symbol), symbol])]
    .filter(ticker => catalog.some(item => item.symbol === ticker))
    .sort()
    .join(',');
  const listSummary = tab === 'watchlist'
    ? `${visible.length} saved ${visible.length === 1 ? 'stock' : 'stocks'}`
    : `${pageItems.length} of ${visible.length}`;

  useEffect(() => { setPage(0); }, [query, tab]);

  return { visible, pageItems, pageCount, currentPage, setPage, listSummary, highlights, quoteSymbols };
}
