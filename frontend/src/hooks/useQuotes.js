import { useEffect, useState } from 'react';
import { request } from '../api';
import { REFRESH_MS } from '../constants';

export function useQuotes(feed, quoteSymbols) {
  const [quotes, setQuotes] = useState({});

  useEffect(() => {
    if (feed === 'demo' || !quoteSymbols) return;
    const controller = new AbortController();
    let busy = false;

    async function refresh() {
      if (busy) return;
      busy = true;
      try {
        const result = await request(`/quotes?symbols=${encodeURIComponent(quoteSymbols)}`, { signal: controller.signal });
        if (!controller.signal.aborted) {
          setQuotes(current => ({ ...current, ...Object.fromEntries(result.items.map(item => [item.symbol, item])) }));
        }
      } catch (error) {
        if (error.name !== 'AbortError' && !controller.signal.aborted) {
          const unavailable = quoteSymbols.split(',').map(symbol => [symbol, { price: null, change_pct: null, quote_status: 'unavailable' }]);
          setQuotes(current => ({ ...current, ...Object.fromEntries(unavailable) }));
        }
      } finally {
        busy = false;
      }
    }

    refresh();
    const timer = setInterval(() => { if (!document.hidden) refresh(); }, REFRESH_MS);
    return () => { controller.abort(); clearInterval(timer); };
  }, [feed, quoteSymbols]);

  return quotes;
}
