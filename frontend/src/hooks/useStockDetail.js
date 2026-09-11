import { useEffect, useState } from 'react';
import { request } from '../api';
import { REFRESH_MS } from '../constants';

export function useStockDetail(symbol, range) {
  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState('');
  const [refreshError, setRefreshError] = useState('');
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let busy = false;
    setDetail(null);
    setDetailError('');
    setRefreshError('');

    async function refresh(initial = false) {
      if (busy) return;
      busy = true;
      try {
        const data = await request(`/stocks/${symbol}?range=${range}`, { signal: controller.signal });
        if (!controller.signal.aborted) {
          setDetail(data);
          setDetailError('');
          setRefreshError('');
        }
      } catch (error) {
        if (error.name !== 'AbortError') {
          if (initial) setDetailError(error.message);
          else setRefreshError('Refresh failed. Showing the last received prices.');
        }
      } finally {
        busy = false;
      }
    }

    refresh(true);
    const timer = setInterval(() => { if (!document.hidden) refresh(); }, REFRESH_MS);
    return () => { controller.abort(); clearInterval(timer); };
  }, [symbol, range, retry]);

  return { detail, setDetail, detailError, refreshError, retryDetail: () => setRetry(count => count + 1) };
}
