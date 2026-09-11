import { useEffect, useState } from 'react';
import { request } from '../api';

export function useCatalog() {
  const [catalog, setCatalog] = useState([]);
  const [feed, setFeed] = useState('market');
  const [catalogStatus, setCatalogStatus] = useState('loading');
  const [catalogRetry, setCatalogRetry] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let busy = false;
    setCatalogStatus('loading');

    async function refresh() {
      if (busy) return;
      busy = true;
      try {
        const data = await request('/stocks', { signal: controller.signal });
        if (!controller.signal.aborted) {
          setCatalog(data.items);
          setFeed(data.mode);
          setCatalogStatus('ready');
        }
      } catch (error) {
        if (error.name !== 'AbortError') setCatalogStatus('error');
      } finally {
        busy = false;
      }
    }

    refresh();
    return () => controller.abort();
  }, [catalogRetry]);

  return { catalog, feed, catalogStatus, setCatalogRetry };
}
