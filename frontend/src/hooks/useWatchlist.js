import { useEffect, useState } from 'react';
import { loadWatchlist } from '../api';

export function useWatchlist() {
  const [watchlist, setWatchlist] = useState(loadWatchlist);
  const [storageError, setStorageError] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem('sppm.watchlist', JSON.stringify(watchlist));
      setStorageError(false);
    } catch {
      setStorageError(true);
    }
  }, [watchlist]);

  function toggle(symbol) {
    setWatchlist(current => current.includes(symbol) ? current.filter(item => item !== symbol) : [...current, symbol]);
  }

  return { watchlist, storageError, toggleWatchlist: toggle };
}
