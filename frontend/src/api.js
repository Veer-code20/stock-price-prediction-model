export async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, options);
  if (response.status === 503) throw new Error('Market data is temporarily unavailable. Try again shortly.');
  if (!response.ok) throw new Error(response.status === 404 ? 'This stock is not available in the catalog.' : 'Could not load data. Check that the backend is running and try again.');
  return response.json();
}
export function loadWatchlist() {
  try {
    const value = JSON.parse(localStorage.getItem('sppm.watchlist') ?? '["AAPL","NVDA","MSFT"]');
    return Array.isArray(value) ? value.filter(item => typeof item === 'string') : [];
  } catch { return ['AAPL', 'NVDA', 'MSFT']; }
}
