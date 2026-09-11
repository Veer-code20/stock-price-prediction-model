import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';
import { loadWatchlist } from './api';
vi.mock('recharts', () => ({ ResponsiveContainer: ({children}) => <div>{children}</div>, AreaChart: () => <div>Price chart</div>, Area: () => null, CartesianGrid: () => null, Tooltip: () => null, XAxis: () => null, YAxis: () => null }));
const stocks = [{symbol:'AAPL',name:'Apple Inc.',sector:'Technology',price:200,change_pct:1},{symbol:'MSFT',name:'Microsoft Corporation',sector:'Technology',price:400,change_pct:-1}];
const detail = symbol => ({stock:stocks.find(s => s.symbol === symbol), history:[{date:'2025-01-01T14:30:00+00:00',close:190},{date:'2025-01-02T14:30:00+00:00',close:200}],stats:{open:190,high:205,low:189,volume:1000000,fifty_two_week_high:250,fifty_two_week_low:150},mode:'demo',quote_at:'2025-01-02T20:00:00+00:00'});
const response = value => Promise.resolve({ok:true,json:async () => value});
const openEstimator = () => fireEvent.click(screen.getAllByRole('button', {name: /Price Estimator/})[0]);
beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('fetch', vi.fn((url, options) => {
    if (url === '/api/stocks') return response({items:stocks,mode:'demo'});
    if (url.startsWith('/api/stocks/')) return response(detail(url.split('/')[3].split('?')[0]));
    const body = JSON.parse(options.body);
    return response({symbol:body.symbol,horizon:body.horizon,projected_price:210,points:[{lower:195,upper:225}]});
  }));
});
afterEach(() => {cleanup(); vi.useRealTimers(); vi.unstubAllGlobals();});
describe('Market workspace acceptance', () => {
  it('searches by company name and explains empty results', async () => {
    render(<App/>);
    await screen.findByRole('button',{name:/AAPL Apple Inc/});
    fireEvent.change(screen.getByLabelText('Search stocks'),{target:{value:'microsoft'}});
    expect(screen.getAllByRole('button',{name:/MSFT.*Microsoft/}).length).toBeGreaterThan(0);
    expect(screen.queryByRole('button',{name:/AAPL Apple Inc/})).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Search stocks'),{target:{value:'zzzz'}});
    expect(screen.getByText('No stocks found')).toBeInTheDocument();
  });
  it('persists watchlist changes and clears a forecast when the horizon changes', async () => {
    render(<App/>);
    fireEvent.click(await screen.findByRole('button',{name:'Remove from watchlist'}));
    expect(JSON.parse(localStorage.getItem('sppm.watchlist'))).not.toContain('AAPL');
    openEstimator();
    fireEvent.click(screen.getByRole('button',{name:'Estimate'}));
    expect(await screen.findByText('AAPL · 1 month')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Timeframe'),{target:{value:'5'}});
    expect(screen.queryByText('AAPL · 1 month')).not.toBeInTheDocument();
  });
  it('labels the watchlist as saved stocks instead of the full S&P 500 catalog', async () => {
    render(<App/>);
    await screen.findByRole('button',{name:/AAPL Apple Inc/});
    fireEvent.click(screen.getByRole('button',{name:/Watchlist/}));
    expect(screen.getByText('2 saved stocks')).toBeInTheDocument();
    expect(screen.queryByText(/of 2 S&P 500 stocks/)).not.toBeInTheDocument();
  });
  it('supports expanded forecast timeframes and swaps the right rail into estimator mode', async () => {
    render(<App/>);
    await screen.findByRole('button',{name:/AAPL Apple Inc/});
    expect(screen.queryByLabelText('Timeframe')).not.toBeInTheDocument();
    openEstimator();
    expect(screen.queryByLabelText('Search stocks')).not.toBeInTheDocument();
    expect(screen.getByRole('option', {name: '1 day'})).toBeInTheDocument();
    expect(screen.getByRole('option', {name: '6 months'})).toBeInTheDocument();
    expect(screen.getByRole('option', {name: '1 year'})).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('button', {name: /Price Estimator/})[1]);
    expect(screen.queryByLabelText('Timeframe')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Search stocks')).toBeInTheDocument();
  });
  it('discards a superseded forecast after switching stocks', async () => {
    let resolveForecast;
    const normalFetch = globalThis.fetch;
    vi.stubGlobal('fetch', vi.fn((url, options) => url === '/api/forecasts' ? new Promise(resolve => {resolveForecast=resolve;}) : normalFetch(url, options)));
    render(<App/>);
    await screen.findByRole('button',{name:'Remove from watchlist'});
    openEstimator();
    fireEvent.click(screen.getByRole('button',{name:'Estimate'}));
    fireEvent.click(screen.getByRole('button',{name:/MSFT.*Microsoft/}));
    await screen.findByRole('heading',{name:'Microsoft Corporation'});
    resolveForecast({ok:true,json:async () => ({symbol:'AAPL',horizon:20,projected_price:210,points:[{lower:195,upper:225}]})});
    await waitFor(() => expect(screen.getByRole('button',{name:'Estimate'})).toBeEnabled());
    expect(screen.queryByText('AAPL · 1 month')).not.toBeInTheDocument();
  });
  it('shows failures and allows retry', async () => {
    globalThis.fetch.mockImplementationOnce(() => Promise.reject(new Error('offline')));
    render(<App/>);
    expect(await screen.findByText('Could not load the stock catalog.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'Try again'}));
    expect(await screen.findByRole('button',{name:/AAPL Apple Inc/})).toBeInTheDocument();
  });
  it('refreshes quotes without clearing the selected stock and marks failed refreshes', async () => {
    vi.useFakeTimers();
    render(<App/>);
    await act(async () => {});
    expect(screen.getByRole('heading', {name: 'Apple Inc.'})).toBeInTheDocument();
    const normalFetch = globalThis.fetch;
    vi.stubGlobal('fetch', vi.fn((url, options) => url.startsWith('/api/stocks/') ? Promise.reject(new Error('offline')) : normalFetch(url, options)));
    await act(async () => { await vi.advanceTimersByTimeAsync(300000); });
    expect(screen.getByText('Refresh failed. Showing the last received prices.')).toBeInTheDocument();
    expect(screen.getByRole('heading', {name: 'Apple Inc.'})).toBeInTheDocument();
    vi.stubGlobal('fetch', normalFetch);
    await act(async () => { await vi.advanceTimersByTimeAsync(300000); });
    expect(screen.queryByText('Refresh failed. Showing the last received prices.')).not.toBeInTheDocument();
  });
  it('recovers from an initial detail failure on the next refresh', async () => {
    vi.useFakeTimers();
    const normalFetch = globalThis.fetch;
    vi.stubGlobal('fetch', vi.fn((url, options) => url.startsWith('/api/stocks/') ? Promise.reject(new Error('offline')) : normalFetch(url, options)));
    render(<App/>);
    await act(async () => {});
    expect(screen.getByText('offline')).toBeInTheDocument();
    vi.stubGlobal('fetch', normalFetch);
    await act(async () => { await vi.advanceTimersByTimeAsync(300000); });
    expect(screen.getByRole('heading', {name: 'Apple Inc.'})).toBeInTheDocument();
    expect(screen.queryByText('offline')).not.toBeInTheDocument();
  });
  it('searches the full S&P 500 universe and only requests quotes for the current page', async () => {
    const members = [...stocks, ...Array.from({length: 28}, (_, i) => ({symbol: `S${i}`, name: `Company ${i}`, sector: 'Technology', price: null, change_pct: null, quote_status: 'pending'}))];
    const normalFetch = globalThis.fetch;
    vi.stubGlobal('fetch', vi.fn((url, options) => {
      if (url === '/api/stocks') return response({items: members, mode: 'market'});
      if (url.startsWith('/api/quotes?')) return response({items: new URLSearchParams(url.split('?')[1]).get('symbols').split(',').map(symbol => ({symbol, price: 123, change_pct: 1, quote_status: 'ready'}))});
      return normalFetch(url, options);
    }));
    render(<App/>);
    await screen.findByRole('button', {name: /S0 Company 0/});
    expect(screen.queryByRole('button', {name: /S27 Company 27/})).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name: 'Next'}));
    expect(await screen.findByRole('button', {name: /S10 Company 10/})).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Search stocks'), {target: {value: 'Company 27'}});
    expect(await screen.findByRole('button', {name: /S27 Company 27/})).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Search stocks'), {target: {value: 'Technology'}});
    expect(await screen.findByRole('button', {name: /AAPL Apple Inc/})).toBeInTheDocument();
    await waitFor(() => expect(globalThis.fetch.mock.calls.some(([url]) => url.startsWith('/api/quotes?') && decodeURIComponent(url).includes('S27'))).toBe(true));
    const quoteCalls = globalThis.fetch.mock.calls.filter(([url]) => url.startsWith('/api/quotes?'));
    expect(quoteCalls.every(([url]) => new URLSearchParams(url.split('?')[1]).get('symbols').split(',').length <= 20)).toBe(true);
  });

  it('shows timing, default 1D range, expanded stats, and estimator warning', async () => {
    render(<App/>);
    await screen.findByRole('button',{name:/AAPL Apple Inc/});
    expect(screen.getByText('Market session · 9:30 AM–4:00 PM ET')).toBeInTheDocument();
    expect(screen.queryByText('One day range')).not.toBeInTheDocument();
    expect(screen.getByRole('button', {name: '1D'})).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText("Today's High")).toBeInTheDocument();
    expect(screen.getByText("Today's Low")).toBeInTheDocument();
    expect(screen.getByText('52-Week High')).toBeInTheDocument();
    expect(screen.getByText('52-Week Low')).toBeInTheDocument();
    expect(screen.getByText(/1D sample change/)).toBeInTheDocument();
    expect(screen.getByRole('button', {name: 'YTD'})).toBeInTheDocument();
    expect(screen.queryByText(/Yahoo Finance · May be delayed/)).not.toBeInTheDocument();
    expect(screen.getByRole('heading', {level: 1, name: 'Discover'})).toBeInTheDocument();
    openEstimator();
    expect(screen.getByText(/not financial advice/i)).toBeInTheDocument();
    expect(screen.queryByText(/🧪/)).not.toBeInTheDocument();
  });

  it('survives malformed watchlist storage', () => {
    localStorage.setItem('sppm.watchlist','bad-json');
    expect(loadWatchlist()).toContain('AAPL');
    localStorage.setItem('sppm.watchlist','{}');
    expect(loadWatchlist()).toEqual([]);
  });
});
