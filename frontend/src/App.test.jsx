import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';
import { loadWatchlist } from './api';
vi.mock('recharts', () => ({ ResponsiveContainer: ({children}) => <div>{children}</div>, AreaChart: () => <div>Price chart</div>, Area: () => null, CartesianGrid: () => null, Tooltip: () => null, XAxis: () => null, YAxis: () => null }));
const stocks = [{symbol:'AAPL',name:'Apple Inc.',sector:'Technology',price:200,change_pct:1},{symbol:'MSFT',name:'Microsoft Corporation',sector:'Technology',price:400,change_pct:-1}];
const detail = symbol => ({stock:stocks.find(s => s.symbol === symbol), history:[{date:'2025-01-01',close:190},{date:'2025-01-02',close:200}],stats:{open:190,high:205,low:189,volume:1000000},mode:'demo'});
const response = value => Promise.resolve({ok:true,json:async () => value});
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
    expect(screen.getByRole('button',{name:/MSFT Microsoft/})).toBeInTheDocument();
    expect(screen.queryByRole('button',{name:/AAPL Apple Inc/})).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Search stocks'),{target:{value:'zzzz'}});
    expect(screen.getByText('No stocks found')).toBeInTheDocument();
  });
  it('persists watchlist changes and clears a forecast when the horizon changes', async () => {
    render(<App/>);
    fireEvent.click(await screen.findByRole('button',{name:'Remove from watchlist'}));
    expect(JSON.parse(localStorage.getItem('sppm.watchlist'))).not.toContain('AAPL');
    fireEvent.click(screen.getByRole('button',{name:'Explore forecast'}));
    expect(await screen.findByText('AAPL · 20 sessions')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Forecast horizon'),{target:{value:'5'}});
    expect(screen.queryByText('AAPL · 20 sessions')).not.toBeInTheDocument();
  });
  it('discards a superseded forecast after switching stocks', async () => {
    let resolveForecast;
    const normalFetch = globalThis.fetch;
    vi.stubGlobal('fetch', vi.fn((url, options) => url === '/api/forecasts' ? new Promise(resolve => {resolveForecast=resolve;}) : normalFetch(url, options)));
    render(<App/>);
    await screen.findByRole('button',{name:'Remove from watchlist'});
    fireEvent.click(screen.getByRole('button',{name:'Explore forecast'}));
    fireEvent.click(screen.getByRole('button',{name:/MSFT Microsoft/}));
    await screen.findByRole('heading',{name:'Microsoft Corporation'});
    resolveForecast({ok:true,json:async () => ({symbol:'AAPL',horizon:20,projected_price:210,points:[{lower:195,upper:225}]})});
    await waitFor(() => expect(screen.getByRole('button',{name:'Explore forecast'})).toBeEnabled());
    expect(screen.queryByText('AAPL · 20 sessions')).not.toBeInTheDocument();
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
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
    expect(screen.getByText('Refresh failed. Showing the last received prices.')).toBeInTheDocument();
    expect(screen.getByRole('heading', {name: 'Apple Inc.'})).toBeInTheDocument();
    vi.stubGlobal('fetch', normalFetch);
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
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
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
    expect(screen.getByRole('heading', {name: 'Apple Inc.'})).toBeInTheDocument();
    expect(screen.queryByText('offline')).not.toBeInTheDocument();
  });
  it('searches the full universe and only requests quotes for the current page', async () => {
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
    await waitFor(() => expect(globalThis.fetch.mock.calls.some(([url]) => url.startsWith('/api/quotes?') && decodeURIComponent(url).includes('S27'))).toBe(true));
    const quoteCalls = globalThis.fetch.mock.calls.filter(([url]) => url.startsWith('/api/quotes?'));
    expect(quoteCalls.every(([url]) => new URLSearchParams(url.split('?')[1]).get('symbols').split(',').length <= 20)).toBe(true);
  });
  it('survives malformed watchlist storage', () => {
    localStorage.setItem('sppm.watchlist','bad-json');
    expect(loadWatchlist()).toContain('AAPL');
    localStorage.setItem('sppm.watchlist','{}');
    expect(loadWatchlist()).toEqual([]);
  });
});
