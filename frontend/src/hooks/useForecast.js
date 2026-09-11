import { useEffect, useRef, useState } from 'react';
import { request } from '../api';

export function useForecast(symbol, horizon) {
  const [forecast, setForecast] = useState(null);
  const [forecastStatus, setForecastStatus] = useState('idle');
  const [forecastError, setForecastError] = useState('');
  const requestRef = useRef(null);

  function resetForecast() {
    requestRef.current?.abort();
    requestRef.current = null;
    setForecast(null);
    setForecastStatus('idle');
    setForecastError('');
  }

  async function generateForecast() {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setForecastStatus('loading');
    setForecastError('');
    setForecast(null);

    try {
      const result = await request('/forecasts', {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, horizon }),
      });
      if (requestRef.current === controller) {
        setForecast(result);
        setForecastStatus('ready');
      }
    } catch (error) {
      if (error.name !== 'AbortError' && requestRef.current === controller) {
        setForecastError(error.message);
        setForecastStatus('error');
      }
    }
  }

  useEffect(() => () => requestRef.current?.abort(), []);

  return { forecast, forecastStatus, forecastError, resetForecast, generateForecast };
}
