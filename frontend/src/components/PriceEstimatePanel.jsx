import { ArrowRight, ChartNoAxesCombined, ChevronDown, ChevronRight, ShieldAlert } from 'lucide-react';
import Failure from './Failure';
import { horizonLabel, money } from '../utils/format';

export default function PriceEstimatePanel({
  open,
  onToggle,
  horizon,
  setHorizon,
  resetForecast,
  generateForecast,
  detail,
  forecastStatus,
  forecastError,
  forecast,
  variant = 'rail',
}) {
  const last = forecast?.points.at(-1);
  const inline = variant === 'inline';
  const trained = forecast?.validation;
  const warning = 'For learning and comparison only. This estimate is not financial advice, not a recommendation to buy or sell, and should not be used to make real investing decisions.';

  return <section className={`panel forecast-panel ${variant === 'inline' ? 'inline-estimate' : ''} ${open ? '' : 'collapsed'}`}>
    <button className="collapse-title forecast-title" aria-expanded={open} onClick={onToggle}>
      <span className="forecast-icon"><ChartNoAxesCombined size={21}/></span>
      <div><h2>Price Estimator</h2></div>
      <span className="outline-pill">EXPERIMENTAL</span>
      {open ? <ChevronDown size={16}/> : <ChevronRight size={16}/>}
    </button>
    {open && <div className="collapse-body">
      <div className="forecast-controls">
        <div>
          <label htmlFor="horizon">Timeframe</label>
          <select id="horizon" value={horizon} onChange={event => { resetForecast(); setHorizon(Number(event.target.value)); }}>
            <option value={1}>1 day</option>
            <option value={5}>1 week</option>
            <option value={20}>1 month</option>
            <option value={60}>3 months</option>
            <option value={126}>6 months</option>
            <option value={252}>1 year</option>
          </select>
        </div>
        <button className="primary" onClick={generateForecast} disabled={!detail || forecastStatus === 'loading'}>
          {forecastStatus === 'loading' ? 'Computing…' : detail?.mode === 'market' ? 'Predict' : 'Estimate'}<ArrowRight size={17}/>
        </button>
      </div>
      <div aria-live="polite">
        {forecastError && <Failure text={forecastError} retry={generateForecast}/>} 
        {forecast && <div className="forecast-result">
          <div><span>{forecast.symbol} · {horizonLabel(forecast.horizon)}</span><strong>{money(forecast.projected_price)}</strong><small>{trained ? 'Model predicted price' : 'Demo estimated price'}</small></div>
          <div><span>Lower estimate</span><strong className="bounds">{money(last.lower)}</strong><small>If the price lands lower</small></div>
          <div><span>Higher estimate</span><strong className="bounds">{money(last.upper)}</strong><small>If the price lands higher</small></div>
        </div>}
        {forecast?.mode === 'demo' && <div className="forecast-note"><p>Synthetic demo estimate. No trained-model performance is measured in demo mode.</p></div>}
      </div>
      <div className="forecast-note warning-note"><ShieldAlert size={15}/><p>{warning}</p></div>
    </div>}
  </section>;
}
