import Change from './Change';
import { money } from '../utils/format';

export default function MarketHighlights({ highlights, catalogStatus, selectStock }) {
  return <section className="market-strip" aria-label="Market highlights">
    {highlights.map(item => <button onClick={() => selectStock(item.symbol)} key={item.symbol}>
      <div><span>{item.symbol}</span><Change value={item.change_pct} status={item.quote_status}/></div>
      <strong>{money(item.price)}</strong>
      <small>{item.name}</small>
    </button>)}
    {catalogStatus === 'loading' && <p role="status">Loading market…</p>}
  </section>;
}
