import { ArrowDownRight, ArrowUpRight } from 'lucide-react';
import { money } from '../utils/format';

export default function Change({ value, amount, status }) {
  if (value == null) {
    return <span className="unavailable">{status === 'pending' ? 'Loading…' : 'Unavailable'}</span>;
  }

  return <span className={value >= 0 ? 'positive change' : 'negative change'}>
    {value >= 0 ? <ArrowUpRight size={14}/> : <ArrowDownRight size={14}/>} {Math.abs(value).toFixed(2)}%{amount != null && ` (${amount >= 0 ? '+' : '−'}${money(Math.abs(amount))})`}
  </span>;
}
