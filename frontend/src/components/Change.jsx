import { ArrowDownRight, ArrowUpRight } from 'lucide-react';

export default function Change({ value, status }) {
  if (value == null) {
    return <span className="unavailable">{status === 'pending' ? 'Loading…' : 'Unavailable'}</span>;
  }

  return <span className={value >= 0 ? 'positive change' : 'negative change'}>
    {value >= 0 ? <ArrowUpRight size={14}/> : <ArrowDownRight size={14}/>} {Math.abs(value).toFixed(2)}%
  </span>;
}
