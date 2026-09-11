export const money = value => value == null ? '—' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);

export const intradayLabel = value => new Date(value).toLocaleTimeString('en-US', {
  hour: 'numeric',
  minute: '2-digit',
  timeZone: 'America/New_York',
});

export const dateLabel = value => new Date(value.includes('T') ? value : `${value}T12:00:00`).toLocaleDateString('en-US', {
  month: 'short',
  day: 'numeric',
});

export const horizonLabel = value => ({
  1: '1 day',
  5: '1 week',
  20: '1 month',
  60: '3 months',
  126: '6 months',
  252: '1 year',
})[value];

export const compactNumber = value => value == null ? '—' : new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(value);

export const rangeLabel = range => ({
  '1D': 'One day range',
  '1W': 'One week range',
  '1M': 'One month range',
  '3M': 'Three month range',
  '1Y': 'One year range',
  'YTD': 'Year to date range',
})[range] || `${range} range`;

export const marketSessionLabel = () => 'Market session · 9:30 AM–4:00 PM ET';
