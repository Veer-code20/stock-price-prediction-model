import { Activity, Bookmark, Calculator, ChevronRight, Globe2 } from 'lucide-react';

export default function Sidebar({ tab, setTab, watchlistCount }) {
  return <aside className="sidebar">
    <a className="brand" href="https://www.spglobal.com/spdji/en/indices/equity/sp-500/" target="_blank" rel="noreferrer" aria-label="SPPM market workspace" title="S&P 500 in one place">
      <span className="brand-row"><span className="brand-icon"><Activity size={23}/></span>SPPM<span className="brand-dot">.</span></span>
      <small>S&P 500 in one place</small>
    </a>
    <span className="sidebar-label">WORKSPACE</span>
    <nav aria-label="Main navigation">
      <button className={tab === 'explore' ? 'nav-item active' : 'nav-item'} onClick={() => setTab('explore')} aria-label="Discover S&P 500 stocks" title="Discover S&P 500 stocks">
        <Globe2 size={23}/>Discover<ChevronRight size={14}/>
      </button>
      <button className={tab === 'watchlist' ? 'nav-item active' : 'nav-item'} onClick={() => setTab('watchlist')}>
        <Bookmark size={19}/>Watchlist<span className="count">{watchlistCount}</span>
      </button>
      <button className={tab === 'estimator' ? 'nav-item active' : 'nav-item'} onClick={() => setTab('estimator')} title="Price scenarios, not advice.">
        <Calculator size={19}/>Price Estimator<ChevronRight size={14}/>
      </button>
    </nav>
    <div className="profile"><span className="avatar">Y</span><div>Your workspace<small>Saved on this browser</small></div><span className="status-dot"/></div>
  </aside>;
}
