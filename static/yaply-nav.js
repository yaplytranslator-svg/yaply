/**
 * yaply-nav.js v3 — Universal Yaply Navigation
 * Add to every feature page before </body>:
 * <script src="/static/yaply-nav.js"></script>
 *
 * Does everything:
 * 1. Injects Yaply header with logo + trip pill + user menu
 * 2. Applies Yaply color palette on top of existing page styles
 * 3. Auto-fills all forms with active trip context
 * 4. Redirects to /login if not authenticated
 * 5. apiFetch() helper for authenticated API calls
 */

(function() {
'use strict';

// ── PALETTE ──
const Y = {
  blue:   '#4A7FD4', orange: '#F4642A', red:    '#E8474A',
  green:  '#2BAF7E', gold:   '#F5A623', purple: '#6B5CE7',
  ink:    '#1A1A2E', text:   '#2D2D3A', muted:  '#6E6E85',
  bg:     '#FAFAF8', surface:'#FFFFFF', border: 'rgba(0,0,0,0.08)',
};

// Page color themes based on which page we're on
const PAGE_THEMES = {
  '/plan':     { color: Y.orange, gradient: `linear-gradient(160deg,#1A1A2E,#2D1A0E)`, label: '✈️ Before Trip' },
  '/during':   { color: Y.red,    gradient: `linear-gradient(160deg,#1A1A2E,#2D0E0E)`, label: '🆘 During Trip' },
  '/after':    { color: Y.purple, gradient: `linear-gradient(160deg,#1A1A2E,#1A0E2D)`, label: '📖 After Trip' },
  '/tools':    { color: Y.green,  gradient: `linear-gradient(160deg,#1A1A2E,#0E2D1A)`, label: '🛠️ Smart Tools' },
  '/discover': { color: Y.purple, gradient: `linear-gradient(160deg,#1A0E2D,#1A1A2E)`, label: '🔍 Discover' },
  '/translate':{ color: Y.blue,   gradient: `linear-gradient(160deg,#0E1A2D,#1A1A2E)`, label: '🎙️ Translate' },
  '/convo':    { color: Y.green,  gradient: `linear-gradient(160deg,#0E2D1A,#1A1A2E)`, label: '💬 Conversation' },
  '/camera':   { color: Y.purple, gradient: `linear-gradient(160deg,#1A0E2D,#1A1A2E)`, label: '📸 Camera Scan' },
};

const path = window.location.pathname;
const theme = PAGE_THEMES[path] || { color: Y.orange, gradient: `linear-gradient(160deg,#1A1A2E,#2D1A2E)`, label: '✈️ Yaply' };

// ── AUTH ──
window.YaplyAuth = {
  getToken: () => localStorage.getItem('yaply_token'),
  getUser:  () => JSON.parse(localStorage.getItem('yaply_user') || 'null'),
  isLoggedIn: () => !!localStorage.getItem('yaply_token'),
  logout() {
    localStorage.removeItem('yaply_token');
    localStorage.removeItem('yaply_user');
    localStorage.removeItem('yaply_active_trip');
    window.location.href = '/login';
  }
};

// ── API FETCH ──
window.apiFetch = async function(url, options = {}) {
  const token = YaplyAuth.getToken();
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  if (token) headers['Authorization'] = 'Bearer ' + token;
  try {
    const res = await fetch(url, { ...options, headers });
    if (res.status === 401) { YaplyAuth.logout(); return { success: false, error: 'Not authenticated' }; }
    return await res.json();
  } catch(e) {
    console.error('apiFetch error:', e);
    return { success: false, error: e.message };
  }
};

// ── CHECK AUTH ──
const publicPaths = ['/', '/login', '/landing'];
if (!publicPaths.includes(path) && !YaplyAuth.isLoggedIn()) {
  window.location.href = '/login?redirect=' + encodeURIComponent(path);
  return;
}

// ── INJECT STYLES ──
const style = document.createElement('style');
style.textContent = `
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&family=Righteous&display=swap');

  :root {
    --y-blue:   ${Y.blue};
    --y-orange: ${Y.orange};
    --y-red:    ${Y.red};
    --y-green:  ${Y.green};
    --y-gold:   ${Y.gold};
    --y-purple: ${Y.purple};
    --y-page:   ${theme.color};
    --accent:   ${Y.orange} !important;
    --accent2:  #D4522A !important;
    --accent-soft: rgba(244,100,42,0.1) !important;
  }

  /* Top color strip */
  #y-strip {
    position: fixed; top: 0; left: 0; right: 0; height: 4px; z-index: 10001;
    background: linear-gradient(90deg,${Y.blue},${Y.orange},${Y.red},${Y.green},${Y.gold},${Y.purple});
  }

  /* Nav bar */
  #y-nav {
    position: fixed; top: 4px; left: 0; right: 0; z-index: 10000;
    height: 52px;
    background: rgba(26,26,46,0.97);
    backdrop-filter: blur(20px);
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0 18px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-family: 'Nunito', sans-serif;
  }

  #y-nav .y-logo {
    font-family: 'Righteous', sans-serif;
    font-size: 1.3rem;
    text-decoration: none;
    background: linear-gradient(135deg, ${Y.orange}, ${Y.red}, ${Y.purple});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.3px;
  }

  #y-nav .y-page-label {
    font-size: 0.7rem;
    font-weight: 800;
    color: rgba(255,255,255,0.5);
    letter-spacing: 0.5px;
    padding: 4px 12px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    border: 1px solid rgba(255,255,255,0.08);
  }

  #y-nav .y-trip-pill {
    display: flex; align-items: center; gap: 7px;
    padding: 5px 13px; border-radius: 100px;
    background: rgba(244,100,42,0.15);
    border: 1px solid rgba(244,100,42,0.25);
    cursor: pointer; font-size: 0.7rem; font-weight: 800;
    color: ${Y.orange}; max-width: 160px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    transition: all 0.2s;
  }
  #y-nav .y-trip-pill:hover { background: rgba(244,100,42,0.25); }

  #y-nav .y-trip-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: ${Y.green}; flex-shrink: 0;
    box-shadow: 0 0 6px ${Y.green};
    animation: yPulse 2s infinite;
  }

  #y-nav .y-user-btn {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, ${Y.orange}, ${Y.red});
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 800; color: white;
    cursor: pointer; border: none; overflow: hidden;
    flex-shrink: 0; transition: all 0.2s;
  }
  #y-nav .y-user-btn:hover { transform: scale(1.08); }

  /* User dropdown */
  #y-dropdown {
    display: none; position: fixed; top: 60px; right: 12px;
    background: rgba(20,20,36,0.98);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px; padding: 8px;
    min-width: 210px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    z-index: 10002;
    font-family: 'Nunito', sans-serif;
  }
  #y-dropdown.open { display: block; animation: yDropIn 0.2s ease; }
  @keyframes yDropIn { from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:translateY(0)} }

  .y-drop-header {
    padding: 10px 12px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 6px;
  }
  .y-drop-name { font-size: 0.85rem; font-weight: 800; color: white; }
  .y-drop-email { font-size: 0.7rem; color: rgba(255,255,255,0.3); margin-top:2px; }

  .y-drop-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: 11px;
    font-size: 0.8rem; font-weight: 700; color: rgba(255,255,255,0.65);
    text-decoration: none; cursor: pointer;
    transition: background 0.15s; margin-bottom: 2px;
  }
  .y-drop-item:hover { background: rgba(255,255,255,0.07); color: white; }
  .y-drop-item.danger { color: ${Y.red}; }
  .y-drop-item.danger:hover { background: rgba(232,71,74,0.1); }
  .y-drop-sep { height: 1px; background: rgba(255,255,255,0.07); margin: 6px 0; }

  /* Trip context bar */
  #y-tripbar {
    position: fixed; top: 56px; left: 0; right: 0;
    padding: 8px 18px;
    background: linear-gradient(135deg, ${theme.color}, ${theme.color}CC);
    display: flex; align-items: center; justify-content: space-between;
    z-index: 9999;
    font-family: 'Nunito', sans-serif;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    flex-wrap: wrap; gap: 6px;
  }
  #y-tripbar.hidden { display: none; }
  .ytb-dest { font-size: 0.82rem; font-weight: 800; color: white; }
  .ytb-meta { font-size: 0.68rem; color: rgba(255,255,255,0.6); font-weight: 600; }
  .ytb-btn {
    padding: 5px 13px; border-radius: 100px;
    font-size: 0.68rem; font-weight: 800;
    cursor: pointer; transition: all 0.2s;
    font-family: 'Nunito', sans-serif;
    border: 1.5px solid rgba(255,255,255,0.3);
    background: rgba(255,255,255,0.12); color: white;
  }
  .ytb-btn:hover { background: rgba(255,255,255,0.22); }
  .ytb-btn-solid { background: white; color: ${theme.color}; border-color: white; }
  .ytb-btn-solid:hover { background: rgba(255,255,255,0.9); }

  /* Body padding */
  body { padding-top: var(--y-nav-height, 56px) !important; }

  /* Animations */
  @keyframes yPulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.6)} }
  @keyframes yFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
`;
document.head.appendChild(style);

// ── BUILD NAV ──
function buildNav() {
  const user = YaplyAuth.getUser();
  const trip = JSON.parse(localStorage.getItem('yaply_active_trip') || 'null');

  // Strip
  const strip = document.createElement('div');
  strip.id = 'y-strip';
  document.body.insertAdjacentElement('afterbegin', strip);

  // Nav
  const nav = document.createElement('div');
  nav.id = 'y-nav';

  const initial = user?.name?.[0]?.toUpperCase() || 'Y';
  const avatar = user?.avatar ? `<img src="${user.avatar}" style="width:100%;height:100%;object-fit:cover;border-radius:50%;"/>` : initial;

  nav.innerHTML = `
    <a href="/app" class="y-logo">Yaply</a>
    <div class="y-page-label">${theme.label}</div>
    <div style="display:flex;align-items:center;gap:8px;">
      ${trip ? `
        <div class="y-trip-pill" onclick="window.location.href='/app'">
          <div class="y-trip-dot"></div>
          <span>✈️ ${trip.destination}</span>
        </div>
      ` : `
        <a href="/app" style="font-size:0.7rem;font-weight:800;color:rgba(255,255,255,0.4);text-decoration:none;padding:5px 12px;background:rgba(255,255,255,0.06);border-radius:100px;">+ New Trip</a>
      `}
      <button class="y-user-btn" onclick="toggleDropdown()" id="yUserBtn">${avatar}</button>
    </div>

    <div id="y-dropdown">
      <div class="y-drop-header">
        <div class="y-drop-name">${user?.name || 'Traveller'}</div>
        <div class="y-drop-email">${user?.email || ''}</div>
      </div>
      <a href="/app" class="y-drop-item">🏠 Home</a>
      <a href="/plan" class="y-drop-item">✈️ Plan Trip</a>
      <a href="/during" class="y-drop-item">🆘 During Trip</a>
      <a href="/after" class="y-drop-item">📖 After Trip</a>
      <a href="/tools" class="y-drop-item">🛠️ Tools</a>
      <a href="/discover" class="y-drop-item">🔍 Discover</a>
      <a href="/translate" class="y-drop-item">🎙️ Translate</a>
      <div class="y-drop-sep"></div>
      <div class="y-drop-item danger" onclick="YaplyAuth.logout()">🚪 Sign Out</div>
    </div>
  `;
  document.body.insertAdjacentElement('afterbegin', nav);

  // Trip context bar
  if (trip) {
    const bar = document.createElement('div');
    bar.id = 'y-tripbar';
    bar.innerHTML = `
      <div>
        <div class="ytb-dest">${trip.origin||'?'} → ${trip.destination}</div>
        <div class="ytb-meta">${trip.days}d · ${trip.people} people · ${trip.currency} ${trip.budget} · ${trip.vibeString||''}</div>
      </div>
      <div style="display:flex;gap:6px;">
        <button class="ytb-btn" onclick="toggleTripBar()">Hide</button>
        <button class="ytb-btn ytb-btn-solid" onclick="window.location.href='/app'">📋 Dashboard</button>
      </div>
    `;
    document.body.insertAdjacentElement('afterbegin', bar);
    document.body.style.paddingTop = '100px';
  } else {
    document.body.style.paddingTop = '60px';
  }

  // Close dropdown on outside click
  document.addEventListener('click', e => {
    if (!e.target.closest('#y-dropdown') && !e.target.closest('#yUserBtn')) {
      document.getElementById('y-dropdown')?.classList.remove('open');
    }
  });
}

window.toggleDropdown = function() {
  document.getElementById('y-dropdown')?.classList.toggle('open');
};
window.toggleTripBar = function() {
  const bar = document.getElementById('y-tripbar');
  if (bar) {
    bar.classList.toggle('hidden');
    document.body.style.paddingTop = bar.classList.contains('hidden') ? '60px' : '100px';
  }
};

// ── AUTO-FILL FORMS ──
function autoFill() {
  const trip = JSON.parse(localStorage.getItem('yaply_active_trip') || 'null');
  if (!trip) return;

  const map = {
    // Before trip
    'destination': trip.destination, 'origin': trip.origin,
    'days': trip.days, 'people': trip.people, 'budget': trip.budget,
    // Tools
    'passportDest': trip.destination, 'safetyDest': trip.destination,
    'lawsDest': trip.destination, 'jlFrom': trip.origin, 'jlTo': trip.destination,
    'festDest': trip.destination, 'budgetDest': trip.destination,
    'budgetDays': trip.days, 'budgetPeople': trip.people, 'budgetAmount': trip.budget,
    'luggageDest': trip.destination, 'ecDest': trip.destination,
    // During
    'medDest': trip.destination, 'priceDest': trip.destination,
    'scamDest': trip.destination, 'allergyDest': trip.destination,
    'routeCity': trip.destination, 'immigDest': trip.destination?.split(',')[0]?.trim()||'',
    // After
    'jDest': trip.destination, 'jDays': trip.days,
    'jVibe': trip.vibeString||(trip.vibes||[]).join(' + '),
    'eDest': trip.destination, 'stDest': trip.destination, 'stDays': trip.days,
    'ntPast': trip.destination,
  };

  Object.entries(map).forEach(([id, val]) => {
    if (!val) return;
    const el = document.getElementById(id);
    if (el && !el.value) el.value = val;
  });

  ['eCurr','splitCurr','budgetCurr','homeCurrency'].forEach(id => {
    const el = document.getElementById(id);
    if (el && trip.currency) el.value = trip.currency;
  });

  // Fill passport fields
  ['immigPassport','visaPassport'].forEach(id => {
    const el = document.getElementById(id);
    if (el && trip.passport && !el.value) el.value = trip.passport;
  });

  console.log('[Yaply] Auto-filled forms with trip context ✅');
}

// ── HIDE OLD NAVS ──
function hideOldNavs() {
  // Hide any existing nav elements that conflict
  const selectors = [
    'nav:not(#y-nav)',
    '.nav-link[href="/"]',
    '.back-nav',
    'a.nav-link',
  ];
  // Don't hide everything — just add padding fix
}

// ── INIT ──
document.addEventListener('DOMContentLoaded', () => {
  buildNav();
  setTimeout(autoFill, 200);
  hideOldNavs();
});

})();