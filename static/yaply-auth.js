/**
 * yaply-auth.js
 * Add to EVERY page: <script src="/static/yaply-auth.js"></script>
 * 
 * Does 3 things:
 * 1. Checks if user is logged in on every page
 * 2. Adds auth token to every API call automatically
 * 3. Redirects to /login if not authenticated
 */

(function() {

// ── TOKEN HELPERS ──
window.YaplyAuth = {
  getToken: () => localStorage.getItem('yaply_token'),
  getUser:  () => JSON.parse(localStorage.getItem('yaply_user') || 'null'),
  isLoggedIn: () => !!localStorage.getItem('yaply_token'),
  
  logout() {
    localStorage.removeItem('yaply_token');
    localStorage.removeItem('yaply_user');
    window.location.href = '/login';
  },

  // Call this on protected pages
  requireLogin() {
    if (!this.isLoggedIn()) {
      window.location.href = '/login?redirect=' + encodeURIComponent(window.location.pathname);
      return false;
    }
    return true;
  }
};

// ── AUTHENTICATED FETCH ──
// Use this instead of fetch() everywhere
// Automatically adds Bearer token to every request
window.apiFetch = async function(url, options = {}) {
  const token = YaplyAuth.getToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };
  if (token) headers['Authorization'] = 'Bearer ' + token;

  const res = await fetch(url, { ...options, headers });

  // If 401 — token expired, redirect to login
  if (res.status === 401) {
    const data = await res.json().catch(() => ({}));
    if (data.code === 'EXPIRED' || data.code === 'NO_TOKEN') {
      YaplyAuth.logout();
      return { success: false, error: 'Session expired' };
    }
  }
  return res.json();
};

// ── INJECT TOP BAR ──
function injectTopBar() {
  const user = YaplyAuth.getUser();
  if (!user) return;

  const bar = document.createElement('div');
  bar.id = 'yaply-topbar';
  bar.style.cssText = `
    position:fixed;top:0;left:0;right:0;z-index:9999;
    background:rgba(10,10,10,0.97);backdrop-filter:blur(20px);
    height:48px;display:flex;align-items:center;
    justify-content:space-between;padding:0 20px;
    border-bottom:1px solid rgba(255,255,255,0.06);
    font-family:'Geist','DM Sans',sans-serif;
  `;

  // Get trip from localStorage or server
  const activeTrip = JSON.parse(localStorage.getItem('yaply_active_trip') || 'null');
  const tripPill = activeTrip ? `
    <div onclick="window.location.href='/app'" style="
      display:flex;align-items:center;gap:7px;
      padding:5px 14px;border-radius:100px;
      background:rgba(232,99,58,0.15);
      border:1px solid rgba(232,99,58,0.25);
      cursor:pointer;font-size:0.72rem;font-weight:700;
      color:#E8633A;
    ">
      <div style="width:5px;height:5px;border-radius:50%;background:#E8633A;animation:pulse 2s infinite;"></div>
      ✈️ ${activeTrip.destination}
    </div>
  ` : `
    <a href="/app" style="
      padding:5px 14px;border-radius:100px;
      background:rgba(255,255,255,0.07);
      border:1px solid rgba(255,255,255,0.1);
      font-size:0.72rem;font-weight:700;
      color:rgba(255,255,255,0.6);
      text-decoration:none;
    ">+ New Trip</a>
  `;

  bar.innerHTML = `
    <style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}</style>
    <a href="/app" style="font-family:'Playfair Display','Georgia',serif;font-size:1.2rem;font-weight:700;color:white;text-decoration:none;letter-spacing:-0.5px;">
      ya<em style="color:#E8633A;font-style:italic;">ply</em>
    </a>

    <div style="display:flex;align-items:center;gap:8px;">
      ${tripPill}
      
      <div style="display:flex;align-items:center;gap:6px;padding:5px 12px;border-radius:100px;background:rgba(255,255,255,0.06);cursor:pointer;" onclick="toggleUserMenu()">
        ${user.avatar 
          ? `<img src="${user.avatar}" style="width:22px;height:22px;border-radius:50%;object-fit:cover;"/>`
          : `<div style="width:22px;height:22px;border-radius:50%;background:linear-gradient(135deg,#E8633A,#D4522A);display:flex;align-items:center;justify-content:center;font-size:0.65rem;color:white;font-weight:700;">${user.name?.[0]?.toUpperCase()||'U'}</div>`
        }
        <span style="font-size:0.75rem;color:rgba(255,255,255,0.7);font-weight:600;max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${user.name?.split(' ')[0] || 'Me'}</span>
        <span style="font-size:0.6rem;color:rgba(255,255,255,0.3);">▾</span>
      </div>
    </div>

    <!-- User dropdown -->
    <div id="userMenu" style="
      display:none;position:fixed;top:52px;right:12px;
      background:rgba(18,18,18,0.98);backdrop-filter:blur(20px);
      border:1px solid rgba(255,255,255,0.1);border-radius:16px;
      padding:8px;min-width:200px;
      box-shadow:0 20px 60px rgba(0,0,0,0.4);
      z-index:10000;
      font-family:'Geist','DM Sans',sans-serif;
    ">
      <div style="padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:6px;">
        <div style="font-size:0.82rem;font-weight:700;color:white;">${user.name}</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);margin-top:2px;">${user.email}</div>
      </div>
      <a href="/app" onclick="closeUserMenu()" style="${menuItemStyle()}">🏠 Home</a>
      <a href="/plan" onclick="closeUserMenu()" style="${menuItemStyle()}">✈️ Plan Trip</a>
      <a href="/during" onclick="closeUserMenu()" style="${menuItemStyle()}">🆘 During Trip</a>
      <a href="/after" onclick="closeUserMenu()" style="${menuItemStyle()}">📖 After Trip</a>
      <a href="/discover" onclick="closeUserMenu()" style="${menuItemStyle()}">🔍 Discover</a>
      <a href="/translate" onclick="closeUserMenu()" style="${menuItemStyle()}">🎙️ Translate</a>
      <a href="/tools" onclick="closeUserMenu()" style="${menuItemStyle()}">🛠️ Tools</a>
      <div style="border-top:1px solid rgba(255,255,255,0.07);margin-top:6px;padding-top:6px;">
        <div onclick="YaplyAuth.logout()" style="${menuItemStyle('rgba(239,68,68,0.1)','#F87171')}">🚪 Sign Out</div>
      </div>
    </div>
  `;

  document.body.insertAdjacentElement('afterbegin', bar);

  // Push page content down
  document.body.style.paddingTop = '48px';

  // Close menu on outside click
  document.addEventListener('click', e => {
    if (!e.target.closest('#userMenu') && !e.target.closest('[onclick="toggleUserMenu()"]')) {
      closeUserMenu();
    }
  });
}

function menuItemStyle(bg = 'transparent', color = 'rgba(255,255,255,0.7)') {
  return `
    display:block;padding:9px 12px;border-radius:10px;
    font-size:0.8rem;font-weight:600;color:${color};
    text-decoration:none;cursor:pointer;
    background:${bg};transition:background 0.15s;
    font-family:'Geist','DM Sans',sans-serif;
  `;
}

window.toggleUserMenu = function() {
  const menu = document.getElementById('userMenu');
  if (menu) menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
};

window.closeUserMenu = function() {
  const menu = document.getElementById('userMenu');
  if (menu) menu.style.display = 'none';
};

// ── AUTO-FILL FORMS WITH TRIP CONTEXT ──
function autoFill() {
  const trip = JSON.parse(localStorage.getItem('yaply_active_trip') || 'null');
  if (!trip) return;

  const map = {
    // Before trip
    'destination': trip.destination,
    'origin': trip.origin,
    'days': trip.days,
    'people': trip.people,
    'budget': trip.budget,
    // Tools
    'passportDest': trip.destination,
    'safetyDest': trip.destination,
    'lawsDest': trip.destination,
    'jlFrom': trip.origin,
    'jlTo': trip.destination,
    'festDest': trip.destination,
    'budgetDest': trip.destination,
    'budgetDays': trip.days,
    'budgetPeople': trip.people,
    'budgetAmount': trip.budget,
    'luggageDest': trip.destination,
    'ecDest': trip.destination,
    // During trip
    'medDest': trip.destination,
    'priceDest': trip.destination,
    'scamDest': trip.destination,
    'allergyDest': trip.destination,
    'routeCity': trip.destination,
    'immigDest': trip.destination?.split(',')[0]?.trim() || '',
    // After trip
    'jDest': trip.destination,
    'jDays': trip.days,
    'jVibe': trip.vibeString || (trip.vibes||[]).join(' + '),
    'eDest': trip.destination,
    'stDest': trip.destination,
    'stDays': trip.days,
    'ntPast': trip.destination,
  };

  Object.entries(map).forEach(([id, val]) => {
    if (!val) return;
    const el = document.getElementById(id);
    if (el && !el.value) el.value = val;
  });

  // Fill currency dropdowns
  ['eCurr','splitCurr','budgetCurr'].forEach(id => {
    const el = document.getElementById(id);
    if (el && trip.currency) el.value = trip.currency;
  });
}

// ── INIT ──
document.addEventListener('DOMContentLoaded', () => {
  // Check login on protected pages
  const publicPages = ['/', '/login', '/landing'];
  const isPublic = publicPages.includes(window.location.pathname);

  if (!isPublic && !YaplyAuth.isLoggedIn()) {
    window.location.href = '/login?redirect=' + encodeURIComponent(window.location.pathname);
    return;
  }

  if (YaplyAuth.isLoggedIn()) {
    injectTopBar();
    setTimeout(autoFill, 150);
  }
});

})();