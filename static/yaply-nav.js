/**
 * yaply-nav.js v3 — Universal Auth + API Helper
 * Add to EVERY page: <script src="/static/yaply-nav.js"></script>
 * This file handles: token storage, apiFetch(), auth redirect, toast
 */

// ── TOKEN MANAGEMENT ──
const YAPLY_TOKEN_KEY = 'yaply_token';
const YAPLY_USER_KEY  = 'yaply_user';

function getToken() {
  return localStorage.getItem(YAPLY_TOKEN_KEY) || sessionStorage.getItem(YAPLY_TOKEN_KEY);
}

function getUser() {
  try {
    const u = localStorage.getItem(YAPLY_USER_KEY) || sessionStorage.getItem(YAPLY_USER_KEY);
    return u ? JSON.parse(u) : null;
  } catch { return null; }
}

function setAuth(token, user, remember = true) {
  const store = remember ? localStorage : sessionStorage;
  store.setItem(YAPLY_TOKEN_KEY, token);
  store.setItem(YAPLY_USER_KEY, JSON.stringify(user));
}

function clearAuth() {
  localStorage.removeItem(YAPLY_TOKEN_KEY);
  localStorage.removeItem(YAPLY_USER_KEY);
  sessionStorage.removeItem(YAPLY_TOKEN_KEY);
  sessionStorage.removeItem(YAPLY_USER_KEY);
}

// ── CORE apiFetch — always attaches token ──
async function apiFetch(url, options = {}) {
  const token = getToken();
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  try {
    const response = await fetch(url, { ...options, headers });

    if (response.status === 401) {
      const data = await response.json().catch(() => ({}));
      if (data.code === 'EXPIRED' || data.code === 'NO_TOKEN') {
        clearAuth();
        if (!window.location.pathname.includes('/login')) {
          showToast('Session expired — please log in again', 'error');
          setTimeout(() => { window.location.href = '/login'; }, 1500);
        }
      }
      return data;
    }

    return await response.json();
  } catch (err) {
    console.error('[apiFetch error]', url, err);
    return { success: false, error: 'Network error — check your connection' };
  }
}

// ── REQUIRE LOGIN ──
function requireLogin() {
  if (!getToken()) {
    window.location.href = '/login?next=' + encodeURIComponent(window.location.href);
    return false;
  }
  return true;
}

// ── LOGOUT ──
function logout() {
  clearAuth();
  window.location.href = '/login';
}

// ── TOAST ──
function showToast(message, type = 'info', duration = 3000) {
  document.getElementById('_yaply_toast')?.remove();
  const bg = { info:'#1A1A2E', success:'#10B981', error:'#EF4444', warning:'#F59E0B' }[type] || '#1A1A2E';
  const t = document.createElement('div');
  t.id = '_yaply_toast';
  t.style.cssText = `position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(10px);
    background:${bg};color:#fff;padding:12px 24px;border-radius:100px;
    font-family:'Nunito',sans-serif;font-size:0.82rem;font-weight:700;
    box-shadow:0 8px 32px rgba(0,0,0,0.3);z-index:99999;
    white-space:nowrap;pointer-events:none;
    opacity:0;transition:all 0.3s ease;`;
  t.textContent = message;
  document.body.appendChild(t);
  requestAnimationFrame(() => {
    t.style.opacity = '1';
    t.style.transform = 'translateX(-50%) translateY(0)';
  });
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateX(-50%) translateY(10px)';
    setTimeout(() => t.remove(), 300);
  }, duration);
}

// ── AUTO INJECT NAV BAR ──
(function injectNav() {
  if (document.getElementById('_yaply_nav')) return;
  if (['/login','/'].includes(window.location.pathname)) return;

  const user = getUser();
  const isLoggedIn = !!getToken();
  const p = window.location.pathname;

  const links = [
    { href:'/plan',      label:'✈️ Plan',       active: p.startsWith('/plan') },
    { href:'/translate', label:'🎙️ Live',        active: p.startsWith('/translate') },
    { href:'/convo',     label:'💬 Convo',        active: p.startsWith('/convo') },
    { href:'/camera',    label:'📷 Camera',       active: p.startsWith('/camera') },
    { href:'/discover',  label:'🔍 Discover',     active: p.startsWith('/discover') },
    { href:'/during',    label:'🌍 During',        active: p.startsWith('/during') },
    { href:'/after',     label:'📖 After',        active: p.startsWith('/after') },
  ];

  const nav = document.createElement('nav');
  nav.id = '_yaply_nav';
  nav.style.cssText = `
    position:fixed;top:0;left:0;right:0;
    background:rgba(10,10,20,0.97);
    backdrop-filter:blur(20px);
    border-bottom:1px solid rgba(255,255,255,0.06);
    display:flex;align-items:center;
    justify-content:space-between;
    padding:10px 24px;z-index:1000;
    font-family:'Nunito',sans-serif;
  `;

  // Logo
  const logo = document.createElement('a');
  logo.href = isLoggedIn ? '/app' : '/';
  logo.style.cssText = 'font-size:1.2rem;font-weight:900;color:#fff;text-decoration:none;letter-spacing:-0.5px;flex-shrink:0;';
  logo.innerHTML = 'Yaply<span style="color:#38BDF8">.</span>';

  // Links
  const linkWrap = document.createElement('div');
  linkWrap.style.cssText = 'display:flex;align-items:center;gap:4px;overflow-x:auto;scrollbar-width:none;';
  links.forEach(l => {
    const a = document.createElement('a');
    a.href = l.href;
    a.textContent = l.label;
    a.style.cssText = `
      padding:6px 12px;border-radius:100px;text-decoration:none;
      font-size:0.72rem;font-weight:700;white-space:nowrap;transition:all 0.2s;
      background:${l.active ? 'rgba(56,189,248,0.15)' : 'transparent'};
      color:${l.active ? '#38BDF8' : 'rgba(255,255,255,0.5)'};
    `;
    a.onmouseover = () => { if (!l.active) { a.style.color='#fff'; a.style.background='rgba(255,255,255,0.07)'; } };
    a.onmouseout  = () => { if (!l.active) { a.style.color='rgba(255,255,255,0.5)'; a.style.background='transparent'; } };
    linkWrap.appendChild(a);
  });

  // Right section
  const right = document.createElement('div');
  right.style.cssText = 'display:flex;align-items:center;gap:8px;flex-shrink:0;';

  if (isLoggedIn && user) {
    const avatar = document.createElement('div');
    avatar.style.cssText = 'width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#38BDF8,#6366F1);display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:800;color:#fff;cursor:pointer;transition:transform 0.15s,box-shadow 0.15s;overflow:hidden;';
    if (user.avatar) {
      avatar.innerHTML = `<img src="${user.avatar}" alt="${user.name}" style="width:100%;height:100%;object-fit:cover;"/>`;
    } else {
      avatar.textContent = (user.name || 'U')[0].toUpperCase();
    }
    avatar.title = 'View Profile';
    avatar.onmouseover = () => { avatar.style.transform='scale(1.12)'; avatar.style.boxShadow='0 4px 12px rgba(56,189,248,0.4)'; };
    avatar.onmouseout  = () => { avatar.style.transform='scale(1)';    avatar.style.boxShadow='none'; };
    avatar.onclick = () => { window.location.href = '/profile'; };

    const logoutBtn = document.createElement('button');
    logoutBtn.textContent = 'Logout';
    logoutBtn.onclick = logout;
    logoutBtn.style.cssText = 'padding:6px 14px;border-radius:100px;border:1px solid rgba(255,255,255,0.15);background:transparent;color:rgba(255,255,255,0.5);font-family:inherit;font-size:0.72rem;font-weight:700;cursor:pointer;';
    right.appendChild(avatar);
    right.appendChild(logoutBtn);
  } else {
    const loginBtn = document.createElement('a');
    loginBtn.href = '/login';
    loginBtn.textContent = 'Login';
    loginBtn.style.cssText = 'padding:8px 18px;border-radius:100px;background:#38BDF8;color:#000;font-size:0.78rem;font-weight:800;text-decoration:none;';
    right.appendChild(loginBtn);
  }

  nav.appendChild(logo);
  nav.appendChild(linkWrap);
  nav.appendChild(right);
  document.body.prepend(nav);

  // Push body down
  document.body.style.paddingTop = '54px';
})();

// ── AUTO CHECK AUTH ON PROTECTED PAGES ──
(function autoCheck() {
  const openPages = ['/', '/login'];
  if (openPages.includes(window.location.pathname)) return;

  const token = getToken();
  if (!token) {
    // Give page 100ms to handle it itself
    setTimeout(() => {
      if (!getToken()) requireLogin();
    }, 100);
  }
})();

// Expose globally
window.apiFetch   = apiFetch;
window.getToken   = getToken;
window.getUser    = getUser;
window.setAuth    = setAuth;
window.clearAuth  = clearAuth;
window.logout     = logout;
window.showToast  = showToast;
window.requireLogin = requireLogin;