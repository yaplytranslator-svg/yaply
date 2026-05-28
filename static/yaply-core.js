/**
 * yaply-core.js — Shared foundation for ALL Yaply pages
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Load this in EVERY page inside <head>:
 * <script src="/static/yaply-core.js"></script>
 *
 * Gives every page:
 * ✅ YaplyAuth  — token, user, login/logout
 * ✅ YaplyAPI   — authenticated fetch wrapper
 * ✅ YaplyUI    — toast, loading, error states
 * ✅ YaplyTrip  — active trip state
 */

// ════════════════════════════════════════════════════════
//  YAPLY AUTH
// ════════════════════════════════════════════════════════

var YaplyAuth = (function () {

  var TOKEN_KEY = 'yaply_token';
  var USER_KEY  = 'yaply_user';
  var _user     = null;
  var _token    = null;

  // ── Token management ──────────────────────────────────

  function getToken() {
    if (_token) return _token;
    _token = localStorage.getItem(TOKEN_KEY);
    return _token;
  }

  function setToken(token) {
    _token = token;
    localStorage.setItem(TOKEN_KEY, token);
  }

  function clearToken() {
    _token = null;
    _user  = null;
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  }

  function isLoggedIn() {
    return !!getToken();
  }

  // ── User management ───────────────────────────────────

  function getUser() {
    if (_user) return _user;
    var stored = localStorage.getItem(USER_KEY);
    if (stored) {
      try { _user = JSON.parse(stored); } catch (e) {}
    }
    return _user;
  }

  function setUser(user) {
    _user = user;
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  }

  // ── Load user from API ────────────────────────────────

  async function loadUser() {
    try {
      var res  = await YaplyAPI.get('/api/me');
      if (res.success && res.user) {
        setUser(res.user);
        return res.user;
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  // ── Redirect to login if not authed ──────────────────

  function requireLogin() {
    if (!isLoggedIn()) {
      window.location.href = '/login?redirect=' + encodeURIComponent(window.location.pathname);
      return false;
    }
    return true;
  }

  // ── Logout ────────────────────────────────────────────

  async function logout() {
    try {
      await YaplyAPI.post('/api/auth/logout', {});
    } catch (e) {}
    clearToken();
    window.location.href = '/login';
  }

  // ── Handle Google OAuth token in URL ─────────────────
  // When Google redirects back to /app?token=xxx&welcome=Name

  function captureOAuthToken() {
    var params  = new URLSearchParams(window.location.search);
    var token   = params.get('token');
    var welcome = params.get('welcome');

    if (token) {
      setToken(token);
      // Clean URL — remove token from address bar
      var cleanURL = window.location.pathname;
      window.history.replaceState({}, document.title, cleanURL);
      // Show welcome toast if name provided
      if (welcome) {
        setTimeout(function () {
          YaplyUI.toast('Welcome, ' + decodeURIComponent(welcome) + '! ✈️', 'success');
        }, 500);
      }
      return true;
    }
    return false;
  }

  // ── Handle error/success params from auth redirects ──

  function handleAuthParams() {
    var params  = new URLSearchParams(window.location.search);
    var error   = params.get('error');
    var success = params.get('success');

    if (error) {
      YaplyUI.toast(decodeURIComponent(error), 'error');
      window.history.replaceState({}, document.title, window.location.pathname);
    }
    if (success) {
      YaplyUI.toast(decodeURIComponent(success), 'success');
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  }

  // ── Init — call on every page load ───────────────────

  async function init(options) {
    options = options || {};

    // Capture Google OAuth token first
    captureOAuthToken();
    handleAuthParams();

    // If page requires auth
    if (options.requireAuth !== false) {
      if (!isLoggedIn()) {
        window.location.href = '/login';
        return null;
      }
    }

    // Load fresh user data
    var user = null;
    if (isLoggedIn()) {
      // Try cache first for speed
      user = getUser();
      // Then refresh from API in background
      loadUser().then(function (freshUser) {
        if (freshUser) {
          setUser(freshUser);
          if (options.onUserLoad) options.onUserLoad(freshUser);
          _updateNavAvatar(freshUser);
        } else if (!getUser()) {
          // Token invalid — force logout
          clearToken();
          window.location.href = '/login';
        }
      });
    }

    // Update nav immediately with cached user
    if (user) _updateNavAvatar(user);

    return user;
  }

  // ── Update nav avatar/name everywhere ────────────────

  function _updateNavAvatar(user) {
    var initial = (user.name || 'P').charAt(0).toUpperCase();
    var avatarEls = document.querySelectorAll('#navAvatar, .nav-avatar, [data-nav-avatar]');
    avatarEls.forEach(function (el) {
      if (user.avatar) {
        el.style.backgroundImage = 'url(' + user.avatar + ')';
        el.style.backgroundSize  = 'cover';
        el.textContent = '';
      } else {
        el.textContent = initial;
      }
    });

    var nameEls = document.querySelectorAll('#navUserName, [data-nav-name]');
    nameEls.forEach(function (el) {
      el.textContent = user.name ? user.name.split(' ')[0] : 'Traveller';
    });

    var greetEls = document.querySelectorAll('#greetName, [data-greet-name]');
    greetEls.forEach(function (el) {
      var first = user.name ? user.name.split(' ')[0] : 'Traveller';
      el.innerHTML = 'Welcome back, <em>' + first + '.</em>';
    });
  }

  return {
    getToken:       getToken,
    setToken:       setToken,
    clearToken:     clearToken,
    isLoggedIn:     isLoggedIn,
    getUser:        getUser,
    setUser:        setUser,
    loadUser:       loadUser,
    requireLogin:   requireLogin,
    logout:         logout,
    init:           init,
    captureOAuthToken: captureOAuthToken,
  };

})();


// ════════════════════════════════════════════════════════
//  YAPLY API — authenticated fetch wrapper
// ════════════════════════════════════════════════════════

var YaplyAPI = (function () {

  var BASE = '';  // same origin

  function _headers() {
    var h = { 'Content-Type': 'application/json' };
    var token = YaplyAuth.getToken();
    if (token) h['Authorization'] = 'Bearer ' + token;
    return h;
  }

  async function _fetch(method, path, body) {
    try {
      var opts = {
        method:  method,
        headers: _headers(),
      };
      if (body !== undefined) {
        opts.body = JSON.stringify(body);
      }

      var res  = await fetch(BASE + path, opts);
      var data = await res.json();

      // Handle auth errors globally
      if (res.status === 401) {
        if (data.code === 'EXPIRED' || data.code === 'NO_TOKEN') {
          YaplyAuth.clearToken();
          // Don't redirect on login/auth pages
          var path = window.location.pathname;
          if (path !== '/login' && path !== '/') {
            window.location.href = '/login';
          }
        }
      }

      // Handle rate limit
      if (res.status === 429) {
        YaplyUI.toast('Too many requests. Please slow down.', 'error');
      }

      return data;
    } catch (e) {
      console.error('[YaplyAPI]', method, path, e);
      return { success: false, error: 'Connection error. Please check your internet.' };
    }
  }

  return {
    get:    function (path)         { return _fetch('GET',    path); },
    post:   function (path, body)   { return _fetch('POST',   path, body); },
    put:    function (path, body)   { return _fetch('PUT',    path, body); },
    delete: function (path)         { return _fetch('DELETE', path); },
  };

})();


// ════════════════════════════════════════════════════════
//  YAPLY UI — toasts, loading states, errors
// ════════════════════════════════════════════════════════

var YaplyUI = (function () {

  var _toastTimeout = null;

  // ── Toast notification ────────────────────────────────

  function toast(message, type, duration) {
    type     = type || 'info';
    duration = duration || 3000;

    // Remove existing toast
    var existing = document.getElementById('yaply-toast');
    if (existing) existing.remove();
    if (_toastTimeout) clearTimeout(_toastTimeout);

    var colors = {
      success: '#18C29C',
      error:   '#EF4444',
      info:    '#0B1220',
      warning: '#F5B942',
    };

    var el = document.createElement('div');
    el.id  = 'yaply-toast';
    el.style.cssText = [
      'position:fixed',
      'bottom:88px',
      'left:50%',
      'transform:translateX(-50%) translateY(10px)',
      'background:' + (colors[type] || colors.info),
      'color:white',
      'padding:10px 20px',
      'border-radius:100px',
      'font-family:DM Sans,sans-serif',
      'font-size:13px',
      'font-weight:500',
      'opacity:0',
      'transition:all .3s',
      'z-index:9999',
      'white-space:nowrap',
      'pointer-events:none',
      'box-shadow:0 8px 24px rgba(0,0,0,0.2)',
      'max-width:90vw',
      'text-align:center',
    ].join(';');
    el.textContent = message;
    document.body.appendChild(el);

    // Animate in
    setTimeout(function () {
      el.style.opacity   = '1';
      el.style.transform = 'translateX(-50%) translateY(0)';
    }, 10);

    // Animate out
    _toastTimeout = setTimeout(function () {
      el.style.opacity   = '0';
      el.style.transform = 'translateX(-50%) translateY(10px)';
      setTimeout(function () { if (el.parentNode) el.remove(); }, 300);
    }, duration);
  }

  // ── Button loading state ──────────────────────────────

  function setLoading(btnEl, loading, originalText) {
    if (!btnEl) return;
    if (loading) {
      btnEl.disabled             = true;
      btnEl.dataset.originalText = btnEl.innerHTML;
      btnEl.innerHTML = (
        '<span style="display:inline-flex;align-items:center;gap:8px;">' +
        '<span style="width:14px;height:14px;border:2px solid rgba(255,255,255,0.3);' +
        'border-top-color:white;border-radius:50%;animation:yaply-spin .7s linear infinite;"></span>' +
        (originalText || 'Loading...') +
        '</span>'
      );
    } else {
      btnEl.disabled  = false;
      btnEl.innerHTML = btnEl.dataset.originalText || originalText || 'Submit';
    }
  }

  // ── Empty state ───────────────────────────────────────

  function emptyState(containerEl, icon, title, subtitle, ctaText, ctaHref) {
    if (!containerEl) return;
    containerEl.innerHTML = (
      '<div style="display:flex;flex-direction:column;align-items:center;text-align:center;padding:48px 24px;">' +
      '<div style="width:64px;height:64px;border-radius:20px;background:rgba(37,99,255,0.08);' +
      'display:flex;align-items:center;justify-content:center;margin-bottom:16px;">' +
      icon +
      '</div>' +
      '<div style="font-family:Instrument Serif,Georgia,serif;font-size:20px;color:#111827;' +
      'letter-spacing:-0.3px;margin-bottom:8px;">' + title + '</div>' +
      '<div style="font-size:14px;color:#6B7280;font-weight:300;line-height:1.6;max-width:240px;margin-bottom:24px;">' + subtitle + '</div>' +
      (ctaText ? '<a href="' + ctaHref + '" style="padding:12px 24px;background:#2563FF;color:white;border-radius:12px;' +
      'font-family:DM Sans,sans-serif;font-size:14px;font-weight:600;text-decoration:none;">' + ctaText + '</a>' : '') +
      '</div>'
    );
  }

  // ── Skeleton loader ───────────────────────────────────

  function skeleton(height, borderRadius) {
    height       = height || '20px';
    borderRadius = borderRadius || '8px';
    return (
      '<div style="height:' + height + ';border-radius:' + borderRadius + ';' +
      'background:linear-gradient(90deg,#F0F0F0 25%,#E0E0E0 50%,#F0F0F0 75%);' +
      'background-size:200% 100%;animation:yaply-shimmer 1.5s infinite;"></div>'
    );
  }

  // ── Inject CSS once ───────────────────────────────────

  function _injectCSS() {
    if (document.getElementById('yaply-core-css')) return;
    var style = document.createElement('style');
    style.id  = 'yaply-core-css';
    style.textContent = (
      '@keyframes yaply-spin{to{transform:rotate(360deg)}}' +
      '@keyframes yaply-shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}'
    );
    document.head.appendChild(style);
  }

  _injectCSS();

  return {
    toast:      toast,
    setLoading: setLoading,
    emptyState: emptyState,
    skeleton:   skeleton,
  };

})();


// ════════════════════════════════════════════════════════
//  YAPLY TRIP — active trip state shared across pages
// ════════════════════════════════════════════════════════

var YaplyTrip = (function () {

  var TRIP_KEY = 'yaply_active_trip';
  var _trip    = null;
  var _trips   = [];

  function getActiveTrip() {
    if (_trip) return _trip;
    var stored = localStorage.getItem(TRIP_KEY);
    if (stored) {
      try { _trip = JSON.parse(stored); } catch (e) {}
    }
    return _trip;
  }

  function setActiveTrip(trip) {
    _trip = trip;
    if (trip) {
      localStorage.setItem(TRIP_KEY, JSON.stringify(trip));
    } else {
      localStorage.removeItem(TRIP_KEY);
    }
  }

  async function loadTrips() {
    try {
      var res = await YaplyAPI.get('/api/trips');
      if (res.success && res.trips) {
        _trips = res.trips;
        // Set active trip (first active one)
        var active = res.trips.find(function (t) { return t.status === 'active'; });
        if (active) setActiveTrip(active);
        return res.trips;
      }
      return [];
    } catch (e) {
      return [];
    }
  }

  function getFlag(destination) {
    var map = {
      'Japan': '🇯🇵', 'Tokyo': '🇯🇵', 'Kyoto': '🇯🇵', 'Osaka': '🇯🇵',
      'Thailand': '🇹🇭', 'Bangkok': '🇹🇭', 'Phuket': '🇹🇭', 'Chiang Mai': '🇹🇭',
      'France': '🇫🇷', 'Paris': '🇫🇷',
      'Indonesia': '🇮🇩', 'Bali': '🇮🇩', 'Jakarta': '🇮🇩',
      'UAE': '🇦🇪', 'Dubai': '🇦🇪', 'Abu Dhabi': '🇦🇪',
      'UK': '🇬🇧', 'London': '🇬🇧',
      'Singapore': '🇸🇬',
      'USA': '🇺🇸', 'New York': '🇺🇸', 'Los Angeles': '🇺🇸',
      'India': '🇮🇳', 'Delhi': '🇮🇳', 'Mumbai': '🇮🇳', 'Goa': '🇮🇳',
      'Nepal': '🇳🇵', 'Kathmandu': '🇳🇵',
      'Sri Lanka': '🇱🇰', 'Colombo': '🇱🇰',
      'Malaysia': '🇲🇾', 'Kuala Lumpur': '🇲🇾',
      'Vietnam': '🇻🇳', 'Hanoi': '🇻🇳', 'Ho Chi Minh': '🇻🇳',
      'Italy': '🇮🇹', 'Rome': '🇮🇹', 'Milan': '🇮🇹',
      'Spain': '🇪🇸', 'Barcelona': '🇪🇸', 'Madrid': '🇪🇸',
      'Germany': '🇩🇪', 'Berlin': '🇩🇪',
      'Greece': '🇬🇷', 'Athens': '🇬🇷', 'Santorini': '🇬🇷',
      'Turkey': '🇹🇷', 'Istanbul': '🇹🇷',
      'Egypt': '🇪🇬', 'Cairo': '🇪🇬',
      'South Africa': '🇿🇦', 'Cape Town': '🇿🇦',
      'Australia': '🇦🇺', 'Sydney': '🇦🇺', 'Melbourne': '🇦🇺',
      'Canada': '🇨🇦', 'Toronto': '🇨🇦', 'Vancouver': '🇨🇦',
      'Mexico': '🇲🇽', 'Cancun': '🇲🇽',
      'Peru': '🇵🇪', 'Machu Picchu': '🇵🇪',
      'Brazil': '🇧🇷', 'Rio': '🇧🇷',
      'Portugal': '🇵🇹', 'Lisbon': '🇵🇹',
      'Netherlands': '🇳🇱', 'Amsterdam': '🇳🇱',
      'Switzerland': '🇨🇭', 'Zurich': '🇨🇭',
      'Austria': '🇦🇹', 'Vienna': '🇦🇹',
      'Czech Republic': '🇨🇿', 'Prague': '🇨🇿',
      'Hungary': '🇭🇺', 'Budapest': '🇭🇺',
      'South Korea': '🇰🇷', 'Seoul': '🇰🇷',
      'China': '🇨🇳', 'Beijing': '🇨🇳', 'Shanghai': '🇨🇳',
      'Maldives': '🇲🇻',
      'Bhutan': '🇧🇹',
      'Cambodia': '🇰🇭', 'Siem Reap': '🇰🇭',
      'Philippines': '🇵🇭', 'Manila': '🇵🇭',
      'Jordan': '🇯🇴', 'Petra': '🇯🇴',
      'Morocco': '🇲🇦', 'Marrakech': '🇲🇦',
    };
    if (!destination) return '🌍';
    var dest = destination.split(',')[0].trim();
    var flag = '🌍';
    Object.keys(map).forEach(function (k) {
      if (destination.includes(k)) flag = map[k];
    });
    return flag;
  }

  function getDayNumber(trip) {
    if (!trip) return 1;
    var start = new Date(trip.start_date || trip.created_at || Date.now());
    var today = new Date();
    var diff  = Math.floor((today - start) / 86400000);
    return Math.max(1, diff + 1);
  }

  function getProgress(trip) {
    if (!trip) return 0;
    var day   = getDayNumber(trip);
    var total = parseInt(trip.days) || 1;
    return Math.min(100, Math.round((day / total) * 100));
  }

  return {
    getActiveTrip: getActiveTrip,
    setActiveTrip: setActiveTrip,
    loadTrips:     loadTrips,
    getFlag:       getFlag,
    getDayNumber:  getDayNumber,
    getProgress:   getProgress,
    getAll:        function () { return _trips; },
  };

})();


// ════════════════════════════════════════════════════════
//  AUTO-INIT on every page
// ════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', function () {

  var path = window.location.pathname;

  // Pages that don't need auth
  var publicPages = ['/', '/login', '/join'];
  var isPublic    = publicPages.some(function (p) { return path === p || path.startsWith('/join/'); });

  if (isPublic) {
    // On login page — capture auth params
    YaplyAuth.captureOAuthToken();
    // If already logged in → redirect to app
    if (path === '/login' && YaplyAuth.isLoggedIn()) {
      window.location.href = '/app';
    }
    return;
  }

  // All other pages — require auth + load user
  YaplyAuth.init({
    requireAuth: true,
    onUserLoad:  function (user) {
      // Dispatch event so page-specific scripts can react
      document.dispatchEvent(new CustomEvent('yaply:userloaded', { detail: user }));
    }
  });

});