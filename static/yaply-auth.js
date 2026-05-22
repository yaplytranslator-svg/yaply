/**
 * yaply-auth.js — Auth page logic (login + signup)
 * Use ONLY on auth.html
 */

async function handleSignup(e) {
  e.preventDefault();
  const name     = document.getElementById('name')?.value.trim();
  const email    = document.getElementById('email')?.value.trim();
  const password = document.getElementById('password')?.value.trim();
  const btn      = document.getElementById('authBtn');

  if (!name || !email || !password) { showToast('Please fill all fields', 'error'); return; }

  btn.disabled = true;
  btn.textContent = 'Creating account...';

  const res = await fetch('/api/auth/signup', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password })
  }).then(r => r.json());

  if (res.success) {
    setAuth(res.token, res.user, true);
    showToast(res.message || 'Welcome to Yaply!', 'success');
    setTimeout(() => { window.location.href = '/app'; }, 1000);
  } else {
    showToast(res.error || 'Signup failed', 'error');
    btn.disabled = false;
    btn.textContent = 'Create Account';
  }
}

async function handleLogin(e) {
  e.preventDefault();
  const email    = document.getElementById('email')?.value.trim();
  const password = document.getElementById('password')?.value.trim();
  const btn      = document.getElementById('authBtn');

  if (!email || !password) { showToast('Please fill all fields', 'error'); return; }

  btn.disabled = true;
  btn.textContent = 'Logging in...';

  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  }).then(r => r.json());

  if (res.success) {
    setAuth(res.token, res.user, true);
    showToast(res.message || 'Welcome back!', 'success');
    const next = new URLSearchParams(window.location.search).get('next') || '/app';
    setTimeout(() => { window.location.href = next; }, 800);
  } else {
    showToast(res.error || 'Login failed', 'error');
    btn.disabled = false;
    btn.textContent = 'Log In';
  }
}

async function handleGoogleLogin(id_token) {
  showToast('Verifying with Google...', 'info');
  const res = await fetch('/api/auth/google', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id_token })
  }).then(r => r.json());

  if (res.success) {
    setAuth(res.token, res.user, true);
    showToast(res.message || 'Welcome!', 'success');
    const next = new URLSearchParams(window.location.search).get('next') || '/app';
    setTimeout(() => { window.location.href = next; }, 800);
  } else {
    showToast(res.error || 'Google login failed', 'error');
  }
}

// If already logged in, redirect
if (getToken() && !window.location.search.includes('logout')) {
  window.location.href = '/app';
}

window.handleSignup      = handleSignup;
window.handleLogin       = handleLogin;
window.handleGoogleLogin = handleGoogleLogin;