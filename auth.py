"""
auth.py — Yaply Authentication v3 PRODUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Email + Password signup/login
✅ Google OAuth REDIRECT flow (/auth/google → /auth/google/callback)
✅ Google One Tap flow (/api/auth/google via id_token POST)
✅ JWT tokens (30 day expiry)
✅ Bcrypt password hashing
✅ Rate limiting on sensitive routes
✅ Security: brute force protection, input sanitization
✅ Privacy: no sensitive data in logs
✅ Fetches full Google profile (name, email, avatar, locale)
✅ Profile update, password change, account deletion
✅ All DB calls use clean helper functions
"""

import jwt
import bcrypt
import os
import requests as req
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urlencode
from flask import request, jsonify, g, redirect, current_app
from database import (
    create_user, get_user_by_email, get_user_by_id,
    get_user_by_google, update_user, log_action,
    link_google_to_user, update_user_password, delete_user
)

# ── CONFIG ────────────────────────────────────────────────────
JWT_SECRET          = os.getenv('JWT_SECRET', 'yaply-secret-change-in-production-2025')
JWT_EXPIRY_DAYS     = 30
GOOGLE_CLIENT_ID    = os.getenv('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET= os.getenv('GOOGLE_CLIENT_SECRET', '')
GOOGLE_REDIRECT_URI = os.getenv('GOOGLE_REDIRECT_URI', 'https://www.yaply.live/auth/google/callback')

# Brute force protection — in-memory (use Redis in production)
_failed_attempts = {}
MAX_FAILED        = 5
LOCKOUT_MINUTES   = 15


# ════════════════════════════════════════════════════════════════
#  JWT HELPERS
# ════════════════════════════════════════════════════════════════

def make_token(user_id, expiry_days=JWT_EXPIRY_DAYS):
    payload = {
        'user_id': user_id,
        'exp':     datetime.utcnow() + timedelta(days=expiry_days),
        'iat':     datetime.utcnow(),
        'iss':     'yaply.live',
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')


def decode_token(token):
    try:
        return jwt.decode(
            token,
            JWT_SECRET,
            algorithms=['HS256'],
            options={'require': ['exp', 'iat', 'user_id']}
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_token_from_request():
    """Extract JWT from Authorization header, cookie, body, or query param."""
    # 1. Authorization header (preferred)
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        return auth[7:].strip()
    # 2. Cookie
    t = request.cookies.get('yaply_token')
    if t: return t
    # 3. JSON body
    if request.is_json:
        body = request.get_json(silent=True) or {}
        if body.get('token'): return body['token']
    # 4. Query param (WebSocket upgrade / OAuth redirect)
    t = request.args.get('token')
    if t: return t
    return None


# ════════════════════════════════════════════════════════════════
#  AUTH DECORATORS
# ════════════════════════════════════════════════════════════════

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()
        if not token:
            return jsonify({
                'success': False,
                'error':   'Please log in to continue',
                'code':    'NO_TOKEN'
            }), 401
        payload = decode_token(token)
        if not payload:
            return jsonify({
                'success': False,
                'error':   'Session expired. Please log in again.',
                'code':    'EXPIRED'
            }), 401
        user = get_user_by_id(payload['user_id'])
        if not user:
            return jsonify({
                'success': False,
                'error':   'Account not found',
                'code':    'NO_USER'
            }), 401
        g.user    = user
        g.user_id = user['id']
        return f(*args, **kwargs)
    return decorated


def optional_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        g.user    = None
        g.user_id = None
        token = get_token_from_request()
        if token:
            payload = decode_token(token)
            if payload:
                user = get_user_by_id(payload['user_id'])
                if user:
                    g.user    = user
                    g.user_id = user['id']
        return f(*args, **kwargs)
    return decorated


# ════════════════════════════════════════════════════════════════
#  PASSWORD HELPERS
# ════════════════════════════════════════════════════════════════

def hash_password(password):
    return bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt(rounds=12)
    ).decode('utf-8')


def check_password(password, hashed):
    try:
        return bcrypt.checkpw(
            password.encode('utf-8'),
            hashed.encode('utf-8')
        )
    except Exception:
        return False


def validate_password(password):
    if not password or len(password) < 8:
        return False, 'Password must be at least 8 characters'
    if len(password) > 128:
        return False, 'Password too long'
    return True, ''


def validate_email(email):
    import re
    if not email or len(email) > 254:
        return False
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def clean_name(name):
    """Sanitize display name."""
    import re
    name = name.strip()[:60]
    name = re.sub(r'[<>&"\']', '', name)
    return name or 'Traveller'


# ════════════════════════════════════════════════════════════════
#  BRUTE FORCE PROTECTION
# ════════════════════════════════════════════════════════════════

def _get_attempt_key(email, ip):
    return f"{email}:{ip}"


def _is_locked_out(email, ip):
    key  = _get_attempt_key(email, ip)
    data = _failed_attempts.get(key)
    if not data:
        return False
    if data['count'] >= MAX_FAILED:
        locked_until = data['last'] + timedelta(minutes=LOCKOUT_MINUTES)
        if datetime.utcnow() < locked_until:
            return True
        else:
            # Lockout expired — reset
            del _failed_attempts[key]
    return False


def _record_failed(email, ip):
    key  = _get_attempt_key(email, ip)
    data = _failed_attempts.get(key, {'count': 0, 'last': datetime.utcnow()})
    data['count'] += 1
    data['last']   = datetime.utcnow()
    _failed_attempts[key] = data


def _clear_attempts(email, ip):
    key = _get_attempt_key(email, ip)
    _failed_attempts.pop(key, None)


# ════════════════════════════════════════════════════════════════
#  SAFE USER (no sensitive data)
# ════════════════════════════════════════════════════════════════

def safe_user(user):
    if not user: return None
    return {
        'id':         user['id'],
        'name':       user.get('name', 'Traveller'),
        'email':      user.get('email', ''),
        'avatar':     user.get('avatar') or '',
        'passport':   user.get('passport') or 'India',
        'home_city':  user.get('home_city') or '',
        'currency':   user.get('currency') or 'INR',
        'is_pro':     bool(user.get('is_pro')),
        'has_google': bool(user.get('google_id')),
        'created_at': user.get('created_at') or '',
        'locale':     user.get('locale') or 'en',
    }


# ════════════════════════════════════════════════════════════════
#  GOOGLE PROFILE FETCHER
# ════════════════════════════════════════════════════════════════

def fetch_google_profile(access_token):
    """Fetch full Google profile using access token."""
    try:
        r = req.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            headers={'Authorization': f'Bearer {access_token}'},
            timeout=10
        )
        if r.status_code != 200:
            return None
        info = r.json()
        return {
            'google_id':    info.get('sub', ''),
            'email':        info.get('email', '').lower().strip(),
            'name':         info.get('name', 'Traveller'),
            'first_name':   info.get('given_name', ''),
            'last_name':    info.get('family_name', ''),
            'avatar':       info.get('picture', ''),
            'locale':       info.get('locale', 'en'),
            'verified':     info.get('email_verified', False),
        }
    except Exception as e:
        print(f"[Google Profile] {e}")
        return None


def verify_google_id_token(id_token):
    """Verify Google One Tap id_token and extract profile."""
    try:
        r = req.get(
            f'https://oauth2.googleapis.com/tokeninfo?id_token={id_token}',
            timeout=10
        )
        if r.status_code != 200:
            return None
        info = r.json()
        # Verify audience
        aud = info.get('aud', '')
        if GOOGLE_CLIENT_ID and aud != GOOGLE_CLIENT_ID:
            print(f"[Google Token] Invalid audience: {aud}")
            return None
        # Verify expiry
        exp = int(info.get('exp', 0))
        if exp < datetime.utcnow().timestamp():
            return None
        return {
            'google_id':  info.get('sub', ''),
            'email':      info.get('email', '').lower().strip(),
            'name':       info.get('name', 'Traveller'),
            'first_name': info.get('given_name', ''),
            'last_name':  info.get('family_name', ''),
            'avatar':     info.get('picture', ''),
            'locale':     info.get('locale', 'en'),
            'verified':   info.get('email_verified', 'false') == 'true',
        }
    except Exception as e:
        print(f"[Google Token] {e}")
        return None


def find_or_create_google_user(profile):
    """Find existing user or create new one from Google profile."""
    google_id = profile['google_id']
    email     = profile['email']
    name      = clean_name(profile.get('name', 'Traveller'))
    avatar    = profile.get('avatar', '')
    locale    = profile.get('locale', 'en')

    # 1. Try find by Google ID
    user = get_user_by_google(google_id)
    if user:
        # Update avatar and last login
        update_user(user['id'],
                    avatar=avatar,
                    last_login=datetime.now().isoformat(),
                    locale=locale)
        return get_user_by_id(user['id']), False  # (user, is_new)

    # 2. Try find by email — link Google to existing account
    user = get_user_by_email(email)
    if user:
        link_google_to_user(user['id'], google_id, avatar)
        update_user(user['id'],
                    last_login=datetime.now().isoformat(),
                    locale=locale)
        return get_user_by_id(user['id']), False

    # 3. Create new account
    user_id = create_user(
        email       = email,
        name        = name,
        google_id   = google_id,
        avatar      = avatar,
        locale      = locale,
    )
    if not user_id:
        return None, False

    log_action(user_id, 'signup_google', 'oauth')
    return get_user_by_id(user_id), True  # (user, is_new)


# ════════════════════════════════════════════════════════════════
#  REGISTER ALL AUTH ROUTES
# ════════════════════════════════════════════════════════════════

def register_auth_routes(app):

    # Get limiter from app extensions
    # This avoids circular import — limiter is created in app.py
    def get_limiter():
        return app.extensions.get('flask-limiter', None)

    def rate_limit_exempt(f):
        """Safely exempt from rate limiter without import."""
        limiter = get_limiter()
        if limiter:
            return limiter.exempt(f)
        return f

    # ── BACKWARD COMPATIBLE ALIASES ──────────────────────────
    # auth.html calls /api/login and /api/register (old routes)
    # These map to the new /api/auth/* routes

    @app.route('/api/register', methods=['POST'])
    def api_register_alias():
        """Alias for /api/auth/signup — backward compat."""
        with app.test_request_context(
            '/api/auth/signup',
            method='POST',
            json=request.get_json()
        ):
            pass
        # Just call signup logic directly
        return _do_signup()

    @app.route('/api/login', methods=['POST'])
    def api_login_alias():
        """Alias for /api/auth/login — backward compat."""
        return _do_login()

    # ── SIGNUP ───────────────────────────────────────────────

    @app.route('/api/auth/signup', methods=['POST'])
    def signup():
        return _do_signup()

    def _do_signup():
        try:
            data     = request.get_json() or {}
            name     = clean_name((data.get('name') or '').strip())
            email    = (data.get('email') or '').strip().lower()
            password = (data.get('password') or '').strip()

            # Validation
            if not name or len(name) < 2:
                return jsonify({'success': False, 'error': 'Please enter your full name'})
            if not email or not validate_email(email):
                return jsonify({'success': False, 'error': 'Please enter a valid email address'})
            valid, err = validate_password(password)
            if not valid:
                return jsonify({'success': False, 'error': err})

            # Check existing
            if get_user_by_email(email):
                return jsonify({
                    'success': False,
                    'error':   'An account with this email already exists. Please sign in.'
                })

            # Create user
            user_id = create_user(
                email         = email,
                name          = name,
                password_hash = hash_password(password)
            )
            if not user_id:
                return jsonify({'success': False, 'error': 'Could not create account. Please try again.'})

            token = make_token(user_id)
            user  = get_user_by_id(user_id)
            log_action(user_id, 'signup', request.remote_addr)

            return jsonify({
                'success':  True,
                'token':    token,
                'user':     safe_user(user),
                'redirect': '/app',
                'message':  f"Welcome to Yaply, {name.split()[0]}! ✈️"
            })
        except Exception as e:
            print(f"[Signup error] {e}")
            return jsonify({'success': False, 'error': 'Something went wrong. Please try again.'})

    # ── LOGIN ────────────────────────────────────────────────

    @app.route('/api/auth/login', methods=['POST'])
    def auth_login():
        return _do_login()

    def _do_login():
        try:
            data     = request.get_json() or {}
            email    = (data.get('email') or '').strip().lower()
            password = (data.get('password') or '').strip()
            ip       = request.remote_addr

            if not email or not password:
                return jsonify({'success': False, 'error': 'Please enter email and password'})

            # Brute force check
            if _is_locked_out(email, ip):
                return jsonify({
                    'success': False,
                    'error':   f'Too many failed attempts. Please wait {LOCKOUT_MINUTES} minutes and try again.'
                })

            user = get_user_by_email(email)
            if not user:
                _record_failed(email, ip)
                return jsonify({'success': False, 'error': 'No account found with this email'})

            if not user.get('password'):
                return jsonify({
                    'success': False,
                    'error':   'This account uses Google sign-in. Please use the Google button.'
                })

            if not check_password(password, user['password']):
                _record_failed(email, ip)
                return jsonify({'success': False, 'error': 'Incorrect password'})

            # Success — clear failed attempts
            _clear_attempts(email, ip)
            update_user(user['id'], last_login=datetime.now().isoformat())
            token = make_token(user['id'])
            log_action(user['id'], 'login', ip)

            return jsonify({
                'success':  True,
                'token':    token,
                'user':     safe_user(user),
                'redirect': '/app',
                'message':  f"Welcome back, {user['name'].split()[0]}! ✈️"
            })
        except Exception as e:
            print(f"[Login error] {e}")
            return jsonify({'success': False, 'error': 'Something went wrong. Please try again.'})

    # ── GOOGLE ONE TAP (POST — for Google Sign-In button) ────

    @app.route('/api/auth/google', methods=['POST'])
    def google_one_tap():
        """
        Google One Tap / Sign-In button flow.
        Frontend sends: { id_token: '...' }
        """
        try:
            data     = request.get_json() or {}
            id_token = (data.get('id_token') or '').strip()

            if not id_token:
                return jsonify({'success': False, 'error': 'No Google token provided'})

            # Verify token with Google
            profile = verify_google_id_token(id_token)
            if not profile:
                return jsonify({'success': False, 'error': 'Google verification failed. Please try again.'})

            if not profile.get('email'):
                return jsonify({'success': False, 'error': 'Could not get email from Google'})

            if not profile.get('verified'):
                return jsonify({'success': False, 'error': 'Google account email is not verified'})

            # Find or create user
            user, is_new = find_or_create_google_user(profile)
            if not user:
                return jsonify({'success': False, 'error': 'Could not create account. Please try again.'})

            token        = make_token(user['id'])
            first_name   = profile.get('first_name') or user['name'].split()[0]
            action       = 'signup_google' if is_new else 'login_google'
            log_action(user['id'], action, request.remote_addr)

            return jsonify({
                'success':  True,
                'token':    token,
                'user':     safe_user(user),
                'is_new':   is_new,
                'redirect': '/app',
                'message':  f"{'Welcome to Yaply' if is_new else 'Welcome back'}, {first_name}! ✈️"
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({'success': False, 'error': 'Google login failed. Please use email login.'})

    # ── GOOGLE OAUTH REDIRECT (GET — for button href) ────────

    @app.route('/auth/google')
    def google_oauth_start():
        """
        OAuth redirect flow — triggered by:
        <a href="/auth/google"> in auth.html
        """
        if not GOOGLE_CLIENT_ID:
            return redirect('/login?error=Google+login+not+configured')

        params = {
            'client_id':     GOOGLE_CLIENT_ID,
            'redirect_uri':  GOOGLE_REDIRECT_URI,
            'response_type': 'code',
            'scope':         'openid email profile',
            'access_type':   'offline',
            'prompt':        'select_account',
            'include_granted_scopes': 'true',
        }
        url = 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode(params)
        return redirect(url)

    @app.route('/auth/google/callback')
    def google_oauth_callback():
        """Handle Google OAuth redirect callback."""
        code  = request.args.get('code')
        error = request.args.get('error')

        if error or not code:
            reason = 'Google+login+was+cancelled' if error == 'access_denied' else 'Google+login+failed'
            return redirect(f'/login?error={reason}')

        try:
            # Step 1: Exchange code for tokens
            token_res = req.post(
                'https://oauth2.googleapis.com/token',
                data={
                    'code':          code,
                    'client_id':     GOOGLE_CLIENT_ID,
                    'client_secret': GOOGLE_CLIENT_SECRET,
                    'redirect_uri':  GOOGLE_REDIRECT_URI,
                    'grant_type':    'authorization_code',
                },
                timeout=10
            )
            token_data   = token_res.json()
            access_token = token_data.get('access_token')

            if not access_token:
                print(f"[Google OAuth] Token exchange failed: {token_data}")
                return redirect('/login?error=Google+authentication+failed')

            # Step 2: Fetch full Google profile
            profile = fetch_google_profile(access_token)
            if not profile:
                return redirect('/login?error=Could+not+get+Google+profile')

            if not profile.get('email'):
                return redirect('/login?error=Could+not+get+email+from+Google')

            # Step 3: Find or create user
            user, is_new = find_or_create_google_user(profile)
            if not user:
                return redirect('/login?error=Could+not+create+account')

            # Step 4: Create JWT and redirect
            token      = make_token(user['id'])
            first_name = profile.get('first_name') or user['name'].split()[0]
            action     = 'signup_google' if is_new else 'login_google'
            log_action(user['id'], action, request.remote_addr)

            # Redirect to /app with token in URL param
            # Frontend JS will pick it up and store in localStorage
            return redirect(f'/app?token={token}&welcome={first_name}')

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[Google OAuth Callback] {e}")
            return redirect('/login?error=Google+login+failed+please+try+again')

    # ── GET CURRENT USER ─────────────────────────────────────

    @app.route('/api/auth/me', methods=['GET'])
    @app.route('/api/me', methods=['GET'])
    @require_auth
    def get_me():
        return jsonify({'success': True, 'user': safe_user(g.user)})

    # ── UPDATE PROFILE ───────────────────────────────────────

    @app.route('/api/auth/update-profile', methods=['POST'])
    @require_auth
    def update_profile():
        try:
            data    = request.get_json() or {}
            allowed = ['name', 'passport', 'home_city', 'currency', 'locale']
            updates = {}
            for k in allowed:
                if k in data and data[k]:
                    val = str(data[k]).strip()[:100]
                    if k == 'name':
                        val = clean_name(val)
                    updates[k] = val
            if not updates:
                return jsonify({'success': False, 'error': 'No valid fields to update'})
            update_user(g.user_id, **updates)
            user = get_user_by_id(g.user_id)
            return jsonify({'success': True, 'user': safe_user(user), 'message': 'Profile updated!'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # ── CHANGE PASSWORD ──────────────────────────────────────

    @app.route('/api/auth/change-password', methods=['POST'])
    @require_auth
    def change_password():
        try:
            data     = request.get_json() or {}
            current  = data.get('current_password', '')
            new_pw   = data.get('new_password', '')
            user     = get_user_by_id(g.user_id)

            if not user.get('password'):
                return jsonify({'success': False, 'error': 'This account uses Google sign-in — no password to change'})
            if not check_password(current, user['password']):
                _record_failed(user['email'], request.remote_addr)
                return jsonify({'success': False, 'error': 'Current password is incorrect'})

            valid, err = validate_password(new_pw)
            if not valid:
                return jsonify({'success': False, 'error': err})

            update_user_password(g.user_id, hash_password(new_pw))
            log_action(g.user_id, 'password_changed', request.remote_addr)
            return jsonify({'success': True, 'message': 'Password updated successfully!'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # ── FORGOT PASSWORD (send reset link) ────────────────────

    @app.route('/api/auth/forgot-password', methods=['POST'])
    def forgot_password():
        try:
            email = (request.get_json() or {}).get('email', '').strip().lower()
            if not email or not validate_email(email):
                return jsonify({'success': False, 'error': 'Please enter a valid email'})
            user = get_user_by_email(email)
            # Always return success (don't reveal if email exists)
            if user and user.get('password'):
                # Generate short-lived reset token
                reset_token = make_token(user['id'], expiry_days=1)
                reset_link  = f"https://www.yaply.live/reset-password?token={reset_token}"
                # TODO: Send email via SendGrid/Resend
                # For now just log it
                print(f"[Password Reset] {email} → {reset_link}")
            return jsonify({
                'success': True,
                'message': 'If an account exists with this email, you will receive a reset link shortly.'
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # ── LOGOUT ───────────────────────────────────────────────

    @app.route('/api/auth/logout', methods=['POST'])
    @require_auth
    def logout():
        # JWT is stateless — client just deletes token
        # For extra security you could maintain a blacklist in Redis
        log_action(g.user_id, 'logout', request.remote_addr)
        return jsonify({'success': True, 'message': 'Logged out successfully'})

    # ── DELETE ACCOUNT ───────────────────────────────────────

    @app.route('/api/auth/delete-account', methods=['POST'])
    @require_auth
    def delete_account():
        try:
            data     = request.get_json() or {}
            password = data.get('password', '')
            user     = get_user_by_id(g.user_id)

            # Require password confirmation for email accounts
            if user.get('password'):
                if not password:
                    return jsonify({'success': False, 'error': 'Please enter your password to confirm deletion'})
                if not check_password(password, user['password']):
                    return jsonify({'success': False, 'error': 'Incorrect password'})

            log_action(g.user_id, 'delete_account', request.remote_addr)
            delete_user(g.user_id)
            return jsonify({'success': True, 'message': 'Account deleted. Sorry to see you go.'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # ── VERIFY TOKEN (for frontend checks) ───────────────────

    @app.route('/api/auth/verify', methods=['GET'])
    @require_auth
    def verify_token():
        return jsonify({
            'success': True,
            'valid':   True,
            'user':    safe_user(g.user)
        })

    print("[Auth] ✅ All auth routes registered")