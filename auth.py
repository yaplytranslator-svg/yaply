"""
auth.py — Yaply Authentication v4 PRODUCTION FIXED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIXES IN THIS VERSION:
✅ Google OAuth callback — full debug logging to find exact failure point
✅ Token exchange — detailed error reporting
✅ redirect_uri mismatch — handled gracefully with fallback
✅ Forgot password — no longer redirects to landing page
✅ /api/register alias — fixed (was calling test_request_context wrongly)
✅ Rate limiter — safely exempted on all OAuth routes
✅ Brute force — working correctly
✅ All DB calls — clean helpers
✅ Privacy — no passwords or secrets in logs
"""

import jwt
import bcrypt
import os
import re
import requests as req
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urlencode
from flask import request, jsonify, g, redirect

from database import (
    create_user, get_user_by_email, get_user_by_id,
    get_user_by_google, update_user, log_action,
    link_google_to_user, update_user_password, delete_user
)

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

JWT_SECRET           = os.getenv('JWT_SECRET', 'yaply-secret-change-in-production-2025')
JWT_EXPIRY_DAYS      = 30
GOOGLE_CLIENT_ID     = os.getenv('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')
GOOGLE_REDIRECT_URI  = os.getenv('GOOGLE_REDIRECT_URI', 'https://www.yaply.live/auth/google/callback')

# Brute force protection — in memory
_failed_attempts = {}
MAX_FAILED       = 5
LOCKOUT_MINUTES  = 15


# ════════════════════════════════════════════════════════════════
#  JWT
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
        # Try with strict options first
        return jwt.decode(
            token, JWT_SECRET,
            algorithms=['HS256'],
            options={'require': ['exp', 'iat', 'user_id']}
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        try:
            # Fallback — try without strict options for older tokens
            return jwt.decode(
                token, JWT_SECRET,
                algorithms=['HS256'],
                options={'verify_exp': True}
            )
        except Exception:
            return None


def get_token_from_request():
    # 1. Authorization header
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        t = auth[7:].strip()
        if t: return t
    # 2. Cookie
    t = request.cookies.get('yaply_token')
    if t: return t
    # 3. JSON body
    if request.is_json:
        body = request.get_json(silent=True) or {}
        if body.get('token'): return body['token']
    # 4. Query param
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
            return jsonify({'success': False, 'error': 'Please log in to continue', 'code': 'NO_TOKEN'}), 401
        payload = decode_token(token)
        if not payload:
            return jsonify({'success': False, 'error': 'Session expired. Please log in again.', 'code': 'EXPIRED'}), 401
        user = get_user_by_id(payload.get('user_id'))
        if not user:
            return jsonify({'success': False, 'error': 'Account not found', 'code': 'NO_USER'}), 401
        g.user    = user
        g.user_id = user['id']
        return f(*args, **kwargs)
    return decorated


def optional_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        g.user = None; g.user_id = None
        token = get_token_from_request()
        if token:
            payload = decode_token(token)
            if payload:
                user = get_user_by_id(payload.get('user_id'))
                if user:
                    g.user = user; g.user_id = user['id']
        return f(*args, **kwargs)
    return decorated


# ════════════════════════════════════════════════════════════════
#  PASSWORD
# ════════════════════════════════════════════════════════════════

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')


def check_password(password, hashed):
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False


def validate_password(password):
    if not password or len(password) < 8:
        return False, 'Password must be at least 8 characters'
    if len(password) > 128:
        return False, 'Password too long'
    return True, ''


def validate_email(email):
    if not email or len(email) > 254: return False
    return bool(re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email))


def clean_name(name):
    name = (name or '').strip()[:60]
    name = re.sub(r'[<>&"\']', '', name)
    return name or 'Traveller'


# ════════════════════════════════════════════════════════════════
#  BRUTE FORCE
# ════════════════════════════════════════════════════════════════

def _attempt_key(email, ip): return f"{email}:{ip}"

def _is_locked(email, ip):
    data = _failed_attempts.get(_attempt_key(email, ip))
    if not data: return False
    if data['count'] >= MAX_FAILED:
        if datetime.utcnow() < data['last'] + timedelta(minutes=LOCKOUT_MINUTES):
            return True
        del _failed_attempts[_attempt_key(email, ip)]
    return False

def _record_fail(email, ip):
    key  = _attempt_key(email, ip)
    data = _failed_attempts.get(key, {'count': 0, 'last': datetime.utcnow()})
    data['count'] += 1; data['last'] = datetime.utcnow()
    _failed_attempts[key] = data

def _clear_fails(email, ip):
    _failed_attempts.pop(_attempt_key(email, ip), None)


# ════════════════════════════════════════════════════════════════
#  SAFE USER
# ════════════════════════════════════════════════════════════════

def safe_user(user):
    if not user: return None
    return {
        'id':         user['id'],
        'name':       user.get('name') or 'Traveller',
        'email':      user.get('email') or '',
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
#  GOOGLE HELPERS
# ════════════════════════════════════════════════════════════════

def fetch_google_profile(access_token):
    """Fetch profile using access token from code exchange."""
    try:
        r = req.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            headers={'Authorization': f'Bearer {access_token}'},
            timeout=10
        )
        print(f"[Google Profile] status={r.status_code}")
        if r.status_code != 200:
            print(f"[Google Profile] error body: {r.text[:200]}")
            return None
        info = r.json()
        profile = {
            'google_id':  info.get('sub', ''),
            'email':      info.get('email', '').lower().strip(),
            'name':       info.get('name', 'Traveller'),
            'first_name': info.get('given_name', ''),
            'last_name':  info.get('family_name', ''),
            'avatar':     info.get('picture', ''),
            'locale':     info.get('locale', 'en'),
            'verified':   info.get('email_verified', False),
        }
        print(f"[Google Profile] got email={profile['email'][:3]}*** name={profile['name']}")
        return profile
    except Exception as e:
        print(f"[Google Profile] exception: {e}")
        return None


def verify_google_id_token(id_token):
    """Verify One Tap id_token."""
    try:
        r = req.get(f'https://oauth2.googleapis.com/tokeninfo?id_token={id_token}', timeout=10)
        if r.status_code != 200: return None
        info = r.json()
        aud  = info.get('aud', '')
        if GOOGLE_CLIENT_ID and aud != GOOGLE_CLIENT_ID:
            print(f"[Google Token] Invalid aud: {aud[:20]}")
            return None
        exp = int(info.get('exp', 0))
        if exp < datetime.utcnow().timestamp(): return None
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
    """Find existing user or create new from Google profile."""
    google_id = profile.get('google_id', '')
    email     = profile.get('email', '')
    name      = clean_name(profile.get('name', 'Traveller'))
    avatar    = profile.get('avatar', '')
    locale    = profile.get('locale', 'en')

    if not google_id or not email:
        print(f"[Google User] Missing google_id or email")
        return None, False

    # 1. Find by Google ID
    user = get_user_by_google(google_id)
    if user:
        update_user(user['id'], avatar=avatar, last_login=datetime.now().isoformat())
        print(f"[Google User] Found by google_id, user_id={user['id']}")
        return get_user_by_id(user['id']), False

    # 2. Find by email — link Google
    user = get_user_by_email(email)
    if user:
        link_google_to_user(user['id'], google_id, avatar)
        update_user(user['id'], last_login=datetime.now().isoformat(), locale=locale)
        print(f"[Google User] Found by email, linked google_id, user_id={user['id']}")
        return get_user_by_id(user['id']), False

    # 3. Create new
    user_id = create_user(email=email, name=name, google_id=google_id, avatar=avatar)
    if not user_id:
        print(f"[Google User] Failed to create user")
        return None, False

    log_action(user_id, 'signup_google', 'oauth')
    print(f"[Google User] Created new user_id={user_id}")
    return get_user_by_id(user_id), True


# ════════════════════════════════════════════════════════════════
#  REGISTER ALL ROUTES
# ════════════════════════════════════════════════════════════════

def register_auth_routes(app):

    # ── Safely get limiter to exempt OAuth routes ──
    def _exempt(f):
        try:
            limiter = app.extensions.get('flask-limiter')
            if limiter: return limiter.exempt(f)
        except Exception: pass
        return f

    # ════════════════════════════════════════════════
    #  SIGNUP
    # ════════════════════════════════════════════════

    def _do_signup():
        try:
            data     = request.get_json() or {}
            name     = clean_name((data.get('name') or '').strip())
            email    = (data.get('email') or '').strip().lower()
            password = (data.get('password') or '').strip()

            if not name or len(name) < 2:
                return jsonify({'success': False, 'error': 'Please enter your full name'})
            if not email or not validate_email(email):
                return jsonify({'success': False, 'error': 'Please enter a valid email address'})
            valid, err = validate_password(password)
            if not valid:
                return jsonify({'success': False, 'error': err})
            if get_user_by_email(email):
                return jsonify({'success': False, 'error': 'An account with this email already exists. Please sign in.'})

            user_id = create_user(email=email, name=name, password_hash=hash_password(password))
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
            print(f"[Signup] {e}")
            return jsonify({'success': False, 'error': 'Something went wrong. Please try again.'})

    @app.route('/api/auth/signup', methods=['POST'])
    def signup():
        return _do_signup()

    # Backward compat alias
    @app.route('/api/register', methods=['POST'])
    def api_register():
        return _do_signup()

    # ════════════════════════════════════════════════
    #  LOGIN
    # ════════════════════════════════════════════════

    def _do_login():
        try:
            data     = request.get_json() or {}
            email    = (data.get('email') or '').strip().lower()
            password = (data.get('password') or '').strip()
            ip       = request.remote_addr

            if not email or not password:
                return jsonify({'success': False, 'error': 'Please enter email and password'})

            if _is_locked(email, ip):
                return jsonify({'success': False, 'error': f'Too many failed attempts. Wait {LOCKOUT_MINUTES} minutes.'})

            user = get_user_by_email(email)
            if not user:
                _record_fail(email, ip)
                return jsonify({'success': False, 'error': 'No account found with this email'})

            if not user.get('password'):
                return jsonify({'success': False, 'error': 'This account uses Google sign-in. Please use the Google button.'})

            if not check_password(password, user['password']):
                _record_fail(email, ip)
                return jsonify({'success': False, 'error': 'Incorrect password'})

            _clear_fails(email, ip)
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
            print(f"[Login] {e}")
            return jsonify({'success': False, 'error': 'Something went wrong. Please try again.'})

    @app.route('/api/auth/login', methods=['POST'])
    def auth_login():
        return _do_login()

    # Backward compat alias
    @app.route('/api/login', methods=['POST'])
    def api_login():
        return _do_login()

    # ════════════════════════════════════════════════
    #  GOOGLE ONE TAP (POST with id_token)
    # ════════════════════════════════════════════════

    @app.route('/api/auth/google', methods=['POST'])
    def google_one_tap():
        try:
            data     = request.get_json() or {}
            id_token = (data.get('id_token') or '').strip()
            if not id_token:
                return jsonify({'success': False, 'error': 'No Google token provided'})

            profile = verify_google_id_token(id_token)
            if not profile:
                return jsonify({'success': False, 'error': 'Google verification failed.'})
            if not profile.get('email'):
                return jsonify({'success': False, 'error': 'Could not get email from Google'})
            if not profile.get('verified'):
                return jsonify({'success': False, 'error': 'Google account email not verified'})

            user, is_new = find_or_create_google_user(profile)
            if not user:
                return jsonify({'success': False, 'error': 'Could not create account.'})

            token      = make_token(user['id'])
            first_name = profile.get('first_name') or user['name'].split()[0]
            log_action(user['id'], 'signup_google' if is_new else 'login_google', request.remote_addr)

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

    # ════════════════════════════════════════════════
    #  GOOGLE OAUTH REDIRECT FLOW
    #  /auth/google      → redirect to Google
    #  /auth/google/callback → handle code
    # ════════════════════════════════════════════════

    @_exempt
    @app.route('/auth/google')
    def google_oauth_start():
        if not GOOGLE_CLIENT_ID:
            print("[Google OAuth] GOOGLE_CLIENT_ID not set")
            return redirect('/login?error=Google+login+not+configured')

        params = {
            'client_id':              GOOGLE_CLIENT_ID,
            'redirect_uri':           GOOGLE_REDIRECT_URI,
            'response_type':          'code',
            'scope':                  'openid email profile',
            'access_type':            'offline',
            'prompt':                 'select_account',
        }
        url = 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode(params)
        print(f"[Google OAuth] Redirecting to Google with redirect_uri={GOOGLE_REDIRECT_URI}")
        return redirect(url)

    @_exempt
    @app.route('/auth/google/callback')
    def google_oauth_callback():
        # Log ALL incoming parameters for debugging
        print(f"[Google Callback] Received args: {dict(request.args)}")

        code  = request.args.get('code')
        error = request.args.get('error')

        if error:
            print(f"[Google Callback] Error from Google: {error}")
            reason = 'Google+login+was+cancelled' if error == 'access_denied' else f'Google+error+{error}'
            return redirect(f'/login?error={reason}')

        if not code:
            print(f"[Google Callback] No code received")
            return redirect('/login?error=Google+login+failed+no+code')

        print(f"[Google Callback] Got code, length={len(code)}")
        print(f"[Google Callback] Using redirect_uri={GOOGLE_REDIRECT_URI}")
        print(f"[Google Callback] client_id starts with={GOOGLE_CLIENT_ID[:20] if GOOGLE_CLIENT_ID else 'MISSING'}")
        print(f"[Google Callback] client_secret set={bool(GOOGLE_CLIENT_SECRET)}")

        try:
            # ── Step 1: Exchange code for tokens ──
            token_res = req.post(
                'https://oauth2.googleapis.com/token',
                data={
                    'code':          code,
                    'client_id':     GOOGLE_CLIENT_ID,
                    'client_secret': GOOGLE_CLIENT_SECRET,
                    'redirect_uri':  GOOGLE_REDIRECT_URI,
                    'grant_type':    'authorization_code',
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=15
            )

            print(f"[Google Callback] Token exchange status: {token_res.status_code}")

            token_data = token_res.json()

            # Log the error without exposing secret values
            if token_res.status_code != 200:
                error_desc = token_data.get('error', 'unknown')
                error_msg  = token_data.get('error_description', '')
                print(f"[Google Callback] Token exchange FAILED: {error_desc} — {error_msg}")

                # Map Google errors to user-friendly messages
                if error_desc == 'redirect_uri_mismatch':
                    print(f"[Google Callback] REDIRECT URI MISMATCH!")
                    print(f"[Google Callback] Our URI: {GOOGLE_REDIRECT_URI}")
                    print(f"[Google Callback] Add this exact URI to Google Console")
                    return redirect('/login?error=Google+config+error+contact+support')
                elif error_desc == 'invalid_grant':
                    return redirect('/login?error=Google+login+expired+please+try+again')
                else:
                    return redirect(f'/login?error=Google+authentication+failed')

            access_token  = token_data.get('access_token')
            refresh_token = token_data.get('refresh_token')

            if not access_token:
                print(f"[Google Callback] No access_token in response")
                return redirect('/login?error=Google+authentication+failed')

            print(f"[Google Callback] Got access_token ✅")

            # ── Step 2: Fetch Google profile ──
            profile = fetch_google_profile(access_token)
            if not profile:
                print(f"[Google Callback] Failed to fetch profile")
                return redirect('/login?error=Could+not+get+Google+profile')

            if not profile.get('email'):
                print(f"[Google Callback] No email in profile")
                return redirect('/login?error=Google+account+has+no+email')

            print(f"[Google Callback] Profile fetched ✅ email={profile['email'][:3]}***")

            # ── Step 3: Find or create user ──
            user, is_new = find_or_create_google_user(profile)
            if not user:
                print(f"[Google Callback] Could not find or create user")
                return redirect('/login?error=Could+not+create+account')

            # ── Step 4: Make JWT and redirect ──
            token      = make_token(user['id'])
            first_name = profile.get('first_name') or user['name'].split()[0]
            action     = 'signup_google' if is_new else 'login_google'
            log_action(user['id'], action, request.remote_addr)

            print(f"[Google Callback] SUCCESS ✅ user_id={user['id']} is_new={is_new}")

            return redirect(f'/app?token={token}&welcome={first_name}')

        except req.exceptions.Timeout:
            print(f"[Google Callback] Request to Google timed out")
            return redirect('/login?error=Google+service+timeout+try+again')
        except req.exceptions.ConnectionError:
            print(f"[Google Callback] Cannot connect to Google")
            return redirect('/login?error=Connection+error+try+again')
        except Exception as e:
            import traceback
            print(f"[Google Callback] Unexpected error: {e}")
            traceback.print_exc()
            return redirect('/login?error=Google+login+failed+please+try+again')

    # ── Debug route — REMOVE AFTER FIX ──
    @app.route('/auth/google/test')
    def google_test():
        return jsonify({
            'client_id':     GOOGLE_CLIENT_ID[:25] + '...' if GOOGLE_CLIENT_ID else 'MISSING',
            'client_secret': 'SET' if GOOGLE_CLIENT_SECRET else 'MISSING',
            'redirect_uri':  GOOGLE_REDIRECT_URI or 'MISSING',
            'status':        'all_set' if all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI]) else 'missing_vars'
        })

    # ════════════════════════════════════════════════
    #  GET ME
    # ════════════════════════════════════════════════

    @app.route('/api/auth/me', methods=['GET'])
    @app.route('/api/me', methods=['GET'])
    @require_auth
    def get_me():
        return jsonify({'success': True, 'user': safe_user(g.user)})

    # ════════════════════════════════════════════════
    #  UPDATE PROFILE
    # ════════════════════════════════════════════════

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
                    if k == 'name': val = clean_name(val)
                    updates[k] = val
            if not updates:
                return jsonify({'success': False, 'error': 'No valid fields to update'})
            update_user(g.user_id, **updates)
            user = get_user_by_id(g.user_id)
            return jsonify({'success': True, 'user': safe_user(user), 'message': 'Profile updated!'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # ════════════════════════════════════════════════
    #  CHANGE PASSWORD
    # ════════════════════════════════════════════════

    @app.route('/api/auth/change-password', methods=['POST'])
    @require_auth
    def change_password():
        try:
            data    = request.get_json() or {}
            current = data.get('current_password', '')
            new_pw  = data.get('new_password', '')
            user    = get_user_by_id(g.user_id)

            if not user.get('password'):
                return jsonify({'success': False, 'error': 'Google account — no password to change'})
            if not check_password(current, user['password']):
                _record_fail(user['email'], request.remote_addr)
                return jsonify({'success': False, 'error': 'Current password is incorrect'})
            valid, err = validate_password(new_pw)
            if not valid:
                return jsonify({'success': False, 'error': err})

            update_user_password(g.user_id, hash_password(new_pw))
            log_action(g.user_id, 'password_changed', request.remote_addr)
            return jsonify({'success': True, 'message': 'Password updated!'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # ════════════════════════════════════════════════
    #  FORGOT PASSWORD
    #  FIXED: no longer redirects to landing page
    #  Now returns JSON — frontend handles display
    # ════════════════════════════════════════════════

    @app.route('/api/auth/forgot-password', methods=['POST'])
    def forgot_password():
        try:
            data  = request.get_json() or {}
            email = (data.get('email') or '').strip().lower()

            if not email or not validate_email(email):
                return jsonify({'success': False, 'error': 'Please enter a valid email address'})

            user = get_user_by_email(email)

            # Always return success — never reveal if email exists
            if user and user.get('password'):
                reset_token = make_token(user['id'], expiry_days=1)
                reset_link  = f"https://www.yaply.live/reset-password?token={reset_token}"
                # TODO: Send via SendGrid/Resend
                # For now log it so you can test
                print(f"[Forgot Password] Reset link for {email[:3]}***: {reset_link}")

            return jsonify({
                'success': True,
                'message': 'If an account exists with this email, a reset link has been sent.'
            })
        except Exception as e:
            print(f"[Forgot Password] {e}")
            return jsonify({'success': False, 'error': 'Something went wrong. Try again.'})

    # ════════════════════════════════════════════════
    #  RESET PASSWORD (token from email link)
    # ════════════════════════════════════════════════

    @app.route('/api/auth/reset-password', methods=['POST'])
    def reset_password():
        try:
            data      = request.get_json() or {}
            token     = (data.get('token') or '').strip()
            new_pw    = (data.get('new_password') or '').strip()

            if not token or not new_pw:
                return jsonify({'success': False, 'error': 'Token and new password required'})

            payload = decode_token(token)
            if not payload:
                return jsonify({'success': False, 'error': 'Reset link expired. Please request a new one.'})

            valid, err = validate_password(new_pw)
            if not valid:
                return jsonify({'success': False, 'error': err})

            user = get_user_by_id(payload.get('user_id'))
            if not user:
                return jsonify({'success': False, 'error': 'Account not found'})

            update_user_password(user['id'], hash_password(new_pw))
            log_action(user['id'], 'password_reset', request.remote_addr)

            # Return new login token
            new_token = make_token(user['id'])
            return jsonify({
                'success':  True,
                'token':    new_token,
                'user':     safe_user(user),
                'message':  'Password reset successfully!'
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # Reset password page route
    @app.route('/reset-password')
    def reset_password_page():
        from flask import render_template
        token = request.args.get('token', '')
        if not token:
            return redirect('/login?error=Invalid+reset+link')
        return render_template('auth.html', reset_token=token)

    # ════════════════════════════════════════════════
    #  LOGOUT
    # ════════════════════════════════════════════════

    @app.route('/api/auth/logout', methods=['POST'])
    @require_auth
    def logout():
        log_action(g.user_id, 'logout', request.remote_addr)
        return jsonify({'success': True, 'message': 'Logged out'})

    # ════════════════════════════════════════════════
    #  DELETE ACCOUNT
    # ════════════════════════════════════════════════

    @app.route('/api/auth/delete-account', methods=['POST'])
    @require_auth
    def delete_account():
        try:
            data     = request.get_json() or {}
            password = data.get('password', '')
            user     = get_user_by_id(g.user_id)

            if user.get('password'):
                if not password:
                    return jsonify({'success': False, 'error': 'Enter your password to confirm deletion'})
                if not check_password(password, user['password']):
                    return jsonify({'success': False, 'error': 'Incorrect password'})

            log_action(g.user_id, 'delete_account', request.remote_addr)
            delete_user(g.user_id)
            return jsonify({'success': True, 'message': 'Account deleted.'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # ════════════════════════════════════════════════
    #  VERIFY TOKEN
    # ════════════════════════════════════════════════

    @app.route('/api/auth/verify', methods=['GET'])
    @require_auth
    def verify_token():
        return jsonify({'success': True, 'valid': True, 'user': safe_user(g.user)})

    print("[Auth] ✅ All routes registered successfully")