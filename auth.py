"""
auth.py — Yaply Authentication v2 FIXED
- Email + Password signup/login
- Google OAuth login
- JWT tokens
- All DB calls use clean helper functions
"""
import jwt, bcrypt, os, requests
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, g
from database import (
    create_user, get_user_by_email, get_user_by_id,
    get_user_by_google, update_user, log_action,
    link_google_to_user, update_user_password, delete_user
)

JWT_SECRET     = os.getenv('JWT_SECRET', 'yaply-secret-change-in-production-2025')
JWT_EXPIRY_DAYS = 30
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')

# ── JWT ──
def make_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(days=JWT_EXPIRY_DAYS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def decode_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except jwt.ExpiredSignatureError: return None
    except jwt.InvalidTokenError:     return None

def get_token_from_request():
    # 1. Authorization header
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        return auth[7:]
    # 2. Cookie
    t = request.cookies.get('yaply_token')
    if t: return t
    # 3. JSON body
    if request.is_json:
        return (request.get_json(silent=True) or {}).get('token')
    # 4. Query param (for WebSocket upgrade)
    return request.args.get('token')

# ── DECORATORS ──
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()
        if not token:
            return jsonify({'success':False,'error':'Please log in to continue','code':'NO_TOKEN'}), 401
        payload = decode_token(token)
        if not payload:
            return jsonify({'success':False,'error':'Session expired. Please log in again.','code':'EXPIRED'}), 401
        user = get_user_by_id(payload['user_id'])
        if not user:
            return jsonify({'success':False,'error':'Account not found','code':'NO_USER'}), 401
        g.user = user
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
                user = get_user_by_id(payload['user_id'])
                if user:
                    g.user = user
                    g.user_id = user['id']
        return f(*args, **kwargs)
    return decorated

# ── PASSWORD ──
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except: return False

def validate_password(password):
    if len(password) < 8:
        return False, 'Password must be at least 8 characters'
    return True, ''

def validate_email(email):
    import re
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

# ── SAFE USER (no password) ──
def safe_user(user):
    if not user: return None
    return {
        'id':        user['id'],
        'name':      user['name'],
        'email':     user['email'],
        'avatar':    user.get('avatar') or '',
        'passport':  user.get('passport') or 'India',
        'home_city': user.get('home_city') or '',
        'currency':  user.get('currency') or 'INR',
        'is_pro':    bool(user.get('is_pro')),
        'created_at':user.get('created_at') or '',
    }

# ── REGISTER ALL AUTH ROUTES ──
def register_auth_routes(app):

    @app.route('/api/auth/signup', methods=['POST'])
    def signup():
        try:
            data = request.get_json() or {}
            name     = (data.get('name') or '').strip()
            email    = (data.get('email') or '').strip().lower()
            password = (data.get('password') or '').strip()

            if not name or len(name) < 2:
                return jsonify({'success':False,'error':'Please enter your full name'})
            if not email or not validate_email(email):
                return jsonify({'success':False,'error':'Please enter a valid email address'})
            valid, err = validate_password(password)
            if not valid:
                return jsonify({'success':False,'error':err})
            if get_user_by_email(email):
                return jsonify({'success':False,'error':'An account with this email already exists. Please log in.'})

            user_id = create_user(email, name, password_hash=hash_password(password))
            if not user_id:
                return jsonify({'success':False,'error':'Could not create account. Please try again.'})

            token = make_token(user_id)
            user  = get_user_by_id(user_id)
            log_action(user_id, 'signup', request.remote_addr)
            return jsonify({'success':True,'token':token,'user':safe_user(user),'message':f'Welcome to Yaply, {name}! ✈️'})
        except Exception as e:
            print(f"[Signup error] {e}")
            return jsonify({'success':False,'error':'Something went wrong. Please try again.'})

    @app.route('/api/auth/login', methods=['POST'])
    def login():
        try:
            data = request.get_json() or {}
            email    = (data.get('email') or '').strip().lower()
            password = (data.get('password') or '').strip()

            if not email or not password:
                return jsonify({'success':False,'error':'Please enter email and password'})

            user = get_user_by_email(email)
            if not user:
                return jsonify({'success':False,'error':'No account found with this email'})
            if not user.get('password'):
                return jsonify({'success':False,'error':'This account uses Google sign-in. Please use the Google button.'})
            if not check_password(password, user['password']):
                return jsonify({'success':False,'error':'Incorrect password'})

            update_user(user['id'], last_login=datetime.now().isoformat())
            token = make_token(user['id'])
            log_action(user['id'], 'login', request.remote_addr)
            return jsonify({'success':True,'token':token,'user':safe_user(user),'message':f'Welcome back, {user["name"].split()[0]}! ✈️'})
        except Exception as e:
            print(f"[Login error] {e}")
            return jsonify({'success':False,'error':'Something went wrong. Please try again.'})

    @app.route('/api/auth/google', methods=['POST'])
    def google_login():
        try:
            data     = request.get_json() or {}
            id_token = data.get('id_token', '')
            if not id_token:
                return jsonify({'success':False,'error':'No Google token provided'})

            r = requests.get(f'https://oauth2.googleapis.com/tokeninfo?id_token={id_token}', timeout=10)
            if r.status_code != 200:
                return jsonify({'success':False,'error':'Google verification failed'})

            info = r.json()
            if GOOGLE_CLIENT_ID and info.get('aud') != GOOGLE_CLIENT_ID:
                return jsonify({'success':False,'error':'Invalid Google client'})

            google_id = info.get('sub')
            email     = info.get('email','').lower()
            name      = info.get('name','Traveller')
            avatar    = info.get('picture','')

            if not google_id or not email:
                return jsonify({'success':False,'error':'Could not get profile from Google'})

            user = get_user_by_google(google_id)
            if not user:
                user = get_user_by_email(email)
                if user:
                    link_google_to_user(user['id'], google_id, avatar)
                    user = get_user_by_id(user['id'])
                else:
                    user_id = create_user(email, name, google_id=google_id, avatar=avatar)
                    if not user_id:
                        return jsonify({'success':False,'error':'Could not create account'})
                    user = get_user_by_id(user_id)
                    log_action(user['id'], 'signup_google', request.remote_addr)

            update_user(user['id'], last_login=datetime.now().isoformat(), avatar=avatar)
            token = make_token(user['id'])
            log_action(user['id'], 'login_google', request.remote_addr)
            return jsonify({'success':True,'token':token,'user':safe_user(user),'message':f'Welcome, {name.split()[0]}! ✈️'})
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({'success':False,'error':'Google login failed. Please use email login.'})

    @app.route('/api/auth/me', methods=['GET'])
    @require_auth
    def get_me():
        return jsonify({'success':True,'user':safe_user(g.user)})

    @app.route('/api/auth/update-profile', methods=['POST'])
    @require_auth
    def update_profile():
        try:
            data = request.get_json() or {}
            allowed = ['name','passport','home_city','currency']
            updates = {k:data[k] for k in allowed if k in data and data[k]}
            update_user(g.user_id, **updates)
            user = get_user_by_id(g.user_id)
            return jsonify({'success':True,'user':safe_user(user)})
        except Exception as e:
            return jsonify({'success':False,'error':str(e)})

    @app.route('/api/auth/change-password', methods=['POST'])
    @require_auth
    def change_password():
        try:
            data    = request.get_json() or {}
            current = data.get('current_password','')
            new_pw  = data.get('new_password','')
            user    = get_user_by_id(g.user_id)
            if not user.get('password'):
                return jsonify({'success':False,'error':'Google account — no password to change'})
            if not check_password(current, user['password']):
                return jsonify({'success':False,'error':'Current password is incorrect'})
            valid, err = validate_password(new_pw)
            if not valid:
                return jsonify({'success':False,'error':err})
            update_user_password(g.user_id, hash_password(new_pw))
            return jsonify({'success':True,'message':'Password updated!'})
        except Exception as e:
            return jsonify({'success':False,'error':str(e)})

    @app.route('/api/auth/delete-account', methods=['POST'])
    @require_auth
    def delete_account():
        try:
            delete_user(g.user_id)
            return jsonify({'success':True,'message':'Account deleted'})
        except Exception as e:
            return jsonify({'success':False,'error':str(e)})