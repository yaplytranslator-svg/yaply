"""
app.py — Yaply COMPLETE Production App v5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Supabase PostgreSQL
✅ check_feature_limit as proper decorator
✅ All limits enforced (free + pro)
✅ /api/me/status — full plan + usage
✅ Post-signup flow: signup → onboarding → pricing → app
✅ All AI features gated
✅ Admin dashboard
✅ Payment + promo system
"""

# ── IMPORTS ──────────────────────────────────────────────────
from groups import groups_bp, init_groups_db, register_socketio_events
import os, io, base64, json, wave, struct, threading, time
import requests as req
from dotenv import load_dotenv
from functools import wraps
load_dotenv()

# ── FLASK ─────────────────────────────────────────────────────
from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, make_response, g, Response
)
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET', 'yaply-secret-2025-change-me')
CORS(app, supports_credentials=True)

# ── BLUEPRINTS ────────────────────────────────────────────────
app.register_blueprint(groups_bp)

# ── SOCKETIO + SOCK ───────────────────────────────────────────
from flask_socketio import SocketIO
from flask_sock import Sock

sock = Sock(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    ping_timeout=60,
    ping_interval=25,
)
register_socketio_events(socketio)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

# ── RATE LIMITER ──────────────────────────────────────────────
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "200 per hour"],
    storage_uri="memory://"
)

# ── DATABASE ──────────────────────────────────────────────────
from database import (
    init_db, log_action, get_user_plan, get_user_usage,
    create_payment, confirm_payment,
    admin_get_stats, admin_get_users, get_user_by_email,
    get_promo_code, check_promo_redeemed, redeem_promo,
    query_all, query_one, execute,
    activate_pro, complete_onboarding, FREE_LIMITS, PRO_LIMITS,
    save_trip, get_trips, get_trip, update_trip, delete_trip,
    save_place, get_places, delete_place,
    add_expense, get_expenses, delete_expense,
    save_journal, get_journal, get_user_stats, get_user_by_id, update_user,
    get_diary_trips, get_diary_trip, create_diary_trip, get_diary_entries,
    create_diary_entry, delete_diary_entry, toggle_diary_favorite, get_diary_stats
)
from auth import (
    register_auth_routes, require_auth, decode_token,
    get_token_from_request, optional_auth, safe_user
)

init_db()
register_auth_routes(app)
init_groups_db()

# ── AI CLIENTS ────────────────────────────────────────────────
from groq import Groq
import edge_tts, asyncio

_groq_key = os.getenv("GROQ_API_KEY")
if not _groq_key:
    raise RuntimeError("GROQ_API_KEY not set — app cannot start")
groq_client = Groq(api_key=_groq_key)

SCOUT     = "meta-llama/llama-4-scout-17b-16e-instruct"
MAVERICK  = "meta-llama/llama-4-maverick-17b-128e-instruct"
MODEL_70B = "llama-3.3-70b-versatile"

try:
    import deepl
    deepl_client = deepl.Translator(os.getenv("DEEPL_API_KEY", ""))
except Exception:
    deepl_client = None

# ── ENV KEYS ──────────────────────────────────────────────────
WEATHER_KEY          = os.getenv("OPENWEATHER_API_KEY", "")
EXCHANGE_KEY         = os.getenv("EXCHANGE_API_KEY", "")
GOOGLE_VISION_KEY    = os.getenv("GOOGLE_VISION_API_KEY", "")
UNSPLASH_KEY         = os.getenv("UNSPLASH_ACCESS_KEY", "")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "https://www.yaply.live/auth/google/callback")

# ── RAZORPAY ──────────────────────────────────────────────────
import razorpay
rzp_client = razorpay.Client(
    auth=(os.getenv('RAZORPAY_KEY_ID'), os.getenv('RAZORPAY_KEY_SECRET'))
)

# ── SECURITY HEADERS ──────────────────────────────────────────
@app.after_request
def add_security_headers(r):
    r.headers['X-Content-Type-Options']  = 'nosniff'
    r.headers['X-Frame-Options']         = 'DENY'
    r.headers['X-XSS-Protection']        = '1; mode=block'
    r.headers['Referrer-Policy']         = 'strict-origin-when-cross-origin'
    r.headers['Permissions-Policy']      = 'geolocation=(self), microphone=(self), camera=(self)'
    return r


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def clean(text, max_len=500):
    if not text: return ""
    return str(text).strip()[:max_len]

def validate(data, required_fields, max_len=500):
    if not data or not isinstance(data, dict):
        return False, "Invalid request data"
    for field in required_fields:
        if field not in data:
            return False, f"Missing field: {field}"
        val = data[field]
        if isinstance(val, str):
            if len(val.strip()) == 0:
                return False, f"{field} cannot be empty"
            if len(val) > max_len:
                return False, f"{field} too long (max {max_len} chars)"
    return True, ""


# ════════════════════════════════════════════════════════════════
#  FEATURE LIMIT SYSTEM
# ════════════════════════════════════════════════════════════════

# Maps feature key → (db_column, limit_key, display_label)
FEATURE_MAP = {
    'plan':        ('plans_used_month',      'plans_month',        'AI Trip Plans'),
    'multicity':   ('multicity_used_month',  'multicity_month',    'Multi-City Planning'),
    'translation': ('translations_today',    'translations_day',   'Live Translation'),
    'voice':       ('voice_today',           'voice_day',          'Voice Translation'),
    'identify':    ('identify_today',        'identify_photo_day', 'Place Finder'),
    'tool':        ('tools_today',           'tools_day',          'Safety Tools'),
    'ai_story':    ('ai_story_used',         'ai_story_month',     'AI Travel Story'),
    'journal':     ('journal_used_month',    'ai_journal_month',   'AI Trip Journal'),
}

def check_feature_limit(feature_key):
    """
    Decorator — use on any AI route to enforce plan limits.
    Usage: @check_feature_limit('plan')
    Returns 429 with limit_reached:true if blocked.
    Does NOT increment — increment manually after success.
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if feature_key not in FEATURE_MAP:
                return f(*args, **kwargs)

            col, limit_key, label = FEATURE_MAP[feature_key]
            plan   = get_user_plan(g.user_id)
            limits = PRO_LIMITS if plan == 'pro' else FREE_LIMITS
            limit  = limits.get(limit_key, 0)

            # Pro-only feature — free user blocked
            if limit == 0:
                return jsonify({
                    'success':        False,
                    'limit_reached':  True,
                    'is_pro':         False,
                    'feature':        feature_key,
                    'feature_label':  label,
                    'used':           0,
                    'limit':          0,
                    'error':          f'{label} is a Pro-only feature. Upgrade to unlock.',
                    'upgrade_prompt': True,
                    'upgrade_url':    '/pricing',
                }), 429

            # Check current usage
            row  = query_one(f'SELECT {col} FROM users WHERE id=%s', (g.user_id,))
            used = (row or {}).get(col, 0) or 0

            if used >= limit:
                if plan == 'pro':
                    msg = f"You've used all {limit} {label} for today. Resets at midnight."
                else:
                    msg = f"You've used your {limit} free {label}. Upgrade to Pro for more."
                return jsonify({
                    'success':        False,
                    'limit_reached':  True,
                    'is_pro':         plan == 'pro',
                    'feature':        feature_key,
                    'feature_label':  label,
                    'used':           used,
                    'limit':          limit,
                    'error':          msg,
                    'upgrade_prompt': plan != 'pro',
                    'upgrade_url':    '/pricing',
                }), 429

            return f(*args, **kwargs)
        return decorated
    return decorator


# ════════════════════════════════════════════════════════════════
#  LANGUAGE MAPS
# ════════════════════════════════════════════════════════════════

EDGE_VOICES = {
    'en':'en-US-JennyNeural','es':'es-ES-ElviraNeural','fr':'fr-FR-DeniseNeural',
    'de':'de-DE-KatjaNeural','ja':'ja-JP-NanamiNeural','zh':'zh-CN-XiaoxiaoNeural',
    'ar':'ar-SA-ZariyahNeural','hi':'hi-IN-SwaraNeural','pt':'pt-BR-FranciscaNeural',
    'ru':'ru-RU-SvetlanaNeural','it':'it-IT-ElsaNeural','ko':'ko-KR-SunHiNeural',
    'EN':'en-US-JennyNeural','ES':'es-ES-ElviraNeural','FR':'fr-FR-DeniseNeural',
    'DE':'de-DE-KatjaNeural','JA':'ja-JP-NanamiNeural','ZH':'zh-CN-XiaoxiaoNeural',
    'AR':'ar-SA-ZariyahNeural','HI':'hi-IN-SwaraNeural','PT':'pt-BR-FranciscaNeural',
    'RU':'ru-RU-SvetlanaNeural','IT':'it-IT-ElsaNeural','KO':'ko-KR-SunHiNeural',
}
DEEPL_LANGS = {
    'en':'EN-US','es':'ES','fr':'FR','de':'DE','ja':'JA','zh':'ZH',
    'pt':'PT-BR','ru':'RU','it':'IT','ko':'KO','ar':None,'hi':None,
    'EN':'EN-US','ES':'ES','FR':'FR','DE':'DE','JA':'JA','ZH':'ZH',
    'PT':'PT-BR','RU':'RU','IT':'IT','KO':'KO','AR':None,'HI':None,
}
LANG_NAMES = {
    'en':'English','es':'Spanish','fr':'French','de':'German','ja':'Japanese',
    'zh':'Chinese','ar':'Arabic','hi':'Hindi','pt':'Portuguese','ru':'Russian',
    'it':'Italian','ko':'Korean','EN':'English','ES':'Spanish','FR':'French',
    'DE':'German','JA':'Japanese','ZH':'Chinese','AR':'Arabic','HI':'Hindi',
    'PT':'Portuguese','RU':'Russian','IT':'Italian','KO':'Korean',
}
WHISPER_LANG = {
    'en':'en','es':'es','fr':'fr','de':'de','ja':'ja','zh':'zh',
    'ar':'ar','hi':'hi','pt':'pt','ru':'ru','it':'it','ko':'ko',
    'EN':'en','ES':'es','FR':'fr','DE':'de','JA':'ja','ZH':'zh',
    'AR':'ar','HI':'hi','PT':'pt','RU':'ru','IT':'it','KO':'ko',
}
WHISPER_PROMPTS = {
    'en':'This is a clear English conversation.',
    'hi':'यह हिंदी में बातचीत है।',
    'ja':'これは日本語の会話です。',
    'zh':'这是普通话对话。',
    'fr':'Ceci est une conversation en français.',
    'es':'Esta es una conversación en español.',
    'de':'Dies ist ein Gespräch auf Deutsch.',
    'ar':'هذه محادثة باللغة العربية.',
    'ru':'Это разговор на русском языке.',
    'ko':'이것은 한국어 대화입니다.',
    'it':'Questa è una conversazione in italiano.',
    'pt':'Esta é uma conversa em português.',
}
HALLUCINATIONS = {
    'you','You','the','The','Thank you.','Thank you','Thanks','Thanks.',
    'Bye','bye','Okay','okay','OK','ok','',' ','.','...','..','Hmm','hmm',
    'Um','um','Uh','uh','Ah','ah','Oh','oh','i','I','A','a',
    'Subtitles by','Subscribe','MBC','Please subscribe',
}
SLOW_LANGS = {'hi','ar','zh','ja','ko'}


# ════════════════════════════════════════════════════════════════
#  AUDIO / AI HELPERS
# ════════════════════════════════════════════════════════════════

def get_rms(audio_bytes):
    try:
        count = len(audio_bytes) // 2
        if count == 0: return 0
        samples = struct.unpack('<' + 'h' * count, audio_bytes[:count * 2])
        return (sum(s * s for s in samples) / count) ** 0.5
    except: return 0

def normalize_audio(raw_bytes):
    try:
        count = len(raw_bytes) // 2
        if count == 0: return raw_bytes
        samples = list(struct.unpack('<' + 'h' * count, raw_bytes[:count * 2]))
        peak = max(abs(s) for s in samples)
        if peak == 0: return raw_bytes
        factor = min((32767 * 0.8) / peak, 4.0)
        normalized = [max(-32768, min(32767, int(s * factor))) for s in samples]
        return struct.pack('<' + 'h' * len(normalized), *normalized)
    except: return raw_bytes

def audio_to_wav(raw_bytes, sample_rate=16000):
    n = normalize_audio(raw_bytes)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sample_rate); wf.writeframes(n)
    buf.seek(0); return buf.read()

def is_valid(text):
    if not text: return False
    t = text.strip()
    if len(t) < 3 or t in HALLUCINATIONS: return False
    alpha = sum(c.isalpha() for c in t)
    return alpha >= len(t) * 0.25

def safe_send(ws, data):
    try: ws.send(json.dumps(data))
    except: pass

def groq_json(prompt, model=SCOUT, temp=0.3, max_tok=1000):
    models_to_try = [model]
    if model == MODEL_70B: models_to_try.append(SCOUT)
    last_error = None
    for m in models_to_try:
        try:
            response = groq_client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. No markdown. No backticks. No explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp, max_tokens=max_tok
            )
            return clean_json(response.choices[0].message.content)
        except Exception as e:
            last_error = e
            print(f"[Groq] {m} failed: {e}")
            continue
    raise last_error

def clean_json(text):
    text = text.strip()
    if '```' in text:
        parts = text.split('```')
        for part in parts:
            if '{' in part:
                text = part
                if text.startswith('json'): text = text[4:]
                break
    start = text.find('{'); end = text.rfind('}') + 1
    if start != -1 and end > start: text = text[start:end]
    return json.loads(text)

def transcribe(wav_data, lang_hint=None):
    kwargs = {
        'file': ('audio.wav', wav_data),
        'model': 'whisper-large-v3-turbo',
        'response_format': 'verbose_json',
        'temperature': 0.0,
    }
    if lang_hint and lang_hint not in ('auto', 'unknown', None, ''):
        wc = WHISPER_LANG.get(lang_hint)
        if wc:
            kwargs['language'] = wc
            prompt = WHISPER_PROMPTS.get(wc, '')
            if prompt: kwargs['prompt'] = prompt
    result   = groq_client.audio.transcriptions.create(**kwargs)
    text     = result.text.strip()
    detected = getattr(result, 'language', 'unknown')
    segments = getattr(result, 'segments', [])
    conf     = (sum(abs(s.get('avg_logprob', -1)) for s in segments) / max(len(segments), 1) if segments else 0.0)
    return text, detected, conf

def translate(text, target_lang, src_lang=None):
    tgt = target_lang.lower()[:2] if len(target_lang) >= 2 else target_lang
    deepl_code = DEEPL_LANGS.get(tgt) or DEEPL_LANGS.get(target_lang)
    if deepl_code and deepl_client:
        try:
            src = None
            if src_lang and src_lang not in ('unknown', 'auto', None, ''):
                src = src_lang.upper()[:2]
                if src.lower() == tgt.lower(): src = None
            result = deepl_client.translate_text(text, target_lang=deepl_code, source_lang=src)
            return result.text, 'DeepL'
        except Exception as e:
            print(f"[DeepL] {e}")
    lang_name = LANG_NAMES.get(target_lang, 'English')
    r = groq_client.chat.completions.create(
        model=SCOUT,
        messages=[
            {'role': 'system', 'content': f'Translate to {lang_name}. Return ONLY the translation.'},
            {'role': 'user', 'content': text}
        ],
        temperature=0.1, max_tokens=500
    )
    return r.choices[0].message.content.strip(), 'Groq AI'

def tts(text, lang_code):
    async def _run():
        voice = EDGE_VOICES.get(lang_code, 'en-US-JennyNeural')
        try:
            communicate = edge_tts.Communicate(text, voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk['type'] == 'audio': buf.write(chunk['data'])
            buf.seek(0); data = buf.read()
            if len(data) > 100: return data
            raise Exception("Empty audio")
        except Exception:
            communicate = edge_tts.Communicate(text, 'en-US-JennyNeural')
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk['type'] == 'audio': buf.write(chunk['data'])
            buf.seek(0); return buf.read()
    return asyncio.run(_run())


# ════════════════════════════════════════════════════════════════
#  ADMIN DECORATOR
# ════════════════════════════════════════════════════════════════

ADMIN_EMAIL = 'yaplytranslator@gmail.com'

def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()
        if not token:
            return jsonify({'success': False, 'error': 'No token'}), 401
        payload = decode_token(token)
        if not payload:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
        user = get_user_by_id(payload.get('user_id'))
        if not user or user['email'].lower() != ADMIN_EMAIL.lower():
            return jsonify({'success': False, 'error': 'Admin only'}), 403
        g.user_id = user['id']
        g.user    = user
        return f(*args, **kwargs)
    return decorated


# ════════════════════════════════════════════════════════════════
#  HEALTH
# ════════════════════════════════════════════════════════════════

@app.route('/health')
def health(): return 'OK', 200

@app.route('/ping')
def ping(): return 'pong', 200


# ════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ════════════════════════════════════════════════════════════════

@app.route('/')
def landing(): return render_template('landing.html')

@app.route('/login')
@limiter.exempt
def login_page(): return render_template('auth.html', google_client_id=GOOGLE_CLIENT_ID)

@app.route('/app')
@limiter.exempt
def main_app():
    token = request.args.get('token', '')
    name  = request.args.get('name', '')
    return render_template('yaply-app.html', google_client_id=GOOGLE_CLIENT_ID, oauth_token=token, oauth_name=name)

@app.route('/onboarding')
def onboarding_page(): return render_template('onboarding.html')

@app.route('/pricing')
def pricing_page(): return render_template('pricing.html')

@app.route('/plan')
def plan_page(): return render_template('before_trip.html')

@app.route('/during')
def during_page(): return render_template('during_trip.html')

@app.route('/after')
@app.route('/after-trip')
def after_page(): return render_template('after_trip.html')

@app.route('/tools')
def tools_page(): return render_template('tools_extra.html')

@app.route('/discover')
def discover_page(): return render_template('discover.html')

@app.route('/translate')
def translate_page(): return render_template('stream.html')

@app.route('/convo')
def convo_page(): return render_template('convo.html')

@app.route('/camera')
def camera_page(): return render_template('camera.html')

@app.route('/diary')
def diary_page(): return render_template('yaply_diary.html')

@app.route('/groups')
@app.route('/groups/<int:group_id>')
def groups_page(group_id=None): return render_template('yaply_groups.html')

@app.route('/join/<code>')
@limiter.exempt
def join_group_magic(code): return render_template('yaply_groups.html', join_code=code)

@app.route('/me')
def profile_page(): return render_template('profile.html')

@app.route('/admin')
def admin_page(): return render_template('yaply_admin.html')

@app.route('/privacy')
def privacy(): return render_template('privacy.html')

@app.route('/terms')
def terms(): return render_template('terms.html')

@app.route('/offline')
def offline(): return render_template('offline.html')


# ════════════════════════════════════════════════════════════════
#  AUTH + USER API
# ════════════════════════════════════════════════════════════════

@app.route('/api/me', methods=['GET'])
@require_auth
def api_me():
    return jsonify({'success': True, 'user': safe_user(g.user)})

@app.route('/api/me/status')
@require_auth
def api_user_status():
    """Full plan status + usage + limits — called on every page load."""
    try:
        usage_data = get_user_usage(g.user_id)
        if not usage_data:
            return jsonify({'success': False, 'error': 'User not found'})

        plan   = usage_data['plan']
        limits = usage_data['limits']
        usage  = usage_data['usage']

        remaining = {}
        for feat, used_key in [
            ('translations_day',   'translations_today'),
            ('voice_day',          'voice_today'),
            ('identify_photo_day', 'identify_today'),
            ('tools_day',          'tools_today'),
            ('plans_month',        'plans_used_month'),
            ('multicity_month',    'multicity_used_month'),
            ('ai_story_month',     'ai_story_used'),
            ('ai_journal_month',   'journal_used_month'),
        ]:
            lim  = limits.get(feat, 0)
            used = usage.get(used_key, 0)
            remaining[feat] = max(0, lim - used)

        return jsonify({
            'success':         True,
            'plan':            plan,
            'is_pro':          plan == 'pro',
            'pro_expires_at':  usage_data.get('pro_expires_at'),
            'days_remaining':  usage_data.get('days_remaining', 0),
            'onboarding_done': usage_data.get('onboarding_done', False),
            'pro_expired':     bool(usage_data.get('pro_expires_at') and plan == 'free'),
            'limits':          limits,
            'usage':           usage,
            'remaining':       remaining,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/me/update', methods=['POST'])
@require_auth
def api_update_profile():
    try:
        data    = request.get_json() or {}
        allowed = ['name', 'home_city', 'passport', 'currency']
        updates = {k: data[k] for k in allowed if k in data and data[k]}
        if updates: update_user(g.user_id, **updates)
        user = get_user_by_id(g.user_id)
        return jsonify({'success': True, 'user': safe_user(user)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/profile', methods=['GET'])
@require_auth
def api_profile():
    return jsonify({'success': True, 'user': safe_user(g.user), 'stats': get_user_stats(g.user_id)})


# ════════════════════════════════════════════════════════════════
#  ONBOARDING
# ════════════════════════════════════════════════════════════════

@app.route('/api/onboarding/complete', methods=['POST'])
@require_auth
def api_complete_onboarding():
    try:
        data         = request.get_json() or {}
        name         = clean(data.get('name', ''), 50)
        home_city    = clean(data.get('home_city', ''), 50)
        passport     = clean(data.get('passport', 'India'), 30)
        currency     = clean(data.get('currency', 'INR'), 3)
        travel_style = clean(data.get('travel_style', ''), 20)
        budget_style = clean(data.get('budget_style', ''), 20)

        if not name:
            return jsonify({'success': False, 'error': 'Name is required'})

        complete_onboarding(g.user_id, name, home_city, passport, currency, travel_style, budget_style)
        user = get_user_by_id(g.user_id)
        log_action(g.user_id, 'onboarding_complete', request.remote_addr)

        return jsonify({'success': True, 'user': safe_user(user), 'redirect': '/pricing'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ════════════════════════════════════════════════════════════════
#  TRIP ROUTES
# ════════════════════════════════════════════════════════════════

@app.route('/api/trips', methods=['GET'])
@require_auth
def api_get_trips(): return jsonify({'success': True, 'trips': get_trips(g.user_id)})

@app.route('/api/trips', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_save_trip():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success': False, 'error': err})
        trip_id = save_trip(
            user_id=g.user_id, destination=clean(data.get('destination', '')),
            origin=clean(data.get('origin', 'India')),
            days=min(max(int(data.get('days', 7)), 1), 365),
            people=min(max(int(data.get('people', 1)), 1), 50),
            budget=clean(data.get('budget', '80000')),
            currency=clean(data.get('currency', 'INR'), 3),
            vibes=clean(data.get('vibes', 'Adventure')),
            passport=clean(data.get('passport', 'India')),
            plan_data=data.get('plan_data'),
            start_date=clean(data.get('start_date', ''))
        )
        log_action(g.user_id, 'save_trip', request.remote_addr)
        return jsonify({'success': True, 'trip_id': trip_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trips/<int:trip_id>', methods=['GET'])
@require_auth
def api_get_trip(trip_id):
    trip = get_trip(trip_id, g.user_id)
    if not trip: return jsonify({'success': False, 'error': 'Trip not found'}), 404
    return jsonify({'success': True, 'trip': trip})

@app.route('/api/trips/<int:trip_id>', methods=['PUT'])
@require_auth
def api_update_trip(trip_id):
    try:
        update_trip(trip_id, g.user_id, **(request.get_json() or {}))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trips/<int:trip_id>', methods=['DELETE'])
@require_auth
def api_delete_trip(trip_id):
    delete_trip(trip_id, g.user_id)
    return jsonify({'success': True})


# ════════════════════════════════════════════════════════════════
#  PLACES + EXPENSES + JOURNAL
# ════════════════════════════════════════════════════════════════

@app.route('/api/places', methods=['GET'])
@require_auth
def api_get_places(): return jsonify({'success': True, 'places': get_places(g.user_id)})

@app.route('/api/places', methods=['POST'])
@require_auth
def api_save_place():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['name'])
        if not ok: return jsonify({'success': False, 'error': err})
        place_id = save_place(
            user_id=g.user_id, name=clean(data.get('name', '')),
            city=clean(data.get('city', '')), country=clean(data.get('country', '')),
            continent=clean(data.get('continent', '')),
            description=clean(data.get('description', ''), 1000),
            image_url=clean(data.get('image_url', ''), 500),
            emoji=clean(data.get('emoji', '📍'), 5),
            tags=data.get('tags', []), trip_id=data.get('trip_id')
        )
        return jsonify({'success': True, 'place_id': place_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/places/<int:place_id>', methods=['DELETE'])
@require_auth
def api_delete_place(place_id):
    delete_place(place_id, g.user_id)
    return jsonify({'success': True})

@app.route('/api/trips/<int:trip_id>/expenses', methods=['GET'])
@require_auth
def api_get_expenses(trip_id):
    return jsonify({'success': True, 'expenses': get_expenses(trip_id, g.user_id)})

@app.route('/api/trips/<int:trip_id>/expenses', methods=['POST'])
@require_auth
def api_add_expense(trip_id):
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['title', 'amount'])
        if not ok: return jsonify({'success': False, 'error': err})
        exp_id = add_expense(
            trip_id=trip_id, user_id=g.user_id,
            title=clean(data.get('title', '')), amount=float(data.get('amount', 0)),
            category=clean(data.get('category', 'Other')),
            currency=clean(data.get('currency', 'INR'), 3),
            paid_by=clean(data.get('paid_by', '')), split_with=data.get('split_with', [])
        )
        return jsonify({'success': True, 'expense_id': exp_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/expenses/<int:expense_id>', methods=['DELETE'])
@require_auth
def api_delete_expense(expense_id):
    delete_expense(expense_id, g.user_id)
    return jsonify({'success': True})

@app.route('/api/trips/<int:trip_id>/journal', methods=['GET'])
@require_auth
def api_get_journal(trip_id):
    return jsonify({'success': True, 'journal': get_journal(trip_id, g.user_id)})

@app.route('/api/trips/<int:trip_id>/journal', methods=['POST'])
@require_auth
def api_save_journal_route(trip_id):
    try:
        data = request.get_json() or {}
        save_journal(trip_id, g.user_id, data.get('content'))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ════════════════════════════════════════════════════════════════
#  AI — TRIP PLANNER
# ════════════════════════════════════════════════════════════════

@app.route('/api/plan', methods=['POST'])
@require_auth
@limiter.limit("15 per hour")
@check_feature_limit('plan')
def api_plan():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success': False, 'error': err})

        destination = clean(data.get('destination', ''))
        origin      = clean(data.get('origin', 'India'))
        days        = min(max(int(data.get('days', 5)), 1), 60)
        budget      = clean(data.get('budget', '50000'))
        vibe        = clean(data.get('vibe', 'adventure'))
        people      = min(max(int(data.get('people', 1)), 1), 20)
        currency    = clean(data.get('currency', 'INR'), 3)
        passport    = clean(data.get('passport', 'India'))
        travel_mode = clean(data.get('travel_mode', 'smart'), 20).lower()

        origin_country = origin.split(',')[-1].strip().lower() if ',' in origin else origin.lower()
        dest_country   = destination.split(',')[-1].strip().lower() if ',' in destination else destination.lower()
        is_domestic    = origin_country in dest_country or dest_country in origin_country

        if travel_mode == 'flight' or (travel_mode == 'smart' and not is_domestic):
            transport_instruction = f"TRANSPORT MODE: Flight\n- Include flight info and costs in {currency}\n- Include best_airlines, flight_duration, best_time_to_book"
        elif travel_mode == 'train':
            transport_instruction = f"TRANSPORT MODE: Train\n- NO flights\n- budget_breakdown.flights = 'Not applicable'\n- Include train_info with operator, class_options, cost in {currency}, duration, booking_tip"
        elif travel_mode == 'bus':
            transport_instruction = f"TRANSPORT MODE: Bus\n- NO flights\n- budget_breakdown.flights = 'Not applicable'\n- Include bus_info with operators, type, cost in {currency}, duration, booking_tip"
        elif travel_mode == 'road':
            transport_instruction = f"TRANSPORT MODE: Road\n- NO flights\n- Include road_info with distance_km, drive_duration, fuel_cost in {currency}, toll_cost, cab_option, stops_enroute"
        else:
            if is_domestic:
                transport_instruction = f"TRANSPORT MODE: Smart Domestic ({origin} to {destination})\n- If <500km: recommend train/bus not flight\n- If 500-1000km: train or flight based on cost\n- If >1000km: flight\n- Include transport_recommendation with mode, reason, cost, duration"
            else:
                transport_instruction = f"TRANSPORT MODE: Smart International\n- Include flight info in {currency}"

        prompt = f"""You are a world-class travel planner. Create a detailed {days}-day trip plan.
TRIP: FROM {origin} TO {destination} | {days} days | {people} people | {currency} {budget} | Style: {vibe} | Passport: {passport}
{transport_instruction}
RULES: All prices in {currency}. Budget realistic. Activities match {vibe} style. Hidden gems must be genuinely lesser-known.
Return ONLY valid JSON with: destination, origin, days, travel_mode, language, currency, timezone, best_time_to_visit,
budget_breakdown, flight_info, train_info, bus_info, road_info, transport_recommendation,
itinerary (day/title/morning/afternoon/evening/lunch/dinner/accommodation),
hidden_gems, local_transport, sim_internet, cultural_guide, vaccinations, packing_list,
emergency_numbers, visa_info, payment_info, must_have_apps, power_plug, what_to_buy, what_to_avoid, local_phrases, tips"""

        result = groq_json(prompt, model=MODEL_70B, temp=0.3, max_tok=6000)

        # Increment AFTER success
        execute('UPDATE users SET plans_used_month=plans_used_month+1 WHERE id=%s', (g.user_id,))

        trip_id = data.get('trip_id')
        if trip_id: update_trip(trip_id, g.user_id, plan_data=result, status='active')

        log_action(g.user_id, 'plan_trip', request.remote_addr)
        return jsonify({'success': True, 'plan': result})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ── MULTI-CITY PLANNER ────────────────────────────────────────

@app.route('/api/multi-city-plan', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
@check_feature_limit('multicity')
def multi_city_plan():
    try:
        data         = request.get_json() or {}
        origin       = clean(data.get('origin', 'India'))
        cities       = data.get('cities', [])
        total_budget = data.get('total_budget', 100000)
        currency     = clean(data.get('currency', 'INR'), 3)
        vibe         = clean(data.get('vibe', 'adventure'))
        people       = min(max(int(data.get('people', 1)), 1), 20)
        start_date   = clean(data.get('start_date', ''))
        passport     = clean(data.get('passport', 'India'))

        if not cities or len(cities) < 2:
            return jsonify({'success': False, 'error': 'Please add at least 2 cities.'})
        if len(cities) > 6:
            return jsonify({'success': False, 'error': 'Maximum 6 cities supported.'})

        total_days = sum(int(c.get('days', 3)) for c in cities)
        city_list  = ', '.join(c['name'] for c in cities)

        prompt = (
            "Plan an epic multi-city trip:\n"
            f"Origin: {origin} | Cities: {city_list} | {total_days} days | {currency} {total_budget} for {people} | Style: {vibe} | Passport: {passport}\n"
            f"Cities data: {json.dumps(cities)}\n"
            f"RULES: All prices in {currency}. Plan each city with full itinerary. Include transit between every city.\n"
            "Return ONLY valid JSON with: trip_title, origin, total_days, total_budget, currency, cities_count, "
            "route_overview, smart_suggestions, budget_split, cities (array with full itinerary per city), "
            "transit_plans, sim_strategy, packing_for_route, money_saving_tips"
        )

        response = groq_client.chat.completions.create(
            model=MODEL_70B,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown. No backticks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, max_tokens=8000
        )
        plan = clean_json(response.choices[0].message.content)

        # Increment AFTER success
        execute('UPDATE users SET multicity_used_month=multicity_used_month+1 WHERE id=%s', (g.user_id,))

        log_action(g.user_id, 'multi_city_plan', request.remote_addr)
        return jsonify({'success': True, 'plan': plan})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ── JOURNEY PLANNER ───────────────────────────────────────────

@app.route('/api/journey', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_journey():
    try:
        data        = request.get_json() or {}
        origin      = clean(data.get('origin', ''))
        destination = clean(data.get('destination', ''))
        travel_mode = clean(data.get('travel_mode', 'any'))
        currency    = clean(data.get('currency', 'INR'), 3)

        if not origin or not destination:
            return jsonify({'success': False, 'error': 'Origin and destination required'})

        if travel_mode == 'train':
            mode_instruction = "TRAVEL MODE: Train only. nearest_airports = train stations. Include train operators, IRCTC. NO flights."
        elif travel_mode == 'bus':
            mode_instruction = "TRAVEL MODE: Bus only. nearest_airports = bus terminals. Include bus operators. NO flights."
        elif travel_mode == 'road':
            mode_instruction = "TRAVEL MODE: Road only. Include distance, drive time, fuel cost, toll. NO flights."
        else:
            mode_instruction = "TRAVEL MODE: Best option. Include flights if international/long distance, train as alternative if domestic."

        result = groq_json(
            f"""Complete door-to-door journey planner.
FROM: {origin} TO: {destination} CURRENCY: {currency}
{mode_instruction}
Return ONLY valid JSON with: origin, destination, nearest_airports (name/city/code/distance_from_origin),
recommended_route (step1/step2/step3/step4/total_duration/total_cost),
flight_options (airline/mode/duration/price/stops/class/recommended),
alternative_routes, important_notes, documents_needed""",
            model=MODEL_70B, temp=0.3, max_tok=3000
        )
        return jsonify({'success': True, 'journey': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── WEATHER ───────────────────────────────────────────────────

@app.route('/api/weather', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
@check_feature_limit('tool')
def api_weather():
    try:
        city = clean((request.get_json() or {}).get('city', ''))
        if not city: return jsonify({'success': False, 'error': 'City required'})
        if not WEATHER_KEY: return jsonify({'success': False, 'error': 'Weather API key not configured'})
        url  = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_KEY}&units=metric&cnt=40"
        r    = req.get(url, timeout=10)
        data = r.json()
        if data.get('cod') != '200': return jsonify({'success': False, 'error': 'City not found'})
        daily = {}
        for item in data['list']:
            date = item['dt_txt'].split(' ')[0]
            if date not in daily:
                daily[date] = {'date': date, 'temp_max': item['main']['temp_max'], 'temp_min': item['main']['temp_min'],
                               'description': item['weather'][0]['description'], 'icon': item['weather'][0]['icon'],
                               'humidity': item['main']['humidity'], 'wind': item['wind']['speed']}
            else:
                daily[date]['temp_max'] = max(daily[date]['temp_max'], item['main']['temp_max'])
                daily[date]['temp_min'] = min(daily[date]['temp_min'], item['main']['temp_min'])
        # Increment AFTER success
        execute('UPDATE users SET tools_today=tools_today+1 WHERE id=%s', (g.user_id,))
        return jsonify({'success': True, 'city': data['city']['name'], 'country': data['city']['country'], 'forecast': list(daily.values())[:7]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── CURRENCY ──────────────────────────────────────────────────

@app.route('/api/currency', methods=['POST'])
@require_auth
@limiter.limit("60 per hour")
def api_currency():
    try:
        data   = request.get_json() or {}
        amount = float(data.get('amount', 1))
        from_c = clean(data.get('from', 'INR'), 3).upper()
        to_c   = clean(data.get('to', 'USD'), 3).upper()
        if not EXCHANGE_KEY: return jsonify({'success': False, 'error': 'Exchange API key not configured'})
        r = req.get(f"https://v6.exchangerate-api.com/v6/{EXCHANGE_KEY}/pair/{from_c}/{to_c}/{amount}", timeout=10).json()
        if r.get('result') != 'success': return jsonify({'success': False, 'error': 'Currency not found'})
        return jsonify({'success': True, 'from': from_c, 'to': to_c, 'amount': amount, 'converted': round(r['conversion_result'], 2), 'rate': r['conversion_rate']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── VISA ──────────────────────────────────────────────────────

@app.route('/api/visa', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
@check_feature_limit('tool')
def api_visa():
    try:
        data        = request.get_json() or {}
        passport    = clean(data.get('passport', 'India'))
        destination = clean(data.get('destination', ''))
        if not destination: return jsonify({'success': False, 'error': 'Destination required'})
        result = groq_json(
            f"""Visa requirements for {passport} passport holder visiting {destination}.
Return JSON: visa_required, visa_type, validity, cost, processing_days, apply_online, apply_url, documents, tips, visa_on_arrival, visa_free_days"""
        )
        execute('UPDATE users SET tools_today=tools_today+1 WHERE id=%s', (g.user_id,))
        return jsonify({'success': True, 'visa': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── PLACE FINDER (PHOTO) ─────────────────────────────────────

@app.route('/api/identify', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
@check_feature_limit('identify')
def api_identify():
    try:
        data         = request.get_json() or {}
        image_base64 = data.get('image', '')
        if not image_base64: return jsonify({'success': False, 'error': 'No image provided'})
        if len(image_base64) > 5 * 1024 * 1024: return jsonify({'success': False, 'error': 'Image too large (max 5MB)'})

        response = groq_client.chat.completions.create(
            model=SCOUT,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": """Identify this location. Return ONLY valid JSON:
{"place_name":"exact name","city":"city","country":"country","confidence":92,"place_type":"type","description":"2-3 sentences",
"tags":["tag1"],"best_time":"months","climate":"type","budget_level":"Budget/Mid-range/Luxury",
"avg_daily_cost":"USD X/day","language":"language","currency":"currency","nearest_airport":"airport",
"why_famous":"reason","nearby":[{"name":"place","distance":"X km","type":"Attraction","icon":"emoji"}],
"similar_places":[{"name":"place","country":"country","why_similar":"reason","emoji":"flag"}],
"travel_tips":["tip1"],"best_food":["dish1"]}"""}
            ]}],
            temperature=0.1, max_tokens=1500
        )
        result = clean_json(response.choices[0].message.content)
        execute('UPDATE users SET identify_today=identify_today+1 WHERE id=%s', (g.user_id,))
        log_action(g.user_id, 'identify_place', request.remote_addr)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ── PLACE FINDER (TEXT) ──────────────────────────────────────

@app.route('/api/identify-text', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_identify_text():
    try:
        description = clean((request.get_json() or {}).get('description', ''), 300)
        if not description: return jsonify({'success': False, 'error': 'Description required'})
        result = groq_json(
            f"""Someone saw this place: "{description}". Identify the exact location.
Return JSON: place_name, city, country, continent, confidence, place_type, description, tags,
best_time, climate, budget_level, avg_daily_cost, language, currency, nearest_airport, airport_code,
why_famous, nearby, similar_places, travel_tips, best_food""",
            model=SCOUT, temp=0.2, max_tok=1000
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── SAFETY TOOLS ──────────────────────────────────────────────

@app.route('/api/scam-alerts', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
@check_feature_limit('tool')
def api_scam_alerts():
    try:
        destination = clean((request.get_json() or {}).get('destination', ''))
        if not destination: return jsonify({'success': False, 'error': 'Destination required'})
        result = groq_json(
            f"""All tourist scams in {destination}. Return JSON: scam_risk_level,
scams (name/category/severity/how_it_works/red_flags/how_to_avoid/what_to_say/icon),
general_rules, safe_alternatives, emergency_if_robbed""",
            model=SCOUT, temp=0.2, max_tok=1000
        )
        execute('UPDATE users SET tools_today=tools_today+1 WHERE id=%s', (g.user_id,))
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/price-check', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def api_price_check():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['item', 'destination'])
        if not ok: return jsonify({'success': False, 'error': err})
        result = groq_json(
            f"""Price check: "{clean(data.get('item',''),100)}" at {clean(data.get('destination',''))} for {clean(data.get('currency','INR'),3)} {clean(data.get('price',''),20)}.
Return JSON: verdict, verdict_color, fair_price_range, local_price, tourist_price, overpaying_by,
verdict_explanation, negotiation_tips, walk_away_price, local_phrase_to_say"""
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/safe-route', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_safe_route():
    try:
        data = request.get_json() or {}
        result = groq_json(
            f"""Safe route from {clean(data.get('from_location',''))} to {clean(data.get('to_location',''))} in {clean(data.get('destination',''))} at {clean(data.get('time_of_day',''))} for {clean(data.get('traveller_type',''))}.
Return JSON: route_safety, safety_score, recommended_transport, areas_to_avoid, if_harassed, trusted_contacts, pro_tips"""
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/medical-translate', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_medical_translate():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['symptoms', 'destination'])
        if not ok: return jsonify({'success': False, 'error': err})
        result = groq_json(
            f"""Medical translation: symptoms="{clean(data.get('symptoms',''),200)}", destination={clean(data.get('destination',''))}, language={clean(data.get('language','Japanese'))}.
Return JSON: severity, possible_conditions, translated_symptoms, pronunciation, say_to_doctor,
immediate_actions, medicines_to_ask, emergency_number, medical_phrases"""
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/safety-check', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
@check_feature_limit('tool')
def api_safety_check():
    try:
        destination = clean((request.get_json() or {}).get('destination', ''))
        if not destination: return jsonify({'success': False, 'error': 'Destination required'})
        result = groq_json(
            f"""Travel safety expert. Specific safety info for {destination}.
Return JSON: destination, safety_score (1-100 specific to {destination}), safety_level, crime_index,
tourist_safety, water_safe, water_advice, food_safety, health_risks, scams_to_avoid,
safe_areas, avoid_areas, emergency_embassy, embassy_phone, travel_advisory,
solo_female_safety, best_safety_tips""",
            model=SCOUT, temp=0.3, max_tok=1200
        )
        execute('UPDATE users SET tools_today=tools_today+1 WHERE id=%s', (g.user_id,))
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/local-laws', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_local_laws():
    try:
        destination = clean((request.get_json() or {}).get('destination', ''))
        if not destination: return jsonify({'success': False, 'error': 'Destination required'})
        result = groq_json(
            f"""Important laws for tourists in {destination}.
Return JSON: strict_laws (law/penalty/severity/icon), photography_rules, dress_code_rules,
alcohol_rules, drug_laws, customs_limits, good_to_know, legal_tip""",
            model=SCOUT, temp=0.1, max_tok=1000
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/flight-rights', methods=['POST'])
@require_auth
@limiter.limit("15 per hour")
def api_flight_rights():
    try:
        data = request.get_json() or {}
        result = groq_json(
            f"""Flight rights: {clean(data.get('airline',''))} on {clean(data.get('route',''))}, issue: {clean(data.get('issue',''))}, delay: {data.get('delay_hours',0)}h.
Return JSON: entitled_to_compensation, compensation_amount, your_rights, immediate_actions,
documents_to_collect, what_airline_must_provide, how_to_claim, exact_phrases_to_say, claim_template"""
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/allergy-card', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_allergy_card():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['name', 'destination'])
        if not ok: return jsonify({'success': False, 'error': err})
        allergies_str = ', '.join([clean(a, 50) for a in data.get('allergies', [])][:10])
        result = groq_json(
            f"""Allergy card for {clean(data.get('name',''),50)} allergic to {allergies_str} visiting {clean(data.get('destination',''))}.
Return JSON: allergy_card_text, dangerous_dishes, safe_dishes, hidden_allergens,
phrases_to_say, restaurant_tips, emergency_protocol, medicines_to_carry"""
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/immigration-help', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_immigration_help():
    try:
        data = request.get_json() or {}
        result = groq_json(
            f"""Immigration guide for {clean(data.get('passport','India'))} entering {clean(data.get('destination',''))} for {clean(data.get('purpose','Tourism'))}.
Return JSON: common_questions, documents_to_keep_ready, declaration_items, common_mistakes,
if_stopped_for_questioning, pro_tips"""
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/emergency-card', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_emergency_card():
    try:
        data        = request.get_json() or {}
        name        = clean(data.get('name', ''))
        blood_group = clean(data.get('blood_group', ''))
        allergies   = clean(data.get('allergies', 'none'))
        destination = clean(data.get('destination', ''))
        if not name or not destination: return jsonify({'success': False, 'error': 'Name and destination required'})
        result = groq_json(
            f"""Emergency card for {name} (blood: {blood_group}, allergies: {allergies}) visiting {destination}.
Return JSON: emergency_numbers, indian_embassy, nearest_hospitals, medical_phrases,
what_to_do_if_robbed, what_to_do_if_sick, what_to_do_if_lost""",
            model=SCOUT, temp=0.1, max_tok=1000
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── PLANNING TOOLS ────────────────────────────────────────────

@app.route('/api/sim-guide', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def sim_guide():
    try:
        data        = request.get_json() or {}
        destination = clean(data.get('destination', ''))
        origin      = clean(data.get('origin', 'India'))
        days        = int(data.get('days', 7))
        data_needs  = clean(data.get('data_needs', 'moderate'))
        countries   = data.get('countries', [destination])
        if not destination: return jsonify({'success': False, 'error': 'Destination required'})
        countries_str = ', '.join(countries) if isinstance(countries, list) else destination
        result = groq_json(
            f"""SIM card guide for traveller from {origin} visiting {countries_str} for {days} days. Data: {data_needs}.
Return JSON: top_recommendation, all_options, esim_options, airport_buying_guide,
roaming_option, connectivity_tips, data_saving_tips, offline_essentials, budget_summary""",
            model=SCOUT, temp=0.2, max_tok=1000
        )
        return jsonify({'success': True, 'guide': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/jetlag', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_jetlag():
    try:
        data        = request.get_json() or {}
        from_city   = clean(data.get('from_city', ''))
        to_city     = clean(data.get('to_city', ''))
        travel_date = clean(data.get('travel_date', 'upcoming'))
        if not from_city or not to_city: return jsonify({'success': False, 'error': 'Both cities required'})
        result = groq_json(
            f"""Jet lag plan for {from_city} to {to_city} on {travel_date}.
Return JSON: from_timezone, to_timezone, time_difference, jet_lag_severity, recovery_days,
direction, symptoms, before_flight, during_flight, after_arrival, sleep_schedule, avoid, recovery_tip""",
            model=SCOUT, temp=0.2, max_tok=1000
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/festivals', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_festivals():
    try:
        data        = request.get_json() or {}
        destination = clean(data.get('destination', ''))
        travel_date = clean(data.get('travel_date', 'this month'))
        if not destination: return jsonify({'success': False, 'error': 'Destination required'})
        result = groq_json(
            f"""Festivals and events in {destination} around {travel_date}.
Return JSON: public_holidays, festivals, peak_season, season_type, price_impact,
crowd_level, booking_advice, weather_this_month, best_festival_tip""",
            model=SCOUT, temp=0.2, max_tok=1000
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/budget-plan', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_budget_plan():
    try:
        data        = request.get_json() or {}
        destination = clean(data.get('destination', ''))
        days        = int(data.get('days', 5))
        people      = int(data.get('people', 1))
        budget      = clean(data.get('budget', '50000'))
        currency    = clean(data.get('currency', 'INR'), 3)
        if not destination: return jsonify({'success': False, 'error': 'Destination required'})
        result = groq_json(
            f"""Budget plan for {people} people in {destination} for {days} days. Total: {currency} {budget}.
Return JSON: total_budget, per_person, per_day, budget_tier, breakdown,
daily_budget, money_saving_tips, hidden_costs, free_things, worth_splurging, budget_verdict""",
            model=SCOUT, temp=0.2, max_tok=1000
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/passport-check', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_passport_check():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['expiry_date', 'travel_date', 'destination'])
        if not ok: return jsonify({'success': False, 'error': err})
        from datetime import datetime as dt
        expiry            = dt.strptime(data['expiry_date'], '%Y-%m-%d')
        travel            = dt.strptime(data['travel_date'], '%Y-%m-%d')
        days_after_travel = (expiry - travel).days
        destination       = clean(data.get('destination', ''))
        result = groq_json(
            f"""Passport check for {destination}. Expiry: {data['expiry_date']}. Travel: {data['travel_date']}. Days valid after travel: {days_after_travel}.
Return JSON: is_valid, validity_status, days_remaining, days_after_travel, destination_requirement,
verdict, action_needed, renewal_urgency, renewal_time, renewal_cost, tatkal_available, tatkal_time, tatkal_cost, tips""",
            model=SCOUT, temp=0.1, max_tok=800
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/luggage-check', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_luggage_check():
    try:
        data        = request.get_json() or {}
        airline     = clean(data.get('airline', ''))
        cabin_class = clean(data.get('cabin_class', 'Economy'))
        destination = clean(data.get('destination', ''))
        if not airline or not destination: return jsonify({'success': False, 'error': 'Airline and destination required'})
        result = groq_json(
            f"""Luggage for {airline} to {destination} in {cabin_class}.
Return JSON: airline, cabin_class, carry_on, checked_baggage, prohibited_items,
liquid_rules, duty_free_allowance, packing_tips, pro_tip""",
            model=SCOUT, temp=0.1, max_tok=1000
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/packing', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_packing():
    try:
        data = request.get_json() or {}
        result = groq_json(
            f"""Packing list for {data.get('days',5)} days in {clean(data.get('destination',''))}. Weather: {clean(data.get('weather','moderate'))}. Style: {clean(data.get('vibe','adventure'))}.
Return JSON: essentials, clothing, toiletries, electronics, documents, health, destination_specific"""
        )
        return jsonify({'success': True, 'packing': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── AFTER TRIP TOOLS ──────────────────────────────────────────

@app.route('/api/trip-journal', methods=['POST'])
@require_auth
@limiter.limit("5 per hour")
@check_feature_limit('journal')
def api_trip_journal():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success': False, 'error': err})
        result = groq_json(
            f"""Write vivid personal travel journal. Destination:{clean(data.get('destination',''))}, {data.get('days',5)} days, with {clean(data.get('travel_with','solo'))}, vibe:{clean(data.get('vibe','adventure'))}, highlights:{clean(data.get('highlights','amazing trip'),300)}.
Return JSON (first person): title, tagline, opening, chapters (day/title/story/highlight/emotion/emoji),
closing, best_memory, lesson_learned, quote, would_return, rating, tags""",
            model=MODEL_70B, temp=0.7, max_tok=3000
        )
        trip_id = data.get('trip_id')
        if trip_id: save_journal(trip_id, g.user_id, result)
        execute('UPDATE users SET journal_used_month=journal_used_month+1 WHERE id=%s', (g.user_id,))
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trip-stats', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_trip_stats():
    try:
        data = request.get_json() or {}
        result = groq_json(
            f"""Fun viral trip stats for {clean(data.get('destination',''))}, {data.get('days',5)} days, {clean(data.get('travel_with','solo'))}, vibes:{clean(data.get('vibes','adventure'))}.
Return JSON: traveller_type, traveller_description, fun_stats (label/value/icon),
achievements (title/description/icon/rarity), travel_score, instagram_caption""",
            model=MODEL_70B, temp=0.5, max_tok=1500
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/review-generator', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_review_generator():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['place', 'experience'])
        if not ok: return jsonify({'success': False, 'error': err})
        result = groq_json(
            f"""Genuine {clean(data.get('platform','Google'))} review for {clean(data.get('place',''))}, rated {data.get('rating',5)}/5. Experience: {clean(data.get('experience',''),500)}.
Return JSON: review_title, review_body, pros, cons, best_for, tip, short_version, hashtags""",
            model=MODEL_70B, temp=0.6, max_tok=1000
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/next-trip', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_next_trip():
    try:
        data = request.get_json() or {}
        result = groq_json(
            f"""Next trip suggestions after {clean(data.get('past_destination',''))}. Loved:{clean(data.get('loved',''),200)}. Budget:{clean(data.get('budget',''))}. Month:{clean(data.get('travel_month',''))}. Passport:{clean(data.get('passport','India'))}.
Return JSON: recommendations (destination/why_perfect/similarity_score/best_time/budget_level/estimated_cost/unique_experience/vibe/emoji/visa_for_india),
travel_pattern, bucket_list_suggestion""",
            model=MODEL_70B, temp=0.4, max_tok=1500
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/expense-summary', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_expense_summary():
    try:
        data     = request.get_json() or {}
        expenses = data.get('expenses', [])[:100]
        total    = sum(float(e.get('amount', 0)) for e in expenses)
        by_cat   = {}
        for e in expenses:
            cat = clean(e.get('category', 'Other'), 30)
            by_cat[cat] = by_cat.get(cat, 0) + float(e.get('amount', 0))
        result = groq_json(
            f"""Analyse trip expenses. Destination:{clean(data.get('destination',''))}, budget:{clean(data.get('currency','INR'),3)} {clean(data.get('budget',''))}, spent:{total:.0f}, categories:{by_cat}.
Return JSON: total_spent, budget, status, per_person, verdict, insights, money_tips_next_trip"""
        )
        result['by_category'] = by_cat
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/split-bill', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def api_split_bill():
    try:
        data     = request.get_json() or {}
        people   = data.get('people', [])[:20]
        expenses = data.get('expenses', [])[:100]
        currency = clean(data.get('currency', 'INR'), 3)
        balances = {p: 0 for p in people}
        total    = 0
        for exp in expenses:
            amount        = float(exp.get('amount', 0))
            paid_by       = exp.get('paid_by', people[0] if people else '')
            split_between = exp.get('split_between', people) or people
            total        += amount
            share         = amount / max(len(split_between), 1)
            if paid_by in balances: balances[paid_by] += amount
            for p in split_between:
                if p in balances: balances[p] -= share
        settlements = []
        pos = sorted([(k, v) for k, v in balances.items() if v > 0.01], key=lambda x: -x[1])
        neg = sorted([(k, v) for k, v in balances.items() if v < -0.01], key=lambda x: x[1])
        i = j = 0
        while i < len(pos) and j < len(neg):
            creditor, credit = pos[i]; debtor, debt = neg[j]
            amount = min(credit, -debt)
            if amount > 0.01: settlements.append({'from': debtor, 'to': creditor, 'amount': round(amount, 2), 'currency': currency})
            pos[i] = (creditor, credit - amount); neg[j] = (debtor, debt + amount)
            if pos[i][1] < 0.01: i += 1
            if neg[j][1] > -0.01: j += 1
        return jsonify({'success': True, 'data': {
            'total': round(total, 2), 'per_person': round(total / max(len(people), 1), 2),
            'balances': {k: round(v, 2) for k, v in balances.items()},
            'settlements': settlements, 'currency': currency, 'all_settled': len(settlements) == 0
        }})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/currency-leftover', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_currency_leftover():
    try:
        data = request.get_json() or {}
        result = groq_json(
            f"""Options for {clean(data.get('currency',''),10)} {clean(data.get('amount',''),20)} leftover. Home:{clean(data.get('home_currency','INR'),3)}.
Return JSON: options (option/description/estimated_value/rating/pros/cons), best_option, tips"""
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── MISC ──────────────────────────────────────────────────────

@app.route('/api/detect-theme', methods=['POST'])
@optional_auth
@limiter.limit("60 per hour")
def api_detect_theme():
    try:
        destination = clean((request.get_json() or {}).get('destination', ''))
        result = groq_json(
            f"""Visual theme for travel app destination "{destination}".
Return JSON: destination_type, theme (primary_color/secondary_color/gradient_start/gradient_end/mood/emoji/vibe_words)""",
            temp=0.3, max_tok=400
        )
        return jsonify({'success': True, 'theme': result.get('theme', result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/place-photo', methods=['POST'])
@optional_auth
@limiter.limit("60 per hour")
def api_place_photo():
    try:
        place_name = clean((request.get_json() or {}).get('place_name', ''))
        if not UNSPLASH_KEY or not place_name: return jsonify({'success': False, 'error': 'No key or place'})
        r = req.get(
            "https://api.unsplash.com/search/photos",
            params={'query': f"{place_name} travel landmark", 'per_page': 5, 'orientation': 'landscape', 'client_id': UNSPLASH_KEY},
            timeout=8
        )
        results = r.json().get('results', [])
        photos  = [x['urls']['regular'] for x in results if x.get('urls', {}).get('regular')]
        if photos: return jsonify({'success': True, 'photos': photos})
        return jsonify({'success': False, 'error': 'No photos found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/places-autocomplete')
@limiter.limit("60 per hour")
def places_autocomplete():
    try:
        q   = request.args.get('q', '')
        key = os.getenv('GOOGLE_PLACES_API_KEY', '')
        if not q or not key: return jsonify({'success': False, 'predictions': []})
        r = req.get('https://maps.googleapis.com/maps/api/place/autocomplete/json',
            params={'input': q, 'types': '(cities)', 'key': key, 'language': 'en'}, timeout=8)
        data = r.json()
        if data.get('status') == 'OK': return jsonify({'success': True, 'predictions': data.get('predictions', [])})
        return jsonify({'success': False, 'predictions': []})
    except Exception as e:
        return jsonify({'success': False, 'predictions': [], 'error': str(e)})

@app.route('/api/nearby-stations')
@require_auth
@limiter.limit("30 per hour")
def nearby_stations():
    try:
        location   = request.args.get('location', '')
        place_type = request.args.get('type', 'train_station')
        key        = os.getenv('GOOGLE_PLACES_API_KEY', '')
        if not location or not key: return jsonify({'success': False, 'stations': []})
        geo = req.get('https://maps.googleapis.com/maps/api/geocode/json',
            params={'address': location, 'key': key}, timeout=8).json()
        if not geo.get('results'): return jsonify({'success': False, 'stations': []})
        loc = geo['results'][0]['geometry']['location']
        lat, lng = loc['lat'], loc['lng']
        places = req.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json',
            params={'location': f"{lat},{lng}", 'radius': 50000, 'type': place_type, 'key': key, 'language': 'en'}, timeout=8).json()
        import math
        stations = []
        for p in places.get('results', [])[:6]:
            plat = p['geometry']['location']['lat']; plng = p['geometry']['location']['lng']
            dist_km = round(math.sqrt((plat-lat)**2 + (plng-lng)**2) * 111, 1)
            stations.append({'name': p.get('name',''), 'address': p.get('vicinity',''), 'distance': f"{dist_km} km", 'rating': p.get('rating', 0), 'place_id': p.get('place_id','')})
        stations.sort(key=lambda x: float(x['distance'].replace(' km', '')))
        return jsonify({'success': True, 'stations': stations})
    except Exception as e:
        return jsonify({'success': False, 'stations': [], 'error': str(e)})


# ── CAMERA SCAN ───────────────────────────────────────────────

def _translate_blocks_batch(blocks, target_lang, src_lang=None):
    if not blocks: return []
    texts      = [b['text'] for b in blocks]
    tgt        = target_lang.lower()[:2] if len(target_lang) >= 2 else target_lang
    deepl_code = DEEPL_LANGS.get(tgt) or DEEPL_LANGS.get(target_lang)
    trans_texts = []
    if deepl_code and deepl_client:
        try:
            src = None
            if src_lang and src_lang not in ('unknown', 'auto', None, ''):
                src = src_lang.upper()[:2]
            results     = deepl_client.translate_text(texts, target_lang=deepl_code, source_lang=src)
            trans_texts = [r.text for r in results]
        except Exception as e:
            print(f"[DeepL batch] {e}")
    if not trans_texts:
        lang_name = LANG_NAMES.get(target_lang, LANG_NAMES.get(tgt.upper(), 'English'))
        sep       = "\n[|||]\n"
        combined  = sep.join(texts)
        try:
            r = groq_client.chat.completions.create(
                model=SCOUT,
                messages=[
                    {'role': 'system', 'content': f'Translate each segment to {lang_name}. Keep [|||] separators exactly. Return ONLY translations.'},
                    {'role': 'user', 'content': combined}
                ], temperature=0.1, max_tokens=1500
            )
            parts       = r.choices[0].message.content.strip().split('[|||]')
            trans_texts = [p.strip() for p in parts]
        except Exception as e:
            print(f"[Groq batch] {e}")
            trans_texts = texts
    return [{'original': b['text'], 'translated': trans_texts[i].strip() if i < len(trans_texts) else b['text'], 'vertices': b['vertices']} for i, b in enumerate(blocks)]

@app.route('/scan', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def scan():
    try:
        data        = request.get_json() or {}
        image_data  = data.get('image', '')
        target_lang = clean(data.get('target_lang', 'EN'), 5).upper()
        if ',' in image_data: image_data = image_data.split(',')[1]
        if not image_data: return jsonify({'success': False, 'error': 'No image provided'})
        if len(image_data) > 5 * 1024 * 1024: return jsonify({'success': False, 'error': 'Image too large'})

        vision_result = [None]; groq_result = [None]
        vision_done   = threading.Event(); groq_done = threading.Event()

        def run_vision():
            try:
                if not GOOGLE_VISION_KEY: vision_done.set(); return
                r      = req.post(f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_KEY}",
                    json={"requests": [{"image": {"content": image_data}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]}, timeout=8)
                resp0     = r.json().get('responses', [{}])[0]
                full_text = resp0.get('fullTextAnnotation', {}).get('text', '')
                if not full_text:
                    anns = resp0.get('textAnnotations', [])
                    full_text = anns[0].get('description', '') if anns else ''
                pages  = resp0.get('fullTextAnnotation', {}).get('pages', [])
                lang   = 'unknown'
                if pages:
                    langs = pages[0].get('property', {}).get('detectedLanguages', [])
                    if langs: lang = langs[0].get('languageCode', 'unknown')
                raw_blocks = []
                for page in pages:
                    for block in page.get('blocks', []):
                        parts = []
                        for para in block.get('paragraphs', []):
                            words = []
                            for word in para.get('words', []):
                                w = ''.join(s.get('text', '') for s in word.get('symbols', []))
                                words.append(w)
                            parts.append(' '.join(words))
                        block_text = '\n'.join(parts).strip()
                        if not block_text: continue
                        verts    = block.get('boundingBox', {}).get('vertices', [])
                        vertices = [[v.get('x', 0), v.get('y', 0)] for v in verts]
                        if vertices: raw_blocks.append({'text': block_text, 'vertices': vertices})
                if full_text.strip(): vision_result[0] = (full_text.strip(), lang, raw_blocks)
            except Exception as e: print(f"[Vision] {e}")
            vision_done.set()

        def run_groq_vision():
            try:
                response = groq_client.chat.completions.create(
                    model=SCOUT,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        {"type": "text", "text": "Extract ALL text in this image exactly as written. Preserve line breaks. Return ONLY the raw text. If no text, return NO_TEXT."}
                    ]}], temperature=0.0, max_tokens=800
                )
                text = response.choices[0].message.content.strip()
                if text and text != 'NO_TEXT': groq_result[0] = (text, 'unknown', [])
            except Exception as e: print(f"[Groq Vision] {e}")
            groq_done.set()

        threading.Thread(target=run_vision, daemon=True).start()
        threading.Thread(target=run_groq_vision, daemon=True).start()

        extracted_text = ''; detected_lang = 'unknown'; raw_blocks = []
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if vision_done.is_set() and vision_result[0]:
                extracted_text, detected_lang, raw_blocks = vision_result[0]; break
            if groq_done.is_set() and groq_result[0]:
                extracted_text, detected_lang, raw_blocks = groq_result[0]; break
            time.sleep(0.05)
        if not extracted_text:
            vision_done.wait(2); groq_done.wait(2)
            if vision_result[0]:  extracted_text, detected_lang, raw_blocks = vision_result[0]
            elif groq_result[0]:  extracted_text, detected_lang, raw_blocks = groq_result[0]
        if not extracted_text:
            return jsonify({'success': False, 'error': 'No text found. Try pointing at clearer text.'})

        translated_text, engine = translate(extracted_text, target_lang, detected_lang)
        text_blocks             = _translate_blocks_batch(raw_blocks, target_lang, detected_lang)
        return jsonify({'success': True, 'original_text': extracted_text, 'translated_text': translated_text,
                        'detected_lang': detected_lang, 'engine': engine, 'text_blocks': text_blocks})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/retranslate-blocks', methods=['POST'])
@require_auth
@limiter.limit("60 per hour")
def api_retranslate_blocks():
    try:
        data        = request.get_json() or {}
        blocks      = data.get('blocks', [])
        target_lang = clean(data.get('target_lang', 'EN'), 5).upper()
        src_lang    = data.get('src_lang', None)
        if not blocks: return jsonify({'success': False, 'error': 'No blocks provided'})
        return jsonify({'success': True, 'text_blocks': _translate_blocks_batch(blocks, target_lang, src_lang)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── GROUPS ────────────────────────────────────────────────────

@app.route('/api/groups/sos-broadcast', methods=['POST'])
@require_auth
def api_sos_broadcast():
    try:
        data     = request.get_json() or {}
        sos_type = clean(data.get('sos_type', 'emergency'))
        lat      = float(data.get('lat', 0))
        lng      = float(data.get('lng', 0))
        socketio.emit('sos_alert', {'user_id': g.user_id, 'sos_type': sos_type, 'lat': lat, 'lng': lng, 'message': f'SOS Alert — {sos_type}'}, broadcast=True)
        return jsonify({'success': True, 'message': 'SOS broadcast sent'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── DIARY ─────────────────────────────────────────────────────

@app.route('/api/diary/trips', methods=['GET'])
@require_auth
def api_diary_trips():
    return jsonify({'success': True, 'trips': get_diary_trips(g.user_id)})

@app.route('/api/diary/trips/<int:trip_id>', methods=['GET'])
@require_auth
def api_diary_trip(trip_id):
    trip = get_diary_trip(trip_id, g.user_id)
    if not trip: return jsonify({'success': False, 'error': 'Trip not found'})
    return jsonify({'success': True, 'trip': trip})

@app.route('/api/diary/trips', methods=['POST'])
@require_auth
def api_create_diary_trip():
    try:
        data        = request.get_json() or {}
        destination = clean(data.get('destination', 'My Trip'))
        currency    = clean(data.get('currency', 'INR'), 3)
        start_date  = data.get('start_date', '')
        trip_id     = create_diary_trip(g.user_id, destination, currency, start_date)
        return jsonify({'success': True, 'trip_id': trip_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/diary/entries', methods=['GET'])
@require_auth
def api_diary_entries():
    try:
        trip_id    = request.args.get('trip_id', type=int)
        entry_type = request.args.get('type')
        search     = request.args.get('search')
        limit      = request.args.get('limit', 100, type=int)
        entries    = get_diary_entries(g.user_id, trip_id, entry_type, search, limit)
        return jsonify({'success': True, 'entries': entries})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/diary/entries', methods=['POST'])
@require_auth
@limiter.limit("60 per hour")
def api_create_diary_entry():
    try:
        data       = request.get_json() or {}
        entry_id   = create_diary_entry(
            g.user_id, data.get('trip_id'), clean(data.get('type', 'note'), 20),
            clean(data.get('text', ''), 2000), clean(data.get('mood', ''), 10),
            clean(data.get('location', ''), 100), data.get('tags', [])[:10],
            data.get('photos', [])[:5], float(data.get('amount', 0) or 0),
            clean(data.get('currency', 'INR'), 3), int(data.get('day_number', 1) or 1)
        )
        entries = get_diary_entries(g.user_id, data.get('trip_id'), limit=1)
        entry   = next((e for e in entries if e['id'] == entry_id), None)
        log_action(g.user_id, 'diary_entry', request.remote_addr)
        return jsonify({'success': True, 'entry': entry, 'entry_id': entry_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/diary/entries/<int:entry_id>', methods=['DELETE'])
@require_auth
def api_delete_diary_entry(entry_id):
    delete_diary_entry(entry_id, g.user_id)
    return jsonify({'success': True})

@app.route('/api/diary/entries/<int:entry_id>/favorite', methods=['POST'])
@require_auth
def api_diary_favorite(entry_id):
    try:
        is_fav = toggle_diary_favorite(entry_id, g.user_id)
        return jsonify({'success': True, 'is_favorite': is_fav})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/diary/stats', methods=['GET'])
@require_auth
def api_diary_stats():
    try:
        trip_id = request.args.get('trip_id', type=int)
        stats   = get_diary_stats(g.user_id, trip_id)
        return jsonify({'success': True, **stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/diary/expenses', methods=['GET'])
@require_auth
def api_diary_expenses():
    try:
        trip_id = request.args.get('trip_id', type=int)
        entries = get_diary_entries(g.user_id, trip_id, entry_type='expense')
        total   = sum(e.get('amount', 0) for e in entries)
        by_cat  = {}
        for e in entries:
            cat = e.get('location') or 'Other'
            by_cat[cat] = by_cat.get(cat, 0) + (e.get('amount') or 0)
        by_category = [{'category': k, 'total': v} for k, v in sorted(by_cat.items(), key=lambda x: -x[1])]
        return jsonify({'success': True, 'total': round(total, 2), 'by_category': by_category, 'entries': entries})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/diary/ai-journal', methods=['POST'])
@require_auth
@limiter.limit("5 per hour")
@check_feature_limit('ai_story')
def api_diary_ai_journal():
    try:
        data    = request.get_json() or {}
        trip_id = data.get('trip_id')
        style   = clean(data.get('style', 'storytelling'), 20)
        entries = get_diary_entries(g.user_id, trip_id, limit=50)
        if not entries: return jsonify({'success': False, 'error': 'No diary entries yet'})
        entries_text = '\n'.join([
            f"Day {e.get('day_number',1)} [{e.get('type','note').upper()}] {e.get('location','')} — {e.get('text','')} {('₹'+str(e.get('amount',''))) if e.get('amount') else ''}"
            for e in entries[:20]
        ])
        result = groq_json(
            f"""Transform these travel diary entries into a beautiful literary travel journal.
Style: {style}. Entries: {entries_text}
Return JSON: title, tagline, opening, chapters (day/chapter_title/story/highlight/mood/emoji),
closing, best_memory, lesson_learned, quote, would_return, rating""",
            model=MODEL_70B, temp=0.7, max_tok=3000
        )
        execute('UPDATE users SET ai_story_used=ai_story_used+1 WHERE id=%s', (g.user_id,))
        log_action(g.user_id, 'ai_journal', request.remote_addr)
        return jsonify({'success': True, 'journal': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/diary/export', methods=['GET'])
@require_auth
def api_diary_export():
    try:
        trip_id = request.args.get('trip_id', type=int)
        entries = get_diary_entries(g.user_id, trip_id, limit=500)
        lines   = []
        for e in entries:
            lines.append(f"[Day {e.get('day_number',1)}] [{e.get('type','note').upper()}] {e.get('created_at','')}")
            if e.get('location'): lines.append(f"📍 {e['location']}")
            if e.get('mood'):     lines.append(f"Mood: {e['mood']}")
            lines.append(e.get('text', ''))
            if e.get('amount'):   lines.append(f"💰 ₹{e['amount']}")
            lines.append('─' * 40)
        return Response('\n'.join(lines), mimetype='text/plain',
            headers={'Content-Disposition': 'attachment; filename=yaply_diary.txt'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── PAYMENT + PROMO ───────────────────────────────────────────

@app.route('/api/payment/create-order', methods=['POST'])
@require_auth
def create_payment_order():
    try:
        data        = request.get_json() or {}
        plan        = data.get('plan', 'monthly')
        promo_code  = (data.get('promo_code') or '').upper().strip()
        base_amounts = {'weekly': 9900, 'monthly': 39900}
        amount      = base_amounts.get(plan, 39900)
        discount_amount = 0
        promo_data  = None

        if promo_code:
            promo_data = get_promo_code(promo_code)
            if promo_data and not check_promo_redeemed(g.user_id, promo_code):
                discount_pct    = promo_data['discount_pct']
                discount_amount = int(amount * discount_pct / 100)
                amount          = amount - discount_amount

        amount = max(amount, 100)
        order  = rzp_client.order.create({
            'amount': amount, 'currency': 'INR',
            'receipt': f'yaply_{g.user_id}_{plan}',
            'notes': {'user_id': str(g.user_id), 'plan': plan, 'promo_code': promo_code or ''}
        })
        create_payment(user_id=g.user_id, razorpay_order_id=order['id'], amount=amount,
                       plan=plan, promo_code=promo_code or None, discount_amount=discount_amount)
        return jsonify({
            'success': True, 'order_id': order['id'], 'amount': amount,
            'original_amount': base_amounts.get(plan, 39900),
            'discount_amount': discount_amount, 'currency': 'INR',
            'key': os.getenv('RAZORPAY_KEY_ID'), 'promo_applied': bool(promo_code and promo_data),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/payment/verify', methods=['POST'])
@require_auth
def verify_payment():
    try:
        data       = request.get_json() or {}
        order_id   = data.get('razorpay_order_id')
        payment_id = data.get('razorpay_payment_id')
        signature  = data.get('razorpay_signature')
        plan       = data.get('plan', 'monthly')
        promo_code = (data.get('promo_code') or '').upper().strip()

        rzp_client.utility.verify_payment_signature({
            'razorpay_order_id': order_id,
            'razorpay_payment_id': payment_id,
            'razorpay_signature': signature
        })
        expires_at = activate_pro(g.user_id, plan)
        confirm_payment(order_id, payment_id, signature, expires_at)

        if promo_code:
            promo = get_promo_code(promo_code)
            if promo and not check_promo_redeemed(g.user_id, promo_code):
                redeem_promo(g.user_id, promo_code, discount_amount=promo['discount_pct'], pro_months=promo['pro_months'])

        log_action(g.user_id, f'payment_success_{plan}', request.remote_addr)
        user = get_user_by_id(g.user_id)
        return jsonify({'success': True, 'message': '🎉 Pro activated!', 'plan': plan, 'pro_expires_at': str(user.get('pro_expires_at', ''))})
    except Exception as e:
        log_action(g.user_id if hasattr(g, 'user_id') else 0, 'payment_failed', request.remote_addr)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/promo/validate', methods=['POST'])
def validate_promo():
    try:
        code  = (request.get_json() or {}).get('code', '').upper().strip()
        promo = get_promo_code(code)
        if not promo: return jsonify({'success': False, 'error': 'Invalid code'})
        return jsonify({
            'success': True, 'discount': promo['discount_pct'], 'pro_months': promo['pro_months'],
            'uses_left': promo['max_uses'] - promo['uses'],
            'expires': promo['expires_at'].isoformat() if hasattr(promo['expires_at'], 'isoformat') else promo['expires_at'],
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/promo/apply', methods=['POST'])
@require_auth
def apply_promo():
    try:
        code  = (request.get_json() or {}).get('code', '').upper().strip()
        promo = get_promo_code(code)
        if not promo: return jsonify({'success': False, 'error': 'Invalid promo code'})
        if check_promo_redeemed(g.user_id, code): return jsonify({'success': False, 'error': 'You have already used this code'})
        from datetime import datetime, timedelta
        days    = int(promo['pro_months']) * 30
        expires = datetime.utcnow() + timedelta(days=days)
        execute("UPDATE users SET plan_type='pro', is_pro=TRUE, pro_expires_at=%s WHERE id=%s", (expires, g.user_id))
        redeem_promo(g.user_id, code, discount_amount=promo['discount_pct'], pro_months=promo['pro_months'])
        log_action(g.user_id, 'promo_redeemed_' + code, request.remote_addr)
        return jsonify({'success': True, 'message': f"🎉 {promo['pro_months']} months Pro activated at {promo['discount_pct']}% off!", 'pro_months': promo['pro_months'], 'discount': promo['discount_pct']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── ADMIN ─────────────────────────────────────────────────────

@app.route('/api/admin/stats')
@require_admin
def admin_stats():
    try:
        stats        = admin_get_stats()
        top_dests    = query_all("SELECT destination, COUNT(*) as count FROM trips WHERE deleted_at IS NULL GROUP BY destination ORDER BY count DESC LIMIT 10")
        feature_rows = query_all("SELECT action, COUNT(*) as count FROM usage_logs GROUP BY action ORDER BY count DESC LIMIT 20")
        stats['top_destinations'] = [{'destination': r['destination'], 'count': r['count']} for r in top_dests]
        stats['feature_usage']    = {r['action']: r['count'] for r in feature_rows}
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/users')
@require_admin
def admin_users():
    try:
        return jsonify({'success': True, 'users': admin_get_users()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/trips')
@require_admin
def admin_trips():
    try:
        rows = query_all("SELECT t.*, u.name as user_name, u.email as user_email FROM trips t LEFT JOIN users u ON u.id = t.user_id WHERE t.deleted_at IS NULL ORDER BY t.created_at DESC LIMIT 200")
        trips = []
        for r in rows:
            t = dict(r); t.pop('plan_data', None)
            for k, v in t.items():
                if hasattr(v, 'isoformat'): t[k] = v.isoformat()
            trips.append(t)
        return jsonify({'success': True, 'trips': trips})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/upgrade-pro', methods=['POST'])
@require_admin
def admin_upgrade_pro():
    try:
        email = (request.get_json() or {}).get('email', '').lower()
        if not email: return jsonify({'success': False, 'error': 'Email required'})
        user = get_user_by_email(email)
        if not user: return jsonify({'success': False, 'error': 'User not found'})
        activate_pro(user['id'], 'monthly')
        return jsonify({'success': True, 'message': email + ' upgraded to Pro ✅'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/activity')
@require_admin
def admin_activity():
    try:
        rows = query_all("SELECT l.*, u.name as user_name, u.email as user_email FROM usage_logs l LEFT JOIN users u ON u.id = l.user_id ORDER BY l.created_at DESC LIMIT 200")
        logs = []
        for r in rows:
            row = dict(r)
            for k, v in row.items():
                if hasattr(v, 'isoformat'): row[k] = v.isoformat()
            logs.append(row)
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/promos')
@require_admin
def admin_promos():
    try:
        promos       = query_all('SELECT * FROM promo_codes ORDER BY created_at DESC')
        redemptions  = query_all("SELECT pr.code, u.name, u.email, pr.redeemed_at FROM promo_redemptions pr JOIN users u ON u.id = pr.user_id ORDER BY pr.redeemed_at DESC")
        for p in promos:
            for k, v in p.items():
                if hasattr(v, 'isoformat'): p[k] = v.isoformat()
        for r in redemptions:
            for k, v in r.items():
                if hasattr(v, 'isoformat'): r[k] = v.isoformat()
        return jsonify({'success': True, 'promos': promos, 'redemptions': redemptions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/revenue')
@require_admin
def admin_revenue():
    try:
        daily  = query_all("SELECT DATE(paid_at) as date, COUNT(*) as transactions, SUM(amount)/100.0 as revenue, COUNT(*) FILTER (WHERE plan='weekly') as weekly, COUNT(*) FILTER (WHERE plan='monthly') as monthly FROM payments WHERE status='paid' AND paid_at > NOW() - INTERVAL '30 days' GROUP BY DATE(paid_at) ORDER BY date DESC")
        totals = query_one("SELECT COALESCE(SUM(amount)/100.0,0) as total_revenue, COALESCE(SUM(amount) FILTER (WHERE paid_at > NOW() - INTERVAL '30 days')/100.0,0) as mrr, COALESCE(SUM(amount) FILTER (WHERE DATE(paid_at)=CURRENT_DATE)/100.0,0) as today, COALESCE(SUM(amount) FILTER (WHERE paid_at > NOW() - INTERVAL '7 days')/100.0,0) as week, COUNT(*) as total_payments, COUNT(*) FILTER (WHERE plan='weekly') as weekly_count, COUNT(*) FILTER (WHERE plan='monthly') as monthly_count, COUNT(*) FILTER (WHERE DATE(paid_at)=CURRENT_DATE) as payments_today FROM payments WHERE status='paid'")
        recent = query_all("SELECT p.*, u.name as user_name, u.email as user_email FROM payments p LEFT JOIN users u ON u.id = p.user_id WHERE p.status='paid' ORDER BY p.paid_at DESC LIMIT 50")
        for r in daily + recent:
            for k, v in r.items():
                if hasattr(v, 'isoformat'): r[k] = v.isoformat()
        if totals:
            for k, v in totals.items():
                if hasattr(v, 'isoformat'): totals[k] = v.isoformat()
        return jsonify({'success': True, 'daily': daily, 'totals': totals or {}, 'recent': recent})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/funnel')
@require_admin
def admin_funnel():
    try:
        row     = query_one("SELECT COUNT(*) as total_users, COUNT(*) FILTER (WHERE onboarding_done=TRUE) as onboarded, COUNT(*) FILTER (WHERE id IN (SELECT DISTINCT user_id FROM trips WHERE deleted_at IS NULL)) as planned, COUNT(*) FILTER (WHERE id IN (SELECT DISTINCT user_id FROM payments WHERE status='paid')) as paid, COUNT(*) FILTER (WHERE plan_type='pro') as pro_now FROM users WHERE deleted_at IS NULL")
        signups = query_all("SELECT DATE(created_at) as date, COUNT(*) as count FROM users WHERE deleted_at IS NULL AND created_at > NOW() - INTERVAL '14 days' GROUP BY DATE(created_at) ORDER BY date ASC")
        for r in signups:
            for k, v in r.items():
                if hasattr(v, 'isoformat'): r[k] = v.isoformat()
        return jsonify({'success': True, 'funnel': dict(row or {}), 'signups': signups})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/retention')
@require_admin
def admin_retention():
    try:
        d1  = query_one("SELECT COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE - 1) as cohort, COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE - 1 AND last_active_at > NOW() - INTERVAL '1 day') as retained FROM users WHERE deleted_at IS NULL")
        d7  = query_one("SELECT COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE - 7) as cohort, COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE - 7 AND last_active_at > NOW() - INTERVAL '7 days') as retained FROM users WHERE deleted_at IS NULL")
        d30 = query_one("SELECT COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE - 30) as cohort, COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE - 30 AND last_active_at > NOW() - INTERVAL '30 days') as retained FROM users WHERE deleted_at IS NULL")
        churn = query_all("SELECT u.name, u.email, u.pro_expires_at, u.pro_plan FROM users u WHERE u.plan_type='free' AND u.pro_expires_at IS NOT NULL AND u.pro_expires_at > NOW() - INTERVAL '30 days' AND u.pro_expires_at < NOW() ORDER BY u.pro_expires_at DESC LIMIT 20")
        for r in churn:
            for k, v in r.items():
                if hasattr(v, 'isoformat'): r[k] = v.isoformat()
        def pct(ret, coh): return round(ret/coh*100, 1) if coh else 0
        return jsonify({'success': True, 'retention': {
            'd1':  {'cohort': d1['cohort'] or 0,  'retained': d1['retained'] or 0,  'pct': pct(d1['retained'] or 0,  d1['cohort'] or 0)},
            'd7':  {'cohort': d7['cohort'] or 0,  'retained': d7['retained'] or 0,  'pct': pct(d7['retained'] or 0,  d7['cohort'] or 0)},
            'd30': {'cohort': d30['cohort'] or 0, 'retained': d30['retained'] or 0, 'pct': pct(d30['retained'] or 0, d30['cohort'] or 0)},
        }, 'churn': churn})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/errors')
@require_admin
def admin_errors():
    try:
        errors        = query_all("SELECT action, error, COUNT(*) as count, MAX(created_at) as last_seen, COUNT(DISTINCT user_id) as affected_users FROM usage_logs WHERE success=FALSE AND created_at > NOW() - INTERVAL '7 days' GROUP BY action, error ORDER BY count DESC LIMIT 50")
        recent_errors = query_all("SELECT l.*, u.email as user_email FROM usage_logs l LEFT JOIN users u ON u.id = l.user_id WHERE l.success=FALSE AND l.created_at > NOW() - INTERVAL '24 hours' ORDER BY l.created_at DESC LIMIT 30")
        total_today   = query_one("SELECT COUNT(*) as c FROM usage_logs WHERE success=FALSE AND DATE(created_at)=CURRENT_DATE")
        for r in errors + recent_errors:
            for k, v in r.items():
                if hasattr(v, 'isoformat'): r[k] = v.isoformat()
        return jsonify({'success': True, 'errors': errors, 'recent': recent_errors, 'total_today': (total_today or {}).get('c', 0)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/geographic')
@require_admin
def admin_geographic():
    try:
        cities       = query_all("SELECT home_city, COUNT(*) as count FROM users WHERE deleted_at IS NULL AND home_city != '' GROUP BY home_city ORDER BY count DESC LIMIT 20")
        passports    = query_all("SELECT passport, COUNT(*) as count FROM users WHERE deleted_at IS NULL AND passport != '' GROUP BY passport ORDER BY count DESC LIMIT 20")
        destinations = query_all("SELECT destination, COUNT(*) as count FROM trips WHERE deleted_at IS NULL GROUP BY destination ORDER BY count DESC LIMIT 20")
        return jsonify({'success': True, 'cities': cities, 'passports': passports, 'destinations': destinations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/export-users')
@require_admin
def admin_export_users():
    try:
        import csv, io as sio
        users  = query_all("SELECT u.id, u.name, u.email, u.plan_type, u.is_pro, u.pro_expires_at, u.home_city, u.passport, u.travel_style, u.budget_style, u.onboarding_done, u.created_at, u.last_active_at, COUNT(t.id) as trip_count FROM users u LEFT JOIN trips t ON t.user_id=u.id AND t.deleted_at IS NULL WHERE u.deleted_at IS NULL GROUP BY u.id ORDER BY u.created_at DESC")
        output = sio.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID','Name','Email','Plan','Is Pro','Pro Expires','City','Passport','Travel Style','Budget Style','Onboarding Done','Joined','Last Active','Trips'])
        for u in users:
            writer.writerow([u.get('id',''), u.get('name',''), u.get('email',''), u.get('plan_type',''), u.get('is_pro',''), u.get('pro_expires_at',''), u.get('home_city',''), u.get('passport',''), u.get('travel_style',''), u.get('budget_style',''), u.get('onboarding_done',''), u.get('created_at',''), u.get('last_active_at',''), u.get('trip_count',0)])
        return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=yaply_users.csv'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── PWA + STATIC ──────────────────────────────────────────────

@app.route('/manifest.json')
def manifest():
    return jsonify({
        "name": "Yaply — AI Travel OS", "short_name": "Yaply",
        "description": "India's first AI Travel Operating System.",
        "start_url": "/app", "scope": "/", "display": "standalone",
        "orientation": "portrait", "background_color": "#F7F9FC", "theme_color": "#2563FF",
        "icons": [
            {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
        "shortcuts": [
            {"name": "Plan a Trip", "url": "/plan", "description": "Start planning"},
            {"name": "Live Translate", "url": "/translate", "description": "Live translation"},
        ]
    })

@app.route('/sw.js')
def service_worker():
    response = make_response(open(os.path.join(app.root_path, 'static', 'sw.js')).read())
    response.headers['Content-Type'] = 'application/javascript'
    response.headers['Service-Worker-Allowed'] = '/'
    return response

@app.route('/.well-known/assetlinks.json')
def assetlinks():
    return jsonify([{"relation": ["delegate_permission/common.handle_all_urls"], "target": {"namespace": "android_app", "package_name": "live.yaply.app", "sha256_cert_fingerprints": ["28:A2:F6:0F:1A:F0:D6:B2:D8:C9:BD:B1:CD:35:5B:A1:8B:24:BB:09:17:5B:6D:E7:E0:50:7B:59:B1:CA:79:5D"]}}])


# ── OFFLINE DOWNLOAD ──────────────────────────────────────────

@app.route('/download-itinerary', methods=['POST'])
@require_auth
def download_itinerary():
    try:
        data       = request.get_json() or {}
        plan       = data.get('plan', {})
        multi_city = data.get('multi_city', False)
        html       = _build_multi_city_html(plan) if multi_city else _build_single_city_html(plan)
        dest       = plan.get('destination') or plan.get('trip_title', 'Trip')
        filename   = 'yaply_' + dest.replace(' ', '_').lower() + '_itinerary.html'
        response   = make_response(html)
        response.headers['Content-Type']        = 'text/html; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def _slot(emoji, label, slot):
    if not slot or not slot.get('activity'): return ''
    tip_html = (f'<div style="font-size:11px;color:#1A8A72;margin-top:4px;font-style:italic;border-left:2px solid #1A8A72;padding-left:8px;">💡 {slot.get("tip","")}</div>' if slot.get('tip') else '')
    return (f'<div style="background:#F7F6F2;border-radius:10px;padding:12px;margin-bottom:8px;"><div style="font-size:10px;font-weight:700;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">{emoji} {label}</div><div style="font-weight:700;font-size:14px;">{slot.get("activity","")}</div><div style="font-size:12px;color:#1A8A72;margin-top:2px;">📍 {slot.get("location","")}</div><div style="font-size:12px;font-weight:700;color:#28B06A;margin-top:4px;">💰 {slot.get("cost","")} · ⏱ {slot.get("duration","")}</div>{tip_html}</div>')

def _meal(emoji, label, meal):
    if not meal or not meal.get('restaurant'): return ''
    return (f'<div style="background:#FFF7ED;border-radius:10px;padding:10px 12px;margin-bottom:8px;"><div style="font-size:10px;font-weight:700;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">{emoji} {label}</div><div style="font-weight:700;font-size:14px;">{meal.get("restaurant","")}</div><div style="font-size:12px;color:#6B6860;">🍽️ {meal.get("cuisine","")} · 💰 {meal.get("cost","")}</div></div>')

def _stay(stay):
    if not stay or not stay.get('name'): return ''
    return (f'<div style="background:#EFF6FF;border-radius:10px;padding:10px 12px;margin-bottom:8px;"><div style="font-size:10px;font-weight:700;color:#3A6BC8;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">🏨 Stay</div><div style="font-weight:700;font-size:14px;">{stay.get("name","")}</div><div style="font-size:12px;color:#6B6860;">📍 {stay.get("area","")} · 💰 {stay.get("cost","")}/night</div></div>')

def _build_single_city_html(plan):
    destination = plan.get('destination', 'Your Trip')
    days        = plan.get('days', 0)
    itinerary   = plan.get('itinerary', [])
    packing     = plan.get('packing_list', [])
    phrases     = plan.get('local_phrases', [])
    gems        = plan.get('hidden_gems', [])
    emergency   = plan.get('emergency_numbers', {})
    tips        = plan.get('tips', [])
    cultural    = plan.get('cultural_guide', {})

    day_cards_html = ''
    for day in itinerary:
        day_cards_html += (f'<div style="background:white;border-radius:16px;padding:20px;margin-bottom:12px;border-left:4px solid #1A8A72;box-shadow:0 2px 8px rgba(0,0,0,0.06);"><div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;"><div style="width:36px;height:36px;border-radius:50%;background:#1A8A72;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;color:white;flex-shrink:0;">{day.get("day","")}</div><div style="font-weight:700;font-size:16px;">{day.get("title","")}</div></div>'
            + _slot('🌅','Morning',day.get('morning')) + _meal('☀️','Lunch',day.get('lunch'))
            + _slot('☀️','Afternoon',day.get('afternoon')) + _slot('🌆','Evening',day.get('evening'))
            + _meal('🌙','Dinner',day.get('dinner')) + _stay(day.get('accommodation')) + '</div>')

    packing_html = ''.join(f'<span style="display:inline-block;background:#EFF6FF;color:#1A8A72;border-radius:20px;padding:4px 12px;font-size:12px;margin:3px;font-weight:500;">{item}</span>' for item in packing)
    phrase_html  = ''.join(f'<div style="display:flex;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid #F0EBE0;flex-wrap:wrap;"><span style="font-weight:600;min-width:120px;font-size:13px;">{p.get("phrase","")}</span><span style="color:#1A8A72;font-size:14px;font-weight:600;">{p.get("translation","")}</span><span style="color:#6B6860;font-size:12px;font-style:italic;">({p.get("pronunciation","")})</span></div>' for p in phrases)
    gems_html    = ''.join(f'<div style="background:#F7F6F2;border-radius:12px;padding:14px;margin-bottom:10px;"><div style="font-weight:700;font-size:14px;margin-bottom:4px;">💎 {g.get("name","")}</div><div style="font-size:12px;color:#3D3730;">{g.get("description","")}</div></div>' for g in gems)
    dos          = ''.join(f'<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">✅ {d}</li>' for d in cultural.get('dos', []))
    donts        = ''.join(f'<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">❌ {d}</li>' for d in cultural.get('donts', []))
    em_html      = ''.join(f'<div style="background:#FEF2F2;border-radius:10px;padding:12px;text-align:center;"><div style="font-size:20px;font-weight:700;color:#D84C3E;">{num}</div><div style="font-size:11px;color:#6B6860;margin-top:2px;">{k.replace("_"," ").title()}</div></div>' for k, num in emergency.items())
    tips_html    = ''.join(f'<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">🎯 {t}</li>' for t in tips)

    return f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{destination} — Yaply</title><style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,sans-serif;background:#F7F6F2;color:#2C2B28}}.header{{background:linear-gradient(135deg,#2563FF,#0B1220);color:white;padding:32px 24px;text-align:center}}.container{{max-width:800px;margin:0 auto;padding:20px 16px 60px}}.section{{background:white;border-radius:16px;padding:20px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}.section-title{{font-size:15px;font-weight:800;color:#2563FF;margin-bottom:14px}}.info-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.info-item{{background:#F7F6F2;border-radius:10px;padding:12px}}.info-label{{font-size:10px;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px}}.info-value{{font-size:13px;font-weight:700}}ul{{padding:0;list-style:none}}.footer{{text-align:center;padding:24px;color:#6B6860;font-size:12px}}</style></head><body><div class="header"><div style="font-size:11px;opacity:.6;letter-spacing:2px;margin-bottom:4px;">YAPLY · AI TRAVEL OS</div><h1 style="font-size:28px;font-weight:800;letter-spacing:-1px;">✈️ {destination}</h1><div style="opacity:.8;font-size:13px;margin-top:6px;">{days}-Day Itinerary</div></div><div class="container"><div style="font-size:14px;font-weight:800;color:#2C2B28;margin-bottom:12px;">📅 Day by Day</div>{day_cards_html}{"<div class='section'><div class='section-title'>💎 Hidden Gems</div>" + gems_html + "</div>" if gems_html else ""}{"<div class='section'><div class='section-title'>💬 Local Phrases</div>" + phrase_html + "</div>" if phrase_html else ""}{"<div class='section'><div class='section-title'>🌍 Cultural Guide</div><div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;'><div><div style='font-size:11px;font-weight:700;color:#28B06A;margin-bottom:8px;'>DOS</div><ul>" + dos + "</ul></div><div><div style='font-size:11px;font-weight:700;color:#D84C3E;margin-bottom:8px;'>DON'TS</div><ul>" + donts + "</ul></div></div></div>" if cultural else ""}{"<div class='section'><div class='section-title'>🚨 Emergency Numbers</div><div style='display:grid;grid-template-columns:1fr 1fr;gap:9px;'>" + em_html + "</div></div>" if em_html else ""}{"<div class='section'><div class='section-title'>🎒 Packing</div>" + packing_html + "</div>" if packing_html else ""}{"<div class='section'><div class='section-title'>🎯 Tips</div><ul>" + tips_html + "</ul></div>" if tips_html else ""}</div><div class="footer">Generated by <strong>Yaply</strong> — <a href="https://yaply.live" style="color:#2563FF;">yaply.live</a></div></body></html>'''

def _build_multi_city_html(plan):
    title  = plan.get('trip_title', 'Multi-City Trip')
    cities = plan.get('cities', [])
    return f'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{title} — Yaply</title></head><body><h1>{title}</h1><p>Multi-city itinerary — {len(cities)} cities</p></body></html>'


# ════════════════════════════════════════════════════════════════
#  WEBSOCKETS
# ════════════════════════════════════════════════════════════════

def process_stream(ws, audio_bytes, target_lang, src_lang, sentence_id):
    t_start = time.time()
    try:
        safe_send(ws, {'type': 'status', 'message': '🎯 Listening...'})
        wav              = audio_to_wav(bytes(audio_bytes))
        text, detected, _ = transcribe(wav, src_lang if src_lang != 'auto' else None)
        if not is_valid(text): safe_send(ws, {'type': 'ready'}); return
        safe_send(ws, {'type': 'transcript', 'text': text, 'lang': detected, 'id': sentence_id})
        safe_send(ws, {'type': 'status', 'message': '🌍 Translating...'})
        translated, engine = translate(text, target_lang, detected)
        safe_send(ws, {'type': 'translation', 'text': translated, 'engine': engine, 'lang': target_lang, 'id': sentence_id})
        safe_send(ws, {'type': 'status', 'message': '🔊 Speaking...'})
        audio_data = tts(translated, target_lang)
        safe_send(ws, {'type': 'audio', 'data': base64.b64encode(audio_data).decode(), 'id': sentence_id})
        safe_send(ws, {'type': 'ready'})
        print(f"[Stream #{sentence_id}] {time.time() - t_start:.2f}s")
    except Exception as e:
        safe_send(ws, {'type': 'error', 'message': str(e)})
        safe_send(ws, {'type': 'ready'})

@sock.route('/stream')
def stream_ws(ws):
    target_lang = 'HI'; src_lang = 'auto'
    audio_buffer = bytearray(); silent_chunks = 0; speaking = False
    sentence_id = 0; processing = False
    SILENCE_THRESHOLD = 450; SILENCE_CHUNKS_NEEDED = 2
    MIN_BYTES = int(16000 * 2 * 0.25)
    while True:
        try:
            msg = ws.receive()
            if msg is None: break
            if isinstance(msg, str):
                try:
                    cfg = json.loads(msg)
                    if 'target_lang' in cfg: target_lang = cfg['target_lang']
                    if 'src_lang'    in cfg: src_lang    = cfg['src_lang']
                except: pass
                continue
            chunk = bytes(msg); rms = get_rms(chunk)
            safe_send(ws, {'type': 'volume', 'level': min(100, int(rms / 35))})
            if rms >= SILENCE_THRESHOLD:
                if not speaking: speaking = True; safe_send(ws, {'type': 'speaking', 'status': True})
                silent_chunks = 0; audio_buffer.extend(chunk)
            elif speaking:
                silent_chunks += 1; audio_buffer.extend(chunk)
                if silent_chunks >= SILENCE_CHUNKS_NEEDED:
                    if len(audio_buffer) >= MIN_BYTES and not processing:
                        sentence_id += 1; processing = True; buf_copy = bytearray(audio_buffer)
                        t = threading.Thread(target=process_stream, args=(ws, buf_copy, target_lang, src_lang, sentence_id), daemon=True)
                        t.start(); t.join(); processing = False
                    audio_buffer = bytearray(); silent_chunks = 0; speaking = False
                    safe_send(ws, {'type': 'speaking', 'status': False})
        except Exception as e:
            print(f"[Stream] {e}"); break

@sock.route('/convo-ws')
def convo_ws(ws):
    lang_a = 'en'; lang_b = 'hi'; active_speaker = 'A'
    audio_buffer = bytearray(); silent_chunks = 0; speaking = False; msg_id = 0
    SILENCE_THRESHOLD = 450; MIN_BYTES = int(16000 * 2 * 0.4)
    while True:
        try:
            msg = ws.receive()
            if msg is None: break
            if isinstance(msg, str):
                try:
                    cfg = json.loads(msg)
                    if 'lang_a'  in cfg: lang_a  = cfg['lang_a'].lower()[:2]
                    if 'lang_b'  in cfg: lang_b  = cfg['lang_b'].lower()[:2]
                    if 'speaker' in cfg:
                        active_speaker = cfg['speaker']
                        audio_buffer = bytearray(); silent_chunks = 0; speaking = False
                        safe_send(ws, {'type': 'speaker_changed', 'speaker': active_speaker})
                except: pass
                continue
            chunk = bytes(msg); rms = get_rms(chunk)
            safe_send(ws, {'type': 'volume', 'level': min(100, int(rms / 35)), 'speaker': active_speaker})
            if rms >= SILENCE_THRESHOLD:
                if not speaking: speaking = True; safe_send(ws, {'type': 'speaking', 'status': True, 'speaker': active_speaker})
                silent_chunks = 0; audio_buffer.extend(chunk)
            elif speaking:
                silent_chunks += 1; audio_buffer.extend(chunk)
                src            = lang_a if active_speaker == 'A' else lang_b
                silence_needed = 5 if src in SLOW_LANGS else 3
                if silent_chunks >= silence_needed:
                    if len(audio_buffer) >= MIN_BYTES:
                        msg_id += 1; tgt = lang_b if active_speaker == 'A' else lang_a
                        try:
                            safe_send(ws, {'type': 'status', 'message': '🎯 Listening...'})
                            wav               = audio_to_wav(bytes(audio_buffer))
                            text, detected, _ = transcribe(wav, WHISPER_LANG.get(src))
                            if is_valid(text):
                                safe_send(ws, {'type': 'transcript', 'text': text, 'speaker': active_speaker, 'lang': detected, 'id': msg_id})
                                safe_send(ws, {'type': 'status', 'message': '🌍 Translating...'})
                                translated, engine = translate(text, tgt, src)
                                safe_send(ws, {'type': 'translation', 'text': translated, 'speaker': active_speaker, 'engine': engine, 'id': msg_id})
                                safe_send(ws, {'type': 'status', 'message': '🔊 Speaking...'})
                                audio_data = tts(translated, tgt)
                                safe_send(ws, {'type': 'audio', 'data': base64.b64encode(audio_data).decode(), 'speaker': active_speaker, 'id': msg_id})
                        except Exception as e:
                            safe_send(ws, {'type': 'error', 'message': str(e)})
                        safe_send(ws, {'type': 'ready'})
                    audio_buffer = bytearray(); silent_chunks = 0; speaking = False
                    safe_send(ws, {'type': 'speaking', 'status': False, 'speaker': active_speaker})
        except Exception as e:
            print(f"[Convo] {e}"); break


# ════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'): return jsonify({'success': False, 'error': 'Endpoint not found'}), 404
    return render_template('landing.html'), 404

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({'success': False, 'error': 'Too many requests. Please slow down.'}), 429

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Server error.'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    socketio.run(app, debug=True, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)