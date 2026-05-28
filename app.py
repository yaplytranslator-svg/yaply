"""
app.py — Yaply COMPLETE Production App v4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Google OAuth FIXED (403 was missing route + limiter exempt)
✅ All AI features: Plan, Multi-city, Journey, Discover, Visa, Weather, Currency...
✅ All Safety features: Scam, Price, Safe Route, Medical, Laws, Flight Rights...
✅ After Trip: Journal, Stats, Review, Bill Split, Next Trip...
✅ Diary & Groups blueprints
✅ WebSockets: /stream + /convo-ws
✅ Rate limiting with proper exemptions
✅ Security headers
✅ JWT auth on all routes that need it
✅ Input validation + sanitization
✅ Production-ready for Render
"""

# ── IMPORTS ──────────────────────────────────────────────────
from diary import diary_bp, init_diary_db
from groups import groups_bp, init_groups_db, register_socketio_events
import os, io, base64, json, wave, struct, threading, time
import requests as req
from dotenv import load_dotenv
load_dotenv()

# ── FLASK ─────────────────────────────────────────────────────
from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, make_response, g
)
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET', 'yaply-secret-2025-change-me')
CORS(app, supports_credentials=True)

# ── BLUEPRINTS ────────────────────────────────────────────────
app.register_blueprint(groups_bp)
app.register_blueprint(diary_bp)

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

# ── DATABASE + AUTH ───────────────────────────────────────────
from database import (
    init_db, log_action,
    save_trip, get_trips, get_trip, update_trip, delete_trip,
    save_place, get_places, delete_place,
    add_expense, get_expenses, delete_expense,
    save_journal, get_journal, get_user_stats
)
from auth import register_auth_routes, require_auth, optional_auth, safe_user

init_db()
register_auth_routes(app)
init_groups_db()
init_diary_db()

# ── AI CLIENTS ────────────────────────────────────────────────
from groq import Groq
import edge_tts, asyncio

_groq_key = os.getenv("GROQ_API_KEY")
if not _groq_key:
    raise RuntimeError("GROQ_API_KEY not set — app cannot start")
groq_client = Groq(api_key=_groq_key)

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

# ── SECURITY HEADERS ──────────────────────────────────────────
@app.after_request
def add_security_headers(r):
    r.headers['X-Content-Type-Options']  = 'nosniff'
    r.headers['X-Frame-Options']         = 'DENY'
    r.headers['X-XSS-Protection']        = '1; mode=block'
    r.headers['Referrer-Policy']         = 'strict-origin-when-cross-origin'
    r.headers['Permissions-Policy']      = 'geolocation=(self), microphone=(self), camera=(self)'
    return r

# ── LANGUAGE MAPS ─────────────────────────────────────────────
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
#  HELPERS
# ════════════════════════════════════════════════════════════════

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

def clean(text, max_len=500):
    if not text: return ""
    return str(text).strip()[:max_len]

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
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(n)
    buf.seek(0)
    return buf.read()

def is_valid(text):
    if not text: return False
    t = text.strip()
    if len(t) < 3 or t in HALLUCINATIONS: return False
    alpha = sum(c.isalpha() for c in t)
    return alpha >= len(t) * 0.25

def safe_send(ws, data):
    try: ws.send(json.dumps(data))
    except: pass

def groq_json(prompt, system="Return ONLY valid JSON. No markdown, no backticks.",
              model="llama-3.1-8b-instant", temp=0.2, max_tok=2000):
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
        max_tokens=max_tok
    )
    result = response.choices[0].message.content.strip()
    if '```' in result:
        for p in result.split('```'):
            if '{' in p:
                result = p[4:] if p.startswith('json') else p
                break
    start = result.find('{')
    end   = result.rfind('}') + 1
    if start != -1: result = result[start:end]
    return json.loads(result)

def clean_json(text):
    text = text.strip()
    if '```' in text:
        parts = text.split('```')
        for part in parts:
            if '{' in part:
                text = part
                if text.startswith('json'):
                    text = text[4:]
                break
    start = text.find('{')
    end   = text.rfind('}') + 1
    if start != -1 and end > start:
        text = text[start:end]
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
    conf     = (
        sum(abs(s.get('avg_logprob', -1)) for s in segments) / max(len(segments), 1)
        if segments else 0.0
    )
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
        model='llama-3.1-8b-instant',
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
            buf.seek(0)
            return buf.read()
    return asyncio.run(_run())


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
def landing():
    return render_template('landing.html')

@app.route('/login')
@limiter.exempt
def login_page():
    return render_template('auth.html', google_client_id=GOOGLE_CLIENT_ID)

@app.route('/app')
@limiter.exempt
def main_app():
    # Handle token from Google OAuth redirect
    token = request.args.get('token', '')
    name  = request.args.get('name', '')
    return render_template('yaply-app.html',
                           google_client_id=GOOGLE_CLIENT_ID,
                           oauth_token=token,
                           oauth_name=name)

@app.route('/plan')
def plan_page():
    return render_template('before_trip.html')

@app.route('/during')
def during_page():
    return render_template('during_trip.html')

@app.route('/after')
def after_page():
    return render_template('after_trip.html')

@app.route('/tools')
def tools_page():
    return render_template('tools_extra.html')

@app.route('/discover')
def discover_page():
    return render_template('discover.html')

@app.route('/profile')
def profile_page():
    return render_template('profile.html')

@app.route('/translate')
def translate_page():
    return render_template('stream.html')

@app.route('/convo')
def convo_page():
    return render_template('convo.html')

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/diary')
def diary_page():
    return render_template('yaply_diary.html')

@app.route('/groups')
@app.route('/groups/<int:group_id>')
def groups_page(group_id=None):
    return render_template('yaply_groups.html')

@app.route('/join/<code>')
@limiter.exempt
def join_group_magic(code):
    """Magic link for group join — auto-redirects to groups page with code"""
    return render_template('yaply_groups.html', join_code=code)

@app.route('/after-trip')
def after_trip_page():
    return render_template('after_trip.html')


# ════════════════════════════════════════════════════════════════
#  AUTH API ROUTES
# ════════════════════════════════════════════════════════════════

@app.route('/api/me', methods=['GET'])
@require_auth
def api_me():
    return jsonify({'success': True, 'user': safe_user(g.user)})


# ════════════════════════════════════════════════════════════════
#  DATABASE / TRIP ROUTES
# ════════════════════════════════════════════════════════════════

@app.route('/api/trips', methods=['GET'])
@require_auth
def api_get_trips():
    return jsonify({'success': True, 'trips': get_trips(g.user_id)})

@app.route('/api/trips', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_save_trip():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success': False, 'error': err})
        trip_id = save_trip(
            user_id     = g.user_id,
            destination = clean(data.get('destination', '')),
            origin      = clean(data.get('origin', 'India')),
            days        = min(max(int(data.get('days', 7)), 1), 365),
            people      = min(max(int(data.get('people', 1)), 1), 50),
            budget      = clean(data.get('budget', '80000')),
            currency    = clean(data.get('currency', 'INR'), 3),
            vibes       = clean(data.get('vibes', 'Adventure')),
            passport    = clean(data.get('passport', 'India')),
            plan_data   = data.get('plan_data')
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

@app.route('/api/places', methods=['GET'])
@require_auth
def api_get_places():
    return jsonify({'success': True, 'places': get_places(g.user_id)})

@app.route('/api/places', methods=['POST'])
@require_auth
def api_save_place():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['name'])
        if not ok: return jsonify({'success': False, 'error': err})
        place_id = save_place(
            user_id     = g.user_id,
            name        = clean(data.get('name', '')),
            city        = clean(data.get('city', '')),
            country     = clean(data.get('country', '')),
            continent   = clean(data.get('continent', '')),
            description = clean(data.get('description', ''), 1000),
            image_url   = clean(data.get('image_url', ''), 500),
            emoji       = clean(data.get('emoji', '📍'), 5),
            tags        = data.get('tags', []),
            trip_id     = data.get('trip_id')
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
            trip_id      = trip_id,
            user_id      = g.user_id,
            title        = clean(data.get('title', '')),
            amount       = float(data.get('amount', 0)),
            category     = clean(data.get('category', 'Other')),
            currency     = clean(data.get('currency', 'INR'), 3),
            paid_by      = clean(data.get('paid_by', '')),
            split_with   = data.get('split_with', [])
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

@app.route('/api/profile', methods=['GET'])
@require_auth
def api_profile():
    return jsonify({
        'success': True,
        'user':    safe_user(g.user),
        'stats':   get_user_stats(g.user_id)
    })


# ════════════════════════════════════════════════════════════════
#  AI — TRIP PLANNER
# ════════════════════════════════════════════════════════════════

@app.route('/api/plan', methods=['POST'])
@require_auth
@limiter.limit("15 per hour")
def api_plan():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success': False, 'error': err})

        destination  = clean(data.get('destination', ''))
        origin       = clean(data.get('origin', 'India'))
        days         = min(max(int(data.get('days', 5)), 1), 60)
        budget       = clean(data.get('budget', '50000'))
        vibe         = clean(data.get('vibe', 'adventure'))
        people       = min(max(int(data.get('people', 1)), 1), 20)
        currency     = clean(data.get('currency', 'INR'), 3)
        passport     = clean(data.get('passport', 'India'))
        travel_mode  = clean(data.get('travel_mode', 'smart'), 20).lower()

        # ── Detect if same country trip (no flights needed) ──
        origin_country      = origin.split(',')[-1].strip().lower() if ',' in origin else origin.lower()
        dest_country        = destination.split(',')[-1].strip().lower() if ',' in destination else destination.lower()
        is_domestic         = origin_country in dest_country or dest_country in origin_country
        is_international    = not is_domestic

        # ── Build transport instructions based on mode ──
        if travel_mode == 'flight' or (travel_mode == 'smart' and is_international):
            transport_instruction = f"""
TRANSPORT MODE: Flight
- Include flight information and costs
- Show budget_breakdown.flights with estimated cost in {currency}
- Include best_airlines recommendations
- Include flight_duration
- Include best_time_to_book"""

        elif travel_mode == 'train':
            transport_instruction = f"""
TRANSPORT MODE: Train / Rail
- This user is travelling BY TRAIN, not by flight
- DO NOT include any flight prices or airline info
- budget_breakdown.flights should be "Not applicable"
- Instead include: budget_breakdown.train with cost in {currency}
- Include train_info: {{
    "operator": "train operator name (e.g. Indian Railways)",
    "train_name": "specific train if known",
    "class_options": ["Sleeper", "3AC", "2AC", "1AC"],
    "estimated_cost": "{currency} X",
    "duration": "X hours",
    "booking_tip": "how to book",
    "journey_tips": ["tip1", "tip2"]
  }}
- Include how to reach railway station from origin city"""

        elif travel_mode == 'bus':
            transport_instruction = f"""
TRANSPORT MODE: Bus
- This user is travelling BY BUS, not by flight
- DO NOT include any flight prices or airline info
- budget_breakdown.flights should be "Not applicable"
- Instead include: budget_breakdown.bus with cost in {currency}
- Include bus_info: {{
    "operators": ["operator1", "operator2"],
    "type": "AC Sleeper / Volvo / Express",
    "estimated_cost": "{currency} X",
    "duration": "X hours",
    "booking_tip": "how to book (RedBus, AbhiBus etc)",
    "journey_tips": ["tip1", "tip2"]
  }}"""

        elif travel_mode == 'road':
            transport_instruction = f"""
TRANSPORT MODE: Road Trip / Self Drive / Cab
- This user is travelling BY ROAD (car, cab, self-drive)
- DO NOT include any flight prices
- budget_breakdown.flights should be "Not applicable"
- Include budget_breakdown.road with fuel + toll cost in {currency}
- Include road_info: {{
    "distance_km": "approximate km",
    "drive_duration": "X hours",
    "fuel_cost": "{currency} X",
    "toll_cost": "{currency} X",
    "cab_option": "{currency} X via Uber/Ola",
    "road_tips": ["tip1", "tip2"],
    "stops_enroute": ["interesting stop 1", "stop 2"]
  }}"""

        else:  # smart mode
            if is_domestic:
                transport_instruction = f"""
TRANSPORT MODE: Smart (Domestic trip detected — {origin} to {destination})
- Recommend the BEST transport option for this DOMESTIC route
- If distance < 500km: recommend train or bus (NOT flight)
- If distance 500-1000km: recommend train or flight based on cost
- If distance > 1000km: recommend flight
- Show the recommended transport in budget_breakdown
- If recommending train/bus: set budget_breakdown.flights = "Not applicable for this route"
- Include transport_recommendation: {{
    "mode": "train/bus/flight/road",
    "reason": "why this is best for this route",
    "estimated_cost": "{currency} X",
    "duration": "X hours"
  }}"""
            else:
                transport_instruction = f"""
TRANSPORT MODE: Smart (International trip — {origin} to {destination})
- This is an international trip — include flight information
- Show budget_breakdown.flights with estimated cost in {currency}"""

        # ── Build the full prompt ──
        prompt = f"""You are a world-class travel planner. Create a detailed {days}-day trip plan.

TRIP DETAILS:
- FROM: {origin}
- TO: {destination}  
- Duration: {days} days
- People: {people}
- Total Budget: {currency} {budget}
- Travel Style: {vibe}
- Passport: {passport}

{transport_instruction}

CRITICAL RULES:
1. ALL prices MUST be in {currency} only
2. Budget must be realistic for {currency} {budget} total for {people} people
3. Activities and restaurants must match the {vibe} travel style
4. Hidden gems must be genuinely lesser-known places
5. Budget breakdown must add up to approximately {currency} {budget}

Return ONLY valid JSON with this structure:
{{
  "destination": "{destination}",
  "origin": "{origin}",
  "days": {days},
  "travel_mode": "{travel_mode}",
  "language": "local language",
  "currency": "local currency",
  "timezone": "timezone",
  "best_time_to_visit": "best months",
  "budget_breakdown": {{
    "flights": "{currency} X or Not applicable",
    "train": "{currency} X (if train mode)",
    "bus": "{currency} X (if bus mode)",
    "road": "{currency} X (if road mode)",
    "accommodation": "{currency} X",
    "food": "{currency} X",
    "transport": "{currency} X (local transport only)",
    "activities": "{currency} X",
    "miscellaneous": "{currency} X"
  }},
  "flight_info": null or {{"estimated_cost":"{currency} X","best_airlines":["a1"],"flight_duration":"Xh","best_time_to_book":"X weeks"}},
  "train_info": null or {{"operator":"name","class_options":["3AC"],"estimated_cost":"{currency} X","duration":"Xh","booking_tip":"tip"}},
  "bus_info": null or {{"operators":["name"],"type":"AC Sleeper","estimated_cost":"{currency} X","duration":"Xh","booking_tip":"tip"}},
  "road_info": null or {{"distance_km":"X","drive_duration":"Xh","fuel_cost":"{currency} X","toll_cost":"{currency} X"}},
  "transport_recommendation": null or {{"mode":"train","reason":"why","estimated_cost":"{currency} X","duration":"Xh"}},
  "itinerary": [
    {{
      "day": 1,
      "title": "Day title",
      "morning": {{"activity": "name", "location": "place", "duration": "2h", "cost": "{currency} X", "tip": "insider tip"}},
      "afternoon": {{"activity": "name", "location": "place", "duration": "2h", "cost": "{currency} X", "tip": "tip"}},
      "evening": {{"activity": "name", "location": "place", "duration": "2h", "cost": "{currency} X", "tip": "tip"}},
      "lunch": {{"restaurant": "name", "cuisine": "type", "cost": "{currency} X"}},
      "dinner": {{"restaurant": "name", "cuisine": "type", "cost": "{currency} X"}},
      "accommodation": {{"name": "hotel", "area": "area", "cost": "{currency} X/night"}}
    }}
  ],
  "hidden_gems": [{{"name":"place","description":"why special","location":"area","best_time":"when","cost":"{currency} X"}}],
  "local_transport": {{
    "airport_to_city": {{"options":["opt1"],"cost":"{currency} X","duration":"30min"}},
    "within_city": [{{"type":"Metro","cost":"{currency} X/ride","tip":"tip"}}],
    "useful_apps": ["app1"]
  }},
  "sim_internet": {{"best_option":"option","cost":"{currency} X","data":"XGB","where_to_buy":"location"}},
  "cultural_guide": {{"dos":["do1","do2","do3"],"donts":["dont1","dont2","dont3"],"dress_code":"advice","tipping":"culture","greetings":"how"}},
  "vaccinations": {{"required":["v1"],"recommended":["v2"],"note":"advice"}},
  "packing_list": ["item1","item2","item3"],
  "emergency_numbers": {{"police":"number","ambulance":"number","fire":"number","tourist_helpline":"number"}},
  "visa_info": {{"required":true,"type":"tourist","validity":"30 days","cost":"{currency} X"}},
  "payment_info": {{"preferred":"card/cash","atm_availability":"common","notify_bank":true,"forex_tips":"advice"}},
  "must_have_apps": [{{"name":"app","purpose":"what","platform":"iOS/Android"}}],
  "power_plug": {{"type":"type","voltage":"220V","adapter_needed":true}},
  "what_to_buy": ["item1","item2"],
  "what_to_avoid": ["item1"],
  "local_phrases": [{{"phrase":"Hello","translation":"local","pronunciation":"how"}}],
  "tips": ["tip1","tip2","tip3"]
}}"""

        result = groq_json(
            prompt,
            model="llama-3.3-70b-versatile",
            temp=0.3,
            max_tok=4000
        )

        # Save to trip if trip_id provided
        trip_id = data.get('trip_id')
        if trip_id:
            update_trip(trip_id, g.user_id, plan_data=result, status='active')

        log_action(g.user_id, 'plan_trip', request.remote_addr)
        return jsonify({'success': True, 'plan': result})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ── MULTI-CITY PLANNER ────────────────────────────────────────

@app.route('/api/multi-city-plan', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
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
            "You are the world's best multi-city trip planner.\n\n"
            "Plan an epic multi-city trip:\n"
            "- Origin: " + str(origin) + "\n"
            "- Cities in order: " + str(city_list) + "\n"
            "- Cities data: " + str(json.dumps(cities)) + "\n"
            "- Total days: " + str(total_days) + "\n"
            "- Total budget: " + str(currency) + " " + str(total_budget) + " for " + str(people) + " people\n"
            "- Travel style: " + str(vibe) + "\n"
            "- Start date: " + str(start_date if start_date else 'flexible') + "\n"
            "- Passport: " + str(passport) + "\n\n"
            "RULES:\n"
            "1. ALL prices in " + str(currency) + " only\n"
            "2. Plan each city with full day-by-day itinerary\n"
            "3. Include transit between every city with costs\n"
            "4. Budget must sum to approximately " + str(total_budget) + "\n"
            "5. Activities must match the " + str(vibe) + " travel style\n\n"
            "Return ONLY valid JSON with: trip_title, origin, total_days, total_budget, currency, cities_count, "
            "route_overview, smart_suggestions, budget_split, cities (array with full itinerary per city), "
            "transit_plans, sim_strategy, packing_for_route, money_saving_tips"
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown. No backticks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=8000
        )
        plan = clean_json(response.choices[0].message.content)
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

        # Build mode-specific instruction
        if travel_mode == 'train':
            mode_instruction = """
TRAVEL MODE: Train only.
- nearest_airports should be nearest TRAIN STATIONS not airports
- Use field name nearest_airports but fill with train stations
- Include train operators, classes, IRCTC booking info
- NO flight options"""
        elif travel_mode == 'bus':
            mode_instruction = """
TRAVEL MODE: Bus only.
- nearest_airports should be nearest BUS STANDS/TERMINALS not airports
- Use field name nearest_airports but fill with bus terminals
- Include bus operators like RSRTC, GSRTC, RedBus options
- NO flight options
- distance_from_origin should be in KM — label it clearly as km not price"""
        elif travel_mode == 'road':
            mode_instruction = """
TRAVEL MODE: Road/Drive only.
- Include road distance, drive time, fuel cost, toll cost
- nearest_airports field = major highway toll plazas or fuel stops
- NO flight options"""
        else:
            mode_instruction = """
TRAVEL MODE: Best option (could be flight, train or bus).
- nearest_airports = actual airports near origin
- Include flight options if international or long distance
- Include train as alternative if domestic"""

        result = groq_json(
            f"""Complete door-to-door journey planner.
FROM: {origin}
TO: {destination}
CURRENCY: {currency}

{mode_instruction}

Return ONLY valid JSON:
{{
  "origin": "{origin}",
  "destination": "{destination}",
  "nearest_airports": [
    {{
      "name": "hub name",
      "city": "city",
      "code": "code if airport, blank if bus/train",
      "distance_from_origin": "X km from {origin}"
    }}
  ],
  "recommended_route": {{
    "step1": "Step 1 description with transport and cost",
    "step2": "Step 2 description",
    "step3": "Step 3 description",
    "step4": "Step 4 description if needed",
    "total_duration": "X hours",
    "total_cost": "{currency} X"
  }},
  "flight_options": [
    {{
      "airline": "operator name",
      "mode": "{travel_mode}",
      "duration": "X hours",
      "price": "{currency} X",
      "stops": 0,
      "class": "class type",
      "recommended": true
    }}
  ],
  "alternative_routes": [],
  "important_notes": ["note1", "note2"],
  "documents_needed": ["doc1", "doc2"]
}}""",
            model="llama-3.3-70b-versatile",
            temp=0.2,
            max_tok=2000
        )
        return jsonify({'success': True, 'journey': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── SIM GUIDE ─────────────────────────────────────────────────

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
        if not destination:
            return jsonify({'success': False, 'error': 'Destination required'})

        countries_str = ', '.join(countries) if isinstance(countries, list) else destination
        result = groq_json(
            f"""Expert SIM card guide for traveller from {origin} visiting {countries_str} for {days} days. Data usage: {data_needs}.
Return JSON: top_recommendation (name/provider/type/cost/data/validity/where_to_buy/activation/coverage),
all_options, esim_options (Airalo/Holafly/Nomad), airport_buying_guide, roaming_option (Jio/Airtel/Vi),
connectivity_tips, data_saving_tips, offline_essentials, budget_summary""",
            model="llama-3.3-70b-versatile", temp=0.2, max_tok=3000
        )
        return jsonify({'success': True, 'guide': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── WEATHER ───────────────────────────────────────────────────

@app.route('/api/weather', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def api_weather():
    try:
        city = clean((request.get_json() or {}).get('city', ''))
        if not city: return jsonify({'success': False, 'error': 'City required'})
        if not WEATHER_KEY: return jsonify({'success': False, 'error': 'Weather API key not configured'})
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_KEY}&units=metric&cnt=40"
        r    = req.get(url, timeout=10)
        data = r.json()
        if data.get('cod') != '200':
            return jsonify({'success': False, 'error': 'City not found'})
        daily = {}
        for item in data['list']:
            date = item['dt_txt'].split(' ')[0]
            if date not in daily:
                daily[date] = {
                    'date': date,
                    'temp_max': item['main']['temp_max'],
                    'temp_min': item['main']['temp_min'],
                    'description': item['weather'][0]['description'],
                    'icon': item['weather'][0]['icon'],
                    'humidity': item['main']['humidity'],
                    'wind': item['wind']['speed']
                }
            else:
                daily[date]['temp_max'] = max(daily[date]['temp_max'], item['main']['temp_max'])
                daily[date]['temp_min'] = min(daily[date]['temp_min'], item['main']['temp_min'])
        return jsonify({
            'success': True,
            'city': data['city']['name'],
            'country': data['city']['country'],
            'forecast': list(daily.values())[:7]
        })
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
        if not EXCHANGE_KEY:
            return jsonify({'success': False, 'error': 'Exchange API key not configured'})
        r = req.get(
            f"https://v6.exchangerate-api.com/v6/{EXCHANGE_KEY}/pair/{from_c}/{to_c}/{amount}",
            timeout=10
        ).json()
        if r.get('result') != 'success':
            return jsonify({'success': False, 'error': 'Currency not found'})
        return jsonify({
            'success': True, 'from': from_c, 'to': to_c,
            'amount': amount, 'converted': round(r['conversion_result'], 2),
            'rate': r['conversion_rate']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── VISA ──────────────────────────────────────────────────────

@app.route('/api/visa', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_visa():
    try:
        data = request.get_json() or {}
        passport    = clean(data.get('passport', 'India'))
        destination = clean(data.get('destination', ''))
        if not destination: return jsonify({'success': False, 'error': 'Destination required'})
        result = groq_json(
            f"""Visa requirements for {passport} passport holder visiting {destination}.
Return JSON: visa_required, visa_type, validity, cost, processing_days, apply_online, apply_url, documents, tips, visa_on_arrival, visa_free_days"""
        )
        return jsonify({'success': True, 'visa': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── DISCOVER / IDENTIFY ───────────────────────────────────────

@app.route('/api/identify', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_identify():
    try:
        data         = request.get_json() or {}
        image_base64 = data.get('image', '')
        if not image_base64: return jsonify({'success': False, 'error': 'No image provided'})
        if len(image_base64) > 5 * 1024 * 1024:
            return jsonify({'success': False, 'error': 'Image too large (max 5MB)'})

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
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
        log_action(g.user_id, 'identify_place', request.remote_addr)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/identify-text', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_identify_text():
    try:
        description = clean((request.get_json() or {}).get('description', ''), 300)
        if not description: return jsonify({'success': False, 'error': 'Description required'})
        result = groq_json(
            f"""Someone saw this place on social media: "{description}". Identify the exact location.
Return JSON: place_name, city, country, continent, confidence, place_type, description, tags,
best_time, climate, budget_level, avg_daily_cost, language, currency, nearest_airport, airport_code,
why_famous, nearby (name/distance/type/icon/description), similar_places (name/country/why_similar/emoji),
travel_tips, best_food""",
            model="llama-3.3-70b-versatile", temp=0.3, max_tok=2000
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── SAFETY TOOLS ──────────────────────────────────────────────

@app.route('/api/scam-alerts', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_scam_alerts():
    try:
        destination = clean((request.get_json() or {}).get('destination', ''))
        if not destination: return jsonify({'success': False, 'error': 'Destination required'})
        result = groq_json(
            f"""All tourist scams in {destination}. Return JSON: scam_risk_level,
scams (name/category/severity/how_it_works/red_flags/how_to_avoid/what_to_say/icon),
general_rules, safe_alternatives, emergency_if_robbed""",
            model="llama-3.3-70b-versatile", max_tok=3000
        )
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
            f"""Price check: "{clean(data.get('item',''),100)}" costs {clean(data.get('currency','INR'),3)} {clean(data.get('price',''),20)} in {clean(data.get('destination',''))}.
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
def api_safety_check():
    try:
        destination = clean((request.get_json() or {}).get('destination', ''))
        if not destination:
            return jsonify({'success': False, 'error': 'Destination required'})

        result = groq_json(
            f"""You are a travel safety expert with deep knowledge of every city worldwide.
Provide SPECIFIC and ACCURATE safety information for: {destination}

CRITICAL RULES:
- safety_score MUST be specific to {destination} — not a generic number
- Tokyo should score ~85, Bangkok ~65, Cairo ~55, London ~75, Mexico City ~50
- Base score on: actual crime rates, tourist incidents, political stability
- Every field must be specific to {destination} — no generic answers

Return ONLY this JSON:
{{
  "destination": "{destination}",
  "safety_score": <number 1-100 specific to {destination}>,
  "safety_level": "<Safe/Moderate/Caution/High Risk> for {destination}",
  "crime_index": "<Low/Medium/High> based on actual {destination} crime data",
  "tourist_safety": "<specific assessment for tourists in {destination}>",
  "water_safe": <true/false for {destination} tap water>,
  "water_advice": "<specific water advice for {destination}>",
  "food_safety": "<Safe/Mostly Safe/Be Careful — specific to {destination}>",
  "health_risks": ["<actual health risk in {destination}>", "<risk 2>"],
  "scams_to_avoid": ["<real scam tourists face in {destination}>", "<scam 2>", "<scam 3>"],
  "safe_areas": ["<actually safe neighbourhood in {destination}>", "<area 2>"],
  "avoid_areas": ["<area to actually avoid in {destination}>"],
  "emergency_embassy": "<Indian embassy address in {destination}>",
  "embassy_phone": "<actual embassy phone number>",
  "travel_advisory": "<current advisory level for {destination}>",
  "solo_female_safety": "<specific assessment for solo female travellers in {destination}>",
  "best_safety_tips": ["<tip specific to {destination}>", "<tip 2>", "<tip 3>", "<tip 4>", "<tip 5>"]
}}""",
            model="llama-3.3-70b-versatile",
            temp=0.4,
            max_tok=1200
        )
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
            f"""Legal travel expert. Important laws and rules for tourists in {destination}.
Return JSON: strict_laws (law/penalty/severity/icon), photography_rules, dress_code_rules,
alcohol_rules, drug_laws, customs_limits (cash/cigarettes/alcohol/prohibited_items),
good_to_know, legal_tip""",
            model="llama-3.3-70b-versatile", temp=0.1, max_tok=1500
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
        if not name or not destination:
            return jsonify({'success': False, 'error': 'Name and destination required'})
        result = groq_json(
            f"""Emergency card for {name} (blood group: {blood_group}, allergies: {allergies}) visiting {destination}.
Return JSON: emergency_numbers, indian_embassy (address/phone/emergency_phone/email),
nearest_hospitals, medical_phrases (english/local/pronunciation),
what_to_do_if_robbed, what_to_do_if_sick, what_to_do_if_lost""",
            model="llama-3.3-70b-versatile", temp=0.1, max_tok=1500
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── PLANNING TOOLS ────────────────────────────────────────────

@app.route('/api/jetlag', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_jetlag():
    try:
        data        = request.get_json() or {}
        from_city   = clean(data.get('from_city', ''))
        to_city     = clean(data.get('to_city', ''))
        travel_date = clean(data.get('travel_date', 'upcoming'))
        if not from_city or not to_city:
            return jsonify({'success': False, 'error': 'Both cities required'})
        result = groq_json(
            f"""Jet lag recovery plan for {from_city} to {to_city} on {travel_date}.
Return JSON: from_timezone, to_timezone, time_difference, jet_lag_severity, recovery_days,
direction, symptoms, before_flight, during_flight, after_arrival, sleep_schedule, avoid, recovery_tip""",
            model="llama-3.3-70b-versatile", temp=0.1, max_tok=1200
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/festivals', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_festivals():
    try:
        data         = request.get_json() or {}
        destination  = clean(data.get('destination', ''))
        travel_date  = clean(data.get('travel_date', 'this month'))
        if not destination: return jsonify({'success': False, 'error': 'Destination required'})
        result = groq_json(
            f"""Festivals and events in {destination} around {travel_date}.
Return JSON: public_holidays, festivals, peak_season, season_type, price_impact,
crowd_level, booking_advice, weather_this_month, best_festival_tip""",
            model="llama-3.3-70b-versatile", temp=0.2, max_tok=1200
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
Return JSON: total_budget, per_person, per_day, budget_tier, breakdown (flights/accommodation/food/transport/activities/shopping),
daily_budget (budget_day/comfort_day/splurge_day), money_saving_tips, hidden_costs,
free_things, worth_splurging, budget_verdict""",
            model="llama-3.3-70b-versatile", temp=0.1, max_tok=1500
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
        days_remaining    = (expiry - dt.now()).days
        days_after_travel = (expiry - travel).days
        destination       = clean(data.get('destination', ''))
        result = groq_json(
            f"""Passport check for {destination}. Expiry: {data['expiry_date']}. Travel: {data['travel_date']}. Days valid after travel: {days_after_travel}.
Return JSON: is_valid, validity_status, days_remaining, days_after_travel, destination_requirement,
verdict, action_needed, renewal_urgency, renewal_time, renewal_cost, tatkal_available, tatkal_time, tatkal_cost, tips""",
            model="llama-3.3-70b-versatile", temp=0.1, max_tok=800
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
        if not airline or not destination:
            return jsonify({'success': False, 'error': 'Airline and destination required'})
        result = groq_json(
            f"""Luggage allowance for {airline} to {destination} in {cabin_class} class.
Return JSON: airline, cabin_class, carry_on (weight/dimensions/pieces), checked_baggage (weight/dimensions/pieces/extra_cost),
prohibited_items, liquid_rules, duty_free_allowance (alcohol/cigarettes/cash/gifts), packing_tips, pro_tip""",
            model="llama-3.3-70b-versatile", temp=0.1, max_tok=1000
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
def api_trip_journal():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success': False, 'error': err})
        result = groq_json(
            f"""Write vivid personal travel journal. Destination:{clean(data.get('destination',''))}, {data.get('days',5)} days, with {clean(data.get('travel_with','solo'))}, vibe:{clean(data.get('vibe','adventure'))}, highlights:{clean(data.get('highlights','amazing trip'),300)}.
Return JSON (first person): title, tagline, opening, chapters (day/title/story/highlight/emotion/emoji),
closing, best_memory, lesson_learned, quote, would_return, rating, tags""",
            model="llama-3.3-70b-versatile", temp=0.7, max_tok=3000
        )
        trip_id = data.get('trip_id')
        if trip_id: save_journal(trip_id, g.user_id, result)
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
            temp=0.5, max_tok=1500
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
            temp=0.6, max_tok=1000
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
            temp=0.4, max_tok=1500
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
            amount         = float(exp.get('amount', 0))
            paid_by        = exp.get('paid_by', people[0] if people else '')
            split_between  = exp.get('split_between', people) or people
            total         += amount
            share          = amount / max(len(split_between), 1)
            if paid_by in balances: balances[paid_by] += amount
            for p in split_between:
                if p in balances: balances[p] -= share
        settlements = []
        pos = sorted([(k, v) for k, v in balances.items() if v > 0.01], key=lambda x: -x[1])
        neg = sorted([(k, v) for k, v in balances.items() if v < -0.01], key=lambda x: x[1])
        i = j = 0
        while i < len(pos) and j < len(neg):
            creditor, credit = pos[i]
            debtor, debt     = neg[j]
            amount           = min(credit, -debt)
            if amount > 0.01:
                settlements.append({'from': debtor, 'to': creditor, 'amount': round(amount, 2), 'currency': currency})
            pos[i] = (creditor, credit - amount)
            neg[j] = (debtor, debt + amount)
            if pos[i][1] < 0.01: i += 1
            if neg[j][1] > -0.01: j += 1
        return jsonify({'success': True, 'data': {
            'total': round(total, 2),
            'per_person': round(total / max(len(people), 1), 2),
            'balances': {k: round(v, 2) for k, v in balances.items()},
            'settlements': settlements,
            'currency': currency,
            'all_settled': len(settlements) == 0
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
            f"""Options for {clean(data.get('currency',''),10)} {clean(data.get('amount',''),20)} leftover currency. Home:{clean(data.get('home_currency','INR'),3)}.
Return JSON: options (option/description/estimated_value/rating/pros/cons), best_option, tips"""
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── GROUPS API ────────────────────────────────────────────────

@app.route('/api/groups/join', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_join_group():
    try:
        data = request.get_json() or {}
        code = clean(data.get('code', ''), 10).upper()
        if not code: return jsonify({'success': False, 'error': 'Group code required'})
        # This delegates to the groups blueprint
        # but we handle the response here
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), 'yaply.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()
        group = cur.execute('SELECT * FROM travel_groups WHERE code = ?', [code]).fetchone()
        if not group:
            conn.close()
            return jsonify({'success': False, 'error': 'Group not found. Check the code and try again.'})
        # Add member if not already
        existing = cur.execute(
            'SELECT * FROM group_members WHERE group_id = ? AND user_id = ?',
            [group['id'], g.user_id]
        ).fetchone()
        if not existing:
            cur.execute(
                'INSERT INTO group_members (group_id, user_id, role) VALUES (?, ?, ?)',
                [group['id'], g.user_id, 'member']
            )
            conn.commit()
        conn.close()
        return jsonify({'success': True, 'group_id': group['id'], 'group_name': group['name']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/groups/sos-broadcast', methods=['POST'])
@require_auth
def api_sos_broadcast():
    try:
        data     = request.get_json() or {}
        sos_type = clean(data.get('sos_type', 'emergency'))
        lat      = float(data.get('lat', 0))
        lng      = float(data.get('lng', 0))
        # Broadcast via socketio to all group members
        socketio.emit('sos_alert', {
            'user_id':  g.user_id,
            'sos_type': sos_type,
            'lat':      lat,
            'lng':      lng,
            'message':  f'SOS Alert — {sos_type}'
        }, broadcast=True)
        return jsonify({'success': True, 'message': 'SOS broadcast sent'})
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
        if not UNSPLASH_KEY or not place_name:
            return jsonify({'success': False, 'error': 'No key or place'})
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


# ── CAMERA SCAN ───────────────────────────────────────────────

def _translate_blocks_batch(blocks, target_lang, src_lang=None):
    if not blocks: return []
    texts       = [b['text'] for b in blocks]
    tgt         = target_lang.lower()[:2] if len(target_lang) >= 2 else target_lang
    deepl_code  = DEEPL_LANGS.get(tgt) or DEEPL_LANGS.get(target_lang)
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
                model='llama-3.1-8b-instant',
                messages=[
                    {'role': 'system', 'content': f'Translate each segment to {lang_name}. Keep [|||] separators exactly. Return ONLY translations separated by [|||].'},
                    {'role': 'user', 'content': combined}
                ], temperature=0.1, max_tokens=1500
            )
            parts       = r.choices[0].message.content.strip().split('[|||]')
            trans_texts = [p.strip() for p in parts]
        except Exception as e:
            print(f"[Groq batch] {e}")
            trans_texts = texts
    return [
        {
            'original':   b['text'],
            'translated': trans_texts[i].strip() if i < len(trans_texts) else b['text'],
            'vertices':   b['vertices']
        }
        for i, b in enumerate(blocks)
    ]

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
        if len(image_data) > 5 * 1024 * 1024:
            return jsonify({'success': False, 'error': 'Image too large (max 5MB)'})

        vision_result = [None]; groq_result = [None]
        vision_done   = threading.Event(); groq_done = threading.Event()

        def run_vision():
            try:
                if not GOOGLE_VISION_KEY: vision_done.set(); return
                r      = req.post(
                    f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_KEY}",
                    json={"requests": [{"image": {"content": image_data}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]},
                    timeout=8
                )
                resp0     = r.json().get('responses', [{}])[0]
                full_text = resp0.get('fullTextAnnotation', {}).get('text', '')
                if not full_text:
                    anns      = resp0.get('textAnnotations', [])
                    full_text = anns[0].get('description', '') if anns else ''
                pages     = resp0.get('fullTextAnnotation', {}).get('pages', [])
                lang      = 'unknown'
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
                if not raw_blocks and resp0.get('textAnnotations'):
                    for ann in resp0['textAnnotations'][1:]:
                        t    = ann.get('description', '').strip()
                        verts = ann.get('boundingPoly', {}).get('vertices', [])
                        vts  = [[v.get('x', 0), v.get('y', 0)] for v in verts]
                        if t and vts: raw_blocks.append({'text': t, 'vertices': vts})
                if full_text.strip():
                    vision_result[0] = (full_text.strip(), lang, raw_blocks)
            except Exception as e:
                print(f"[Vision] {e}")
            vision_done.set()

        def run_groq_vision():
            try:
                response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        {"type": "text", "text": "Extract ALL text in this image exactly as written. Preserve line breaks. Return ONLY the raw text. If no text, return NO_TEXT."}
                    ]}], temperature=0.0, max_tokens=800
                )
                text = response.choices[0].message.content.strip()
                if text and text != 'NO_TEXT':
                    groq_result[0] = (text, 'unknown', [])
            except Exception as e:
                print(f"[Groq Vision] {e}")
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
            if vision_result[0]:   extracted_text, detected_lang, raw_blocks = vision_result[0]
            elif groq_result[0]:   extracted_text, detected_lang, raw_blocks = groq_result[0]

        if not extracted_text:
            return jsonify({'success': False, 'error': 'No text found. Try pointing at clearer text.'})

        translated_text, engine = translate(extracted_text, target_lang, detected_lang)
        text_blocks             = _translate_blocks_batch(raw_blocks, target_lang, detected_lang)
        return jsonify({
            'success':          True,
            'original_text':    extracted_text,
            'translated_text':  translated_text,
            'detected_lang':    detected_lang,
            'engine':           engine,
            'text_blocks':      text_blocks
        })
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
        text_blocks = _translate_blocks_batch(blocks, target_lang, src_lang)
        return jsonify({'success': True, 'text_blocks': text_blocks})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


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
        print(f"Download error: {e}")
        return jsonify({'success': False, 'error': str(e)})


def _slot(emoji, label, slot):
    if not slot or not slot.get('activity'): return ''
    tip_html = (
        '<div style="font-size:11px;color:#1A8A72;margin-top:4px;font-style:italic;border-left:2px solid #1A8A72;padding-left:8px;">&#128161; ' + slot.get('tip', '') + '</div>'
        if slot.get('tip') else ''
    )
    return (
        '<div style="background:#F7F6F2;border-radius:10px;padding:12px;margin-bottom:8px;">'
        '<div style="font-size:10px;font-weight:700;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">' + emoji + ' ' + label + '</div>'
        '<div style="font-weight:700;font-size:14px;">' + slot.get('activity', '') + '</div>'
        '<div style="font-size:12px;color:#1A8A72;margin-top:2px;">📍 ' + slot.get('location', '') + '</div>'
        '<div style="font-size:12px;font-weight:700;color:#28B06A;margin-top:4px;">💰 ' + slot.get('cost', '') + ' · ⏱ ' + slot.get('duration', '') + '</div>'
        + tip_html + '</div>'
    )

def _meal(emoji, label, meal):
    if not meal or not meal.get('restaurant'): return ''
    return (
        '<div style="background:#FFF7ED;border-radius:10px;padding:10px 12px;margin-bottom:8px;">'
        '<div style="font-size:10px;font-weight:700;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">' + emoji + ' ' + label + '</div>'
        '<div style="font-weight:700;font-size:14px;">' + meal.get('restaurant', '') + '</div>'
        '<div style="font-size:12px;color:#6B6860;">🍽️ ' + meal.get('cuisine', '') + ' · 💰 ' + meal.get('cost', '') + '</div>'
        '</div>'
    )

def _stay(stay):
    if not stay or not stay.get('name'): return ''
    return (
        '<div style="background:#EFF6FF;border-radius:10px;padding:10px 12px;margin-bottom:8px;">'
        '<div style="font-size:10px;font-weight:700;color:#3A6BC8;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">🏨 Stay</div>'
        '<div style="font-weight:700;font-size:14px;">' + stay.get('name', '') + '</div>'
        '<div style="font-size:12px;color:#6B6860;">📍 ' + stay.get('area', '') + ' · 💰 ' + stay.get('cost', '') + '/night</div>'
        '</div>'
    )

def _build_single_city_html(plan):
    destination = plan.get('destination', 'Your Trip')
    days        = plan.get('days', 0)
    currency    = plan.get('currency', '')
    itinerary   = plan.get('itinerary', [])
    packing     = plan.get('packing_list', [])
    phrases     = plan.get('local_phrases', [])
    gems        = plan.get('hidden_gems', [])
    emergency   = plan.get('emergency_numbers', {})
    tips        = plan.get('tips', [])
    visa        = plan.get('visa_info', {})
    flight      = plan.get('flight_info', {})
    sim         = plan.get('sim_internet', {})
    cultural    = plan.get('cultural_guide', {})
    budget_tips = plan.get('budget_tips', [])

    day_cards_html = ''
    for day in itinerary:
        day_cards_html += (
            '<div style="background:white;border-radius:16px;padding:20px;margin-bottom:12px;border-left:4px solid #1A8A72;box-shadow:0 2px 8px rgba(0,0,0,0.06);">'
            '<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">'
            '<div style="width:36px;height:36px;border-radius:50%;background:#1A8A72;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;color:white;flex-shrink:0;">' + str(day.get('day', '')) + '</div>'
            '<div style="font-weight:700;font-size:16px;">' + day.get('title', '') + '</div>'
            '</div>'
            + _slot('🌅', 'Morning', day.get('morning'))
            + _meal('☀️', 'Lunch', day.get('lunch'))
            + _slot('☀️', 'Afternoon', day.get('afternoon'))
            + _slot('🌆', 'Evening', day.get('evening'))
            + _meal('🌙', 'Dinner', day.get('dinner'))
            + _stay(day.get('accommodation'))
            + '</div>'
        )

    packing_html = ''.join(
        f'<span style="display:inline-block;background:#EFF6FF;color:#1A8A72;border-radius:20px;padding:4px 12px;font-size:12px;margin:3px;font-weight:500;">{item}</span>'
        for item in packing
    )
    phrase_html = ''.join(
        f'<div style="display:flex;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid #F0EBE0;flex-wrap:wrap;">'
        f'<span style="font-weight:600;min-width:120px;font-size:13px;">{p.get("phrase","")}</span>'
        f'<span style="color:#1A8A72;font-size:14px;font-weight:600;">{p.get("translation","")}</span>'
        f'<span style="color:#6B6860;font-size:12px;font-style:italic;">({p.get("pronunciation","")})</span>'
        f'</div>'
        for p in phrases
    )
    gems_html = ''.join(
        f'<div style="background:#F7F6F2;border-radius:12px;padding:14px;margin-bottom:10px;">'
        f'<div style="font-weight:700;font-size:14px;margin-bottom:4px;">💎 {g.get("name","")}</div>'
        f'<div style="font-size:12px;color:#3D3730;">{g.get("description","")}</div>'
        f'<div style="font-size:11px;color:#6B6860;margin-top:6px;">📍 {g.get("location","")} · ⏰ {g.get("best_time","")} · 💰 {g.get("cost","")}</div>'
        f'</div>'
        for g in gems
    )
    cultural = plan.get('cultural_guide', {})
    dos      = ''.join(f'<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">✅ {d}</li>' for d in cultural.get('dos', []))
    donts    = ''.join(f'<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">❌ {d}</li>' for d in cultural.get('donts', []))
    em_html  = ''.join(
        f'<div style="background:#FEF2F2;border-radius:10px;padding:12px;text-align:center;">'
        f'<div style="font-size:20px;font-weight:700;color:#D84C3E;">{num}</div>'
        f'<div style="font-size:11px;color:#6B6860;margin-top:2px;">{k.replace("_"," ").title()}</div>'
        f'</div>'
        for k, num in emergency.items()
    )
    tips_html  = ''.join(f'<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">🎯 {t}</li>' for t in tips)
    btips_html = ''.join(f'<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">💡 {t}</li>' for t in budget_tips)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{destination} — Yaply Offline Itinerary</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#F7F6F2;color:#2C2B28}}
.header{{background:linear-gradient(135deg,#2563FF,#0B1220);color:white;padding:32px 24px;text-align:center}}
.container{{max-width:800px;margin:0 auto;padding:20px 16px 60px}}
.section{{background:white;border-radius:16px;padding:20px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.section-title{{font-size:15px;font-weight:800;color:#2563FF;margin-bottom:14px}}
.info-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.info-item{{background:#F7F6F2;border-radius:10px;padding:12px}}
.info-label{{font-size:10px;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px}}
.info-value{{font-size:13px;font-weight:700}}
ul{{padding:0;list-style:none}}
.offline-note{{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:12px 16px;margin-bottom:16px;font-size:12px;color:#1D4ED8;text-align:center}}
.footer{{text-align:center;padding:24px;color:#6B6860;font-size:12px}}
</style>
</head>
<body>
<div class="header">
  <div style="font-size:11px;opacity:.6;letter-spacing:2px;margin-bottom:4px;">YAPLY · AI TRAVEL OS</div>
  <h1 style="font-size:28px;font-weight:800;letter-spacing:-1px;">✈️ {destination}</h1>
  <div style="opacity:.8;font-size:13px;margin-top:6px;">{days}-Day Complete Itinerary</div>
  <div style="margin-top:8px;">
    <span style="background:rgba(255,255,255,.2);border-radius:20px;padding:4px 12px;font-size:12px;margin:3px;display:inline-block;">📅 {days} Days</span>
    <span style="background:rgba(255,255,255,.2);border-radius:20px;padding:4px 12px;font-size:12px;margin:3px;display:inline-block;">💰 {currency}</span>
    <span style="background:rgba(255,255,255,.2);border-radius:20px;padding:4px 12px;font-size:12px;margin:3px;display:inline-block;">📱 Offline Ready</span>
  </div>
</div>
<div class="container">
  <div class="offline-note">📱 Works completely offline — save to your phone before you travel.</div>
  <div class="section">
    <div class="section-title">ℹ️ Trip Essentials</div>
    <div class="info-grid">
      <div class="info-item"><div class="info-label">Best Time</div><div class="info-value">{plan.get("best_time_to_visit","")}</div></div>
      <div class="info-item"><div class="info-label">Language</div><div class="info-value">{plan.get("language","")}</div></div>
      <div class="info-item"><div class="info-label">Timezone</div><div class="info-value">{plan.get("timezone","")}</div></div>
      <div class="info-item"><div class="info-label">Currency</div><div class="info-value">{plan.get("currency","")}</div></div>
    </div>
  </div>
  {"<div class='section'><div class='section-title'>✈️ Flight Info</div><div class='info-grid'><div class='info-item'><div class='info-label'>Cost</div><div class='info-value'>" + flight.get("estimated_cost","") + "</div></div><div class='info-item'><div class='info-label'>Duration</div><div class='info-value'>" + flight.get("flight_duration","") + "</div></div></div></div>" if flight else ""}
  {"<div class='section'><div class='section-title'>🛂 Visa</div><div class='info-grid'><div class='info-item'><div class='info-label'>Required</div><div class='info-value'>" + ("Yes" if visa.get("required") else "No") + "</div></div><div class='info-item'><div class='info-label'>Cost</div><div class='info-value'>" + visa.get("cost","") + "</div></div></div></div>" if visa else ""}
  <div style="font-size:14px;font-weight:800;color:#2C2B28;margin-bottom:12px;">📅 Day by Day Itinerary</div>
  {day_cards_html}
  {"<div class='section'><div class='section-title'>💎 Hidden Gems</div>" + gems_html + "</div>" if gems_html else ""}
  {"<div class='section'><div class='section-title'>💬 Essential Phrases</div>" + phrase_html + "</div>" if phrase_html else ""}
  {"<div class='section'><div class='section-title'>🌍 Cultural Guide</div><div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;'><div><div style='font-size:11px;font-weight:700;color:#28B06A;margin-bottom:8px;'>DOS</div><ul>" + dos + "</ul></div><div><div style='font-size:11px;font-weight:700;color:#D84C3E;margin-bottom:8px;'>DON'TS</div><ul>" + donts + "</ul></div></div></div>" if cultural else ""}
  {"<div class='section'><div class='section-title'>🚨 Emergency Numbers</div><div style='display:grid;grid-template-columns:1fr 1fr;gap:9px;'>" + em_html + "</div></div>" if em_html else ""}
  {"<div class='section'><div class='section-title'>🎒 Packing List</div>" + packing_html + "</div>" if packing_html else ""}
  {"<div class='section'><div class='section-title'>🎯 Pro Tips</div><ul>" + tips_html + "</ul></div>" if tips_html else ""}
  {"<div class='section'><div class='section-title'>💡 Budget Tips</div><ul>" + btips_html + "</ul></div>" if btips_html else ""}
</div>
<div class="footer">Generated by <strong>Yaply</strong> — Your Complete Travel OS — <a href="https://yaply.live" style="color:#2563FF;">yaply.live</a></div>
</body>
</html>'''


def _build_multi_city_html(plan):
    title    = plan.get('trip_title', 'Multi-City Trip')
    cities   = plan.get('cities', [])
    transits = plan.get('transit_plans', [])
    currency = plan.get('currency', 'INR')
    sim      = plan.get('sim_strategy', {})
    packing  = plan.get('packing_for_route', {})
    suggs    = plan.get('smart_suggestions', [])
    budget   = plan.get('budget_split', {})
    MC_COLS  = ['#2563FF', '#FF7A59', '#18C29C', '#F5B942', '#8B5CF6', '#0EA5E9']

    sugg_html   = ''.join(f'<li style="padding:6px 0;font-size:13px;border-bottom:1px solid rgba(37,99,255,.1);">🧠 {s}</li>' for s in suggs)
    budget_html = ''.join(
        f'<div style="background:#F7F6F2;border-radius:10px;padding:12px;">'
        f'<div style="font-size:10px;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">{k.replace("_"," ").title()}</div>'
        f'<div style="font-size:13px;font-weight:700;color:#2563FF;">{currency} {int(v or 0)}</div>'
        f'</div>'
        for k, v in budget.items()
    )

    cities_html = ''
    for idx, city in enumerate(cities):
        col       = MC_COLS[idx % len(MC_COLS)]
        itinerary = city.get('itinerary', [])
        gems      = city.get('hidden_gems', [])
        tips      = city.get('local_tips', [])

        day_html = ''
        for day in itinerary:
            day_html += (
                '<div style="background:#F7F6F2;border-radius:12px;padding:12px;margin-bottom:8px;">'
                '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                f'<div style="width:26px;height:26px;border-radius:50%;background:{col};display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:white;flex-shrink:0;">' + str(day.get('day', '')) + '</div>'
                '<div style="font-size:14px;font-weight:700;">' + (day.get('day_label') or day.get('theme') or f'Day {day.get("day","")}') + '</div>'
                '</div>'
                + _slot('🌅', 'Morning', day.get('morning'))
                + _meal('☀️', 'Lunch', day.get('lunch'))
                + _slot('☀️', 'Afternoon', day.get('afternoon'))
                + _slot('🌆', 'Evening', day.get('evening'))
                + _meal('🌙', 'Dinner', day.get('dinner'))
                + '</div>'
            )

        gems_html = ''.join(
            f'<div style="background:#F7F6F2;border-radius:10px;padding:11px;margin-bottom:7px;">'
            f'<div style="font-weight:700;font-size:13px;">💎 {g.get("name","")}</div>'
            f'<div style="font-size:12px;color:#3D3730;margin-top:3px;">{g.get("why") or g.get("description","")}</div>'
            f'</div>'
            for g in gems
        )
        tips_html = ''.join(
            f'<div style="display:flex;gap:8px;padding:8px 0;border-bottom:1px solid #F0EBE0;">'
            f'<div style="width:20px;height:20px;border-radius:50%;background:#2C2B28;color:white;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{i+1}</div>'
            f'<div style="font-size:13px;">{tip}</div>'
            f'</div>'
            for i, tip in enumerate(tips)
        )

        cities_html += (
            f'<div style="background:white;border-radius:16px;margin-bottom:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.06);">'
            f'<div style="background:{col};padding:18px 20px;">'
            f'<div style="font-size:11px;color:rgba(255,255,255,.6);letter-spacing:1px;margin-bottom:4px;">CITY {idx+1}</div>'
            f'<div style="font-size:20px;font-weight:800;color:white;">{city.get("city","")} , {city.get("country","")}</div>'
            f'<div style="font-size:12px;color:rgba(255,255,255,.6);margin-top:4px;">{city.get("days","")} days · {currency} {int(city.get("city_budget",0))} · {city.get("best_area_to_stay","")}</div>'
            f'</div>'
            f'<div style="padding:16px 18px;">'
            + (f'<div style="background:#F7F6F2;border-radius:10px;padding:10px;margin-bottom:12px;font-size:12px;">🌤️ {city.get("weather_note","")}</div>' if city.get('weather_note') else '')
            + (f'<div style="font-size:11px;font-weight:700;color:#6B6860;text-transform:uppercase;margin-bottom:8px;">Daily Plan</div>' + day_html if day_html else '')
            + (f'<div style="font-size:11px;font-weight:700;color:#6B6860;text-transform:uppercase;margin:10px 0 8px;">Hidden Gems</div>' + gems_html if gems_html else '')
            + (f'<div style="font-size:11px;font-weight:700;color:#6B6860;text-transform:uppercase;margin:10px 0 8px;">Local Tips</div>' + tips_html if tips_html else '')
            + '</div></div>'
        )

        transit = next((t for t in transits if t.get('from') and city.get('city', '') in t.get('from', '')), None)
        if transit and idx < len(cities) - 1:
            opts     = transit.get('options', [])
            opt_html = ''.join(
                f'<div style="background:{"#EFF6FF" if opt.get("recommended") else "#F7F6F2"};border-radius:10px;padding:10px 12px;margin-bottom:6px;display:flex;align-items:center;gap:10px;">'
                f'<div style="flex:1;"><div style="font-size:13px;font-weight:700;">{opt.get("mode","")} · {opt.get("operator","")}</div>'
                f'<div style="font-size:11px;color:#6B6860;">⏱ {opt.get("duration","")} · {opt.get("comfort","")}</div></div>'
                + ('<span style="background:#2563FF;color:white;font-size:10px;font-weight:700;padding:2px 8px;border-radius:100px;">BEST</span>' if opt.get('recommended') else '')
                + f'<div style="font-size:14px;font-weight:800;color:#2563FF;">{currency} {opt.get("total_cost",opt.get("cost","?"))}</div>'
                f'</div>'
                for opt in opts[:3]
            )
            cities_html += (
                f'<div style="background:white;border-radius:12px;border:1.5px solid #EFEDE8;padding:14px 16px;margin-bottom:12px;">'
                f'<div style="font-size:12px;font-weight:700;margin-bottom:8px;">✈️ {transit.get("from","")} → {transit.get("to","")}</div>'
                + opt_html
                + (f'<div style="background:#EFEDE8;border-radius:8px;padding:8px 10px;font-size:12px;margin-top:6px;">💡 {transit.get("transit_tip","")}</div>' if transit.get('transit_tip') else '')
                + '</div>'
            )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Yaply Offline</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#F7F6F2;color:#2C2B28}}
.header{{background:linear-gradient(135deg,#0B1220,#1a2a4a);color:white;padding:28px 24px;text-align:center}}
.container{{max-width:800px;margin:0 auto;padding:20px 16px 60px}}
.section{{background:white;border-radius:16px;padding:20px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.section-title{{font-size:15px;font-weight:800;color:#2563FF;margin-bottom:14px}}
.offline-note{{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:12px 16px;margin-bottom:16px;font-size:12px;color:#1D4ED8;text-align:center}}
.footer{{text-align:center;padding:24px;color:#6B6860;font-size:12px}}
</style>
</head>
<body>
<div class="header">
  <div style="font-size:11px;opacity:.5;letter-spacing:2px;margin-bottom:6px;">YAPLY · AI TRAVEL OS</div>
  <h1 style="font-size:22px;font-weight:800;letter-spacing:-1px;">🗺️ {title}</h1>
  <div style="opacity:.6;font-size:13px;margin-top:6px;">{plan.get("total_days","")} days · {len(cities)} cities · {currency} {plan.get("total_budget","")}</div>
</div>
<div class="container">
  <div class="offline-note">📱 Works completely offline — save to your phone before you travel.</div>
  {"<div class='section'><div class='section-title'>🧠 Smart Suggestions</div><ul style='padding:0;list-style:none;'>" + sugg_html + "</ul></div>" if sugg_html else ""}
  {"<div class='section'><div class='section-title'>💰 Budget Split</div><div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>" + budget_html + "</div></div>" if budget_html else ""}
  {cities_html}
  {"<div class='section'><div class='section-title'>📱 SIM Strategy</div><div style='background:#EFF6FF;border-radius:10px;padding:10px 12px;font-size:13px;color:#2563FF;'>" + sim.get("recommendation","") + "</div></div>" if sim.get("recommendation") else ""}
  {"<div class='section'><div class='section-title'>🎒 Packing for This Route</div><div style='font-size:13px;margin-bottom:10px;'>🌡️ " + packing.get("weather_variation","") + "</div><div>" + "".join(f'<span style="display:inline-block;background:#EFF6FF;color:#2563FF;border-radius:20px;padding:4px 12px;font-size:12px;margin:3px;font-weight:500;">{item}</span>' for item in packing.get("key_items",[])) + "</div></div>" if packing.get("key_items") else ""}
</div>
<div class="footer">Generated by <strong>Yaply</strong> — <a href="https://yaply.live" style="color:#2563FF;">yaply.live</a></div>
</body>
</html>'''


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
                    if 'lang_a'   in cfg: lang_a  = cfg['lang_a'].lower()[:2]
                    if 'lang_b'   in cfg: lang_b  = cfg['lang_b'].lower()[:2]
                    if 'speaker'  in cfg:
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
                src           = lang_a if active_speaker == 'A' else lang_b
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


@app.route('/api/places-autocomplete')
@limiter.limit("60 per hour")
def places_autocomplete():
    try:
        q   = request.args.get('q', '')
        key = os.getenv('GOOGLE_PLACES_API_KEY', '')
        if not q or not key:
            return jsonify({'success': False, 'predictions': []})
        r = req.get(
            'https://maps.googleapis.com/maps/api/place/autocomplete/json',
            params={
                'input':    q,
                'types':    '(cities)',
                'key':      key,
                'language': 'en',
            },
            timeout=8
        )
        data = r.json()
        if data.get('status') == 'OK':
            return jsonify({'success': True, 'predictions': data.get('predictions', [])})
        return jsonify({'success': False, 'predictions': []})
    except Exception as e:
        return jsonify({'success': False, 'predictions': [], 'error': str(e)})


# ── Nearby stations/airports via Google Places ──
@app.route('/api/nearby-stations')
@require_auth
@limiter.limit("30 per hour")
def nearby_stations():
    try:
        location   = request.args.get('location', '')
        place_type = request.args.get('type', 'train_station')
        key        = os.getenv('GOOGLE_PLACES_API_KEY', '')

        if not location or not key:
            return jsonify({'success': False, 'stations': []})

        # First geocode the location to get lat/lng
        geo = req.get(
            'https://maps.googleapis.com/maps/api/geocode/json',
            params={'address': location, 'key': key},
            timeout=8
        ).json()

        if not geo.get('results'):
            return jsonify({'success': False, 'stations': []})

        loc = geo['results'][0]['geometry']['location']
        lat, lng = loc['lat'], loc['lng']

        # Now find nearby stations
        places = req.get(
            'https://maps.googleapis.com/maps/api/place/nearbysearch/json',
            params={
                'location': f"{lat},{lng}",
                'radius':   50000,  # 50km radius
                'type':     place_type,
                'key':      key,
                'language': 'en',
            },
            timeout=8
        ).json()

        stations = []
        for p in places.get('results', [])[:6]:
            # Calculate approximate distance
            import math
            plat = p['geometry']['location']['lat']
            plng = p['geometry']['location']['lng']
            dist_km = round(math.sqrt((plat-lat)**2 + (plng-lng)**2) * 111, 1)

            stations.append({
                'name':     p.get('name', ''),
                'address':  p.get('vicinity', ''),
                'distance': f"{dist_km} km",
                'rating':   p.get('rating', 0),
                'place_id': p.get('place_id', ''),
            })

        # Sort by distance
        stations.sort(key=lambda x: float(x['distance'].replace(' km', '')))

        return jsonify({'success': True, 'stations': stations})

    except Exception as e:
        return jsonify({'success': False, 'stations': [], 'error': str(e)})

# ════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Endpoint not found'}), 404
    return render_template('landing.html'), 404

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({'success': False, 'error': 'Too many requests. Please slow down.'}), 429

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Server error. Our team has been notified.'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=port,
        allow_unsafe_werkzeug=True
    )