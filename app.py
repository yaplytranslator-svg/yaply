"""
app.py — Yaply COMPLETE Production App v3 (FIXED)
- Single unified server — DO NOT run trip_planner.py or main_translate.py
- ALL routes here including backward-compatible aliases
- /api/ prefix for all AI routes
- Real auth on all routes that need it
- Rate limiting, security headers, input validation
"""
from diary import diary_bp, init_diary_db
from groups import groups_bp, init_groups_db, register_socketio_events
from urllib import response
import os, io, base64, json, wave, struct, threading, time
import requests as req
from dotenv import load_dotenv
load_dotenv()

# ── FLASK FIRST ──
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from flask_sock import Sock

app = Flask(__name__)
CORS(app)
 
app.register_blueprint(groups_bp)
app.register_blueprint(diary_bp)

from flask_socketio import SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_timeout=60,
    ping_interval=25,   
)
register_socketio_events(socketio)
sock = Sock(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET', 'yaply-secret-2025-change-me')

# ── HEALTH (no auth needed — Railway checks these) ──
@app.route('/health')
def health(): return 'OK', 200

@app.route('/ping')
def ping(): return 'pong', 200

# ── RATE LIMITER ──
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["500 per day", "100 per hour"],
    storage_uri="memory://"
)

# ── DATABASE + AUTH ──
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


# ── AI CLIENTS ──
from groq import Groq
import edge_tts, asyncio

_groq_key = os.getenv("GROQ_API_KEY")
if not _groq_key:
    raise RuntimeError("GROQ_API_KEY not set — app cannot start")
groq_client = Groq(api_key=_groq_key)

try:
    import deepl
    deepl_client = deepl.Translator(os.getenv("DEEPL_API_KEY"))
except:
    deepl_client = None

WEATHER_KEY       = os.getenv("OPENWEATHER_API_KEY")
EXCHANGE_KEY      = os.getenv("EXCHANGE_API_KEY")
GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_API_KEY")
UNSPLASH_KEY      = os.getenv("UNSPLASH_ACCESS_KEY")
GOOGLE_CLIENT_ID  = os.getenv("GOOGLE_CLIENT_ID", "")

# ── SECURITY HEADERS ──
@app.after_request
def add_security_headers(r):
    r.headers['X-Content-Type-Options'] = 'nosniff'
    r.headers['X-Frame-Options'] = 'DENY'
    r.headers['X-XSS-Protection'] = '1; mode=block'
    r.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return r

# ── LANGUAGE MAPS ──
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

# ══════════════════════════════════════
# HELPERS
# ══════════════════════════════════════

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
        count = len(audio_bytes)//2
        if count == 0: return 0
        samples = struct.unpack('<'+'h'*count, audio_bytes[:count*2])
        return (sum(s*s for s in samples)/count)**0.5
    except: return 0

def normalize_audio(raw_bytes):
    try:
        count = len(raw_bytes)//2
        if count == 0: return raw_bytes
        samples = list(struct.unpack('<'+'h'*count, raw_bytes[:count*2]))
        peak = max(abs(s) for s in samples)
        if peak == 0: return raw_bytes
        factor = min((32767*0.8)/peak, 4.0)
        normalized = [max(-32768, min(32767, int(s*factor))) for s in samples]
        return struct.pack('<'+'h'*len(normalized), *normalized)
    except: return raw_bytes

def audio_to_wav(raw_bytes, sample_rate=16000):
    n = normalize_audio(raw_bytes)
    buf = io.BytesIO()
    with wave.open(buf,'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sample_rate); wf.writeframes(n)
    buf.seek(0); return buf.read()

def is_valid(text):
    if not text: return False
    t = text.strip()
    if len(t) < 3 or t in HALLUCINATIONS: return False
    alpha = sum(c.isalpha() for c in t)
    return alpha >= len(t)*0.25

def safe_send(ws, data):
    try: ws.send(json.dumps(data))
    except: pass

def groq_json(prompt, system="Return ONLY valid JSON. No markdown, no backticks.", model="llama-3.1-8b-instant", temp=0.2, max_tok=2000):
    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        temperature=temp, max_tokens=max_tok
    )
    result = response.choices[0].message.content.strip()
    # Strip markdown fences
    if '```' in result:
        for p in result.split('```'):
            if '{' in p:
                result = p[4:] if p.startswith('json') else p
                break
    # Find JSON object
    start = result.find('{'); end = result.rfind('}')+1
    if start != -1: result = result[start:end]
    return json.loads(result)

def transcribe(wav_data, lang_hint=None):
    kwargs = {
        'file': ('audio.wav', wav_data),
        'model': 'whisper-large-v3-turbo',
        'response_format': 'verbose_json',
        'temperature': 0.0,
    }
    if lang_hint and lang_hint not in ('auto','unknown',None,''):
        wc = WHISPER_LANG.get(lang_hint)
        if wc:
            kwargs['language'] = wc
            prompt = WHISPER_PROMPTS.get(wc,'')
            if prompt: kwargs['prompt'] = prompt
    result = groq_client.audio.transcriptions.create(**kwargs)
    text     = result.text.strip()
    detected = getattr(result,'language','unknown')
    segments = getattr(result,'segments',[])
    conf = sum(abs(s.get('avg_logprob',-1)) for s in segments)/max(len(segments),1) if segments else 0.0
    return text, detected, conf

def translate(text, target_lang, src_lang=None):
    tgt = target_lang.lower()[:2] if len(target_lang) >= 2 else target_lang
    deepl_code = DEEPL_LANGS.get(tgt) or DEEPL_LANGS.get(target_lang)
    if deepl_code and deepl_client:
        try:
            src = None
            if src_lang and src_lang not in ('unknown','auto',None,''):
                src = src_lang.upper()[:2]
                if src.lower() == tgt.lower(): src = None
            result = deepl_client.translate_text(text, target_lang=deepl_code, source_lang=src)
            return result.text, 'DeepL'
        except Exception as e: print(f"[DeepL] {e}")
    lang_name = LANG_NAMES.get(target_lang, 'English')
    r = groq_client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role':'system','content':f'Translate to {lang_name}. Return ONLY the translation.'},
            {'role':'user','content':text}
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
        except:
            communicate = edge_tts.Communicate(text, 'en-US-JennyNeural')
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk['type'] == 'audio': buf.write(chunk['data'])
            buf.seek(0); return buf.read()
    return asyncio.run(_run())

# ══════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════

@app.route('/')
def landing(): return render_template('landing.html')

@app.route('/login')
def login_page(): return render_template('auth.html', google_client_id=GOOGLE_CLIENT_ID)

@app.route('/app')
def main_app(): return render_template('yaply-app.html', google_client_id=GOOGLE_CLIENT_ID)

@app.route('/plan')
def plan_page(): return render_template('before_trip.html')

@app.route('/during')
def during_page(): return render_template('during_trip.html')

@app.route('/after')
def after_page(): return render_template('after_trip.html')

@app.route('/tools')
def tools_page(): return render_template('tools_extra.html')

@app.route('/discover')
def discover_page(): return render_template('discover.html')

@app.route('/profile')
def profile_page(): return render_template('profile.html')

@app.route('/translate')
def translate_page(): return render_template('stream.html')

@app.route('/convo')
def convo_page(): return render_template('convo.html')

@app.route('/camera')
def camera_page(): return render_template('camera.html')

# ══════════════════════════════════════
# DATABASE ROUTES
# ══════════════════════════════════════

@app.route('/api/trips', methods=['GET'])
@require_auth
def api_get_trips():
    return jsonify({'success':True,'trips':get_trips(g.user_id)})

@app.route('/api/trips', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_save_trip():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success':False,'error':err})
        trip_id = save_trip(
            user_id=g.user_id,
            destination=clean(data.get('destination','')),
            origin=clean(data.get('origin','India')),
            days=min(max(int(data.get('days',7)),1),365),
            people=min(max(int(data.get('people',1)),1),50),
            budget=clean(data.get('budget','80000')),
            currency=clean(data.get('currency','INR'),3),
            vibes=clean(data.get('vibes','Adventure')),
            passport=clean(data.get('passport','India')),
            plan_data=data.get('plan_data')
        )
        log_action(g.user_id,'save_trip',request.remote_addr)
        return jsonify({'success':True,'trip_id':trip_id})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/trips/<int:trip_id>', methods=['GET'])
@require_auth
def api_get_trip(trip_id):
    trip = get_trip(trip_id, g.user_id)
    if not trip: return jsonify({'success':False,'error':'Trip not found'}), 404
    return jsonify({'success':True,'trip':trip})

@app.route('/api/trips/<int:trip_id>', methods=['PUT'])
@require_auth
def api_update_trip(trip_id):
    try:
        update_trip(trip_id, g.user_id, **(request.get_json() or {}))
        return jsonify({'success':True})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/trips/<int:trip_id>', methods=['DELETE'])
@require_auth
def api_delete_trip(trip_id):
    delete_trip(trip_id, g.user_id)
    return jsonify({'success':True})

@app.route('/api/places', methods=['GET'])
@require_auth
def api_get_places():
    return jsonify({'success':True,'places':get_places(g.user_id)})

@app.route('/api/places', methods=['POST'])
@require_auth
def api_save_place():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['name'])
        if not ok: return jsonify({'success':False,'error':err})
        place_id = save_place(
            user_id=g.user_id,
            name=clean(data.get('name','')),
            city=clean(data.get('city','')),
            country=clean(data.get('country','')),
            continent=clean(data.get('continent','')),
            description=clean(data.get('description',''),1000),
            image_url=clean(data.get('image_url',''),500),
            emoji=clean(data.get('emoji','📍'),5),
            tags=data.get('tags',[]),
            trip_id=data.get('trip_id')
        )
        return jsonify({'success':True,'place_id':place_id})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/places/<int:place_id>', methods=['DELETE'])
@require_auth
def api_delete_place(place_id):
    delete_place(place_id, g.user_id)
    return jsonify({'success':True})

@app.route('/api/trips/<int:trip_id>/expenses', methods=['GET'])
@require_auth
def api_get_expenses(trip_id):
    return jsonify({'success':True,'expenses':get_expenses(trip_id, g.user_id)})

@app.route('/api/trips/<int:trip_id>/expenses', methods=['POST'])
@require_auth
def api_add_expense(trip_id):
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['title','amount'])
        if not ok: return jsonify({'success':False,'error':err})
        exp_id = add_expense(
            trip_id=trip_id, user_id=g.user_id,
            title=clean(data.get('title','')),
            amount=float(data.get('amount',0)),
            category=clean(data.get('category','Other')),
            currency=clean(data.get('currency','INR'),3),
            paid_by=clean(data.get('paid_by','')),
            split_with=data.get('split_with',[])
        )
        return jsonify({'success':True,'expense_id':exp_id})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/expenses/<int:expense_id>', methods=['DELETE'])
@require_auth
def api_delete_expense(expense_id):
    delete_expense(expense_id, g.user_id)
    return jsonify({'success':True})

@app.route('/api/trips/<int:trip_id>/journal', methods=['GET'])
@require_auth
def api_get_journal(trip_id):
    return jsonify({'success':True,'journal':get_journal(trip_id, g.user_id)})

@app.route('/api/trips/<int:trip_id>/journal', methods=['POST'])
@require_auth
def api_save_journal(trip_id):
    try:
        data = request.get_json() or {}
        save_journal(trip_id, g.user_id, data.get('content'))
        return jsonify({'success':True})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/profile', methods=['GET'])
@require_auth
def api_profile():
    return jsonify({'success':True,'user':safe_user(g.user),'stats':get_user_stats(g.user_id)})

# ══════════════════════════════════════
# AI ROUTES — all /api/ prefix
# ══════════════════════════════════════

@app.route('/api/plan', methods=['POST'])
@require_auth
@limiter.limit("15 per hour")
def api_plan():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success':False,'error':err})

        destination = clean(data.get('destination',''))
        origin      = clean(data.get('origin','India'))
        days        = min(max(int(data.get('days',5)),1),60)
        budget      = clean(data.get('budget','50000'))
        vibe        = clean(data.get('vibe','adventure'))
        people      = min(max(int(data.get('people',1)),1),20)
        currency    = clean(data.get('currency','INR'),3)
        passport    = clean(data.get('passport','India'))

        prompt = f"""World-class travel planner. Create UNIQUE {days}-day itinerary.
FROM: {origin} → TO: {destination}
Budget: {currency} {budget} for {people} people | Style: {vibe} | Passport: {passport}
ALL prices MUST be in {currency}.

Return ONLY valid JSON:
{{"destination":"{destination}","days":{days},"language":"local lang","currency":"local currency","timezone":"tz","best_time_to_visit":"months",
"budget_breakdown":{{"flights":"{currency} X","accommodation":"{currency} X","food":"{currency} X","transport":"{currency} X","activities":"{currency} X","miscellaneous":"{currency} X"}},
"flight_info":{{"estimated_cost":"{currency} X","best_airlines":["a1","a2"],"flight_duration":"Xh","best_time_to_book":"X weeks ahead"}},
"itinerary":[{{"day":1,"title":"Day title",
  "morning":{{"activity":"name","location":"place","duration":"2h","cost":"{currency} X","tip":"insider tip"}},
  "afternoon":{{"activity":"name","location":"place","duration":"2h","cost":"{currency} X","tip":"tip"}},
  "evening":{{"activity":"name","location":"place","duration":"2h","cost":"{currency} X","tip":"tip"}},
  "lunch":{{"restaurant":"name","cuisine":"type","cost":"{currency} X"}},
  "dinner":{{"restaurant":"name","cuisine":"type","cost":"{currency} X"}},
  "accommodation":{{"name":"hotel","area":"area","cost":"{currency} X/night"}}}}],
"hidden_gems":[{{"name":"place","description":"why special","location":"area","best_time":"when","cost":"{currency} X"}}],
"local_transport":{{"airport_to_city":{{"options":["opt1"],"cost":"{currency} X","duration":"30min"}},"within_city":[{{"type":"Metro","cost":"{currency} X/ride","tip":"tip"}}],"useful_apps":["app1"]}},
"sim_internet":{{"best_option":"option","cost":"{currency} X","data":"XGB","where_to_buy":"location"}},
"cultural_guide":{{"dos":["do1","do2","do3"],"donts":["dont1","dont2","dont3"],"dress_code":"advice","tipping":"culture","greetings":"how"}},
"vaccinations":{{"required":["v1"],"recommended":["v2"],"note":"advice"}},
"packing_list":["item1","item2","item3"],
"emergency_numbers":{{"police":"number","ambulance":"number","fire":"number","tourist_helpline":"number"}},
"visa_info":{{"required":true,"type":"tourist","validity":"30 days","cost":"{currency} X"}},
"payment_info":{{"preferred":"card/cash","atm_availability":"common","notify_bank":true,"forex_tips":"advice"}},
"must_have_apps":[{{"name":"app","purpose":"what","platform":"iOS/Android"}}],
"power_plug":{{"type":"type","voltage":"220V","adapter_needed":true}},
"insurance":{{"recommended":true,"type":"comprehensive","estimated_cost":"{currency} X","must_cover":["medical","cancellation"]}},
"what_to_buy":["item1","item2"],
"what_to_avoid":["item1"],
"local_phrases":[{{"phrase":"Hello","translation":"local","pronunciation":"how"}}],
"tips":["tip1","tip2","tip3"]}}"""

        result = groq_json(prompt, model="llama-3.3-70b-versatile", temp=0.3, max_tok=4000)
        trip_id = data.get('trip_id')
        if trip_id:
            update_trip(trip_id, g.user_id, plan_data=result, status='active')
        log_action(g.user_id,'plan_trip',request.remote_addr)
        return jsonify({'success':True,'plan':result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)})

# ── BACKWARD-COMPATIBLE ALIAS (old HTML calls /api/plan via /plan POST) ──
# The page routes above handle GET /plan → render template
# POST to /api/plan is the correct endpoint — HTML must use /api/plan

@app.route('/api/weather', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def api_weather():
    try:
        city = clean((request.get_json() or {}).get('city',''))
        if not city: return jsonify({'success':False,'error':'City required'})
        if not WEATHER_KEY: return jsonify({'success':False,'error':'Weather API key not configured'})
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_KEY}&units=metric&cnt=40"
        r = req.get(url, timeout=10); data = r.json()
        if data.get('cod') != '200': return jsonify({'success':False,'error':'City not found'})
        daily = {}
        for item in data['list']:
            date = item['dt_txt'].split(' ')[0]
            if date not in daily:
                daily[date] = {'date':date,'temp_max':item['main']['temp_max'],'temp_min':item['main']['temp_min'],
                               'description':item['weather'][0]['description'],'icon':item['weather'][0]['icon'],
                               'humidity':item['main']['humidity'],'wind':item['wind']['speed']}
            else:
                daily[date]['temp_max'] = max(daily[date]['temp_max'], item['main']['temp_max'])
                daily[date]['temp_min'] = min(daily[date]['temp_min'], item['main']['temp_min'])
        return jsonify({'success':True,'city':data['city']['name'],'country':data['city']['country'],'forecast':list(daily.values())[:7]})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/currency', methods=['POST'])
@require_auth
@limiter.limit("60 per hour")
def api_currency():
    try:
        data = request.get_json() or {}
        amount = float(data.get('amount',1))
        from_c = clean(data.get('from','INR'),3).upper()
        to_c   = clean(data.get('to','USD'),3).upper()
        if not EXCHANGE_KEY: return jsonify({'success':False,'error':'Exchange API key not configured'})
        r = req.get(f"https://v6.exchangerate-api.com/v6/{EXCHANGE_KEY}/pair/{from_c}/{to_c}/{amount}", timeout=10).json()
        if r.get('result') != 'success': return jsonify({'success':False,'error':'Currency not found'})
        return jsonify({'success':True,'from':from_c,'to':to_c,'amount':amount,'converted':round(r['conversion_result'],2),'rate':r['conversion_rate']})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/visa', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_visa():
    try:
        data = request.get_json() or {}
        passport    = clean(data.get('passport','India'))
        destination = clean(data.get('destination',''))
        if not destination: return jsonify({'success':False,'error':'Destination required'})
        result = groq_json(f"""Visa requirements for {passport} passport holder visiting {destination}.
Return JSON: {{"visa_required":true,"visa_type":"type","validity":"30 days","cost":"USD X","processing_days":"5","apply_online":true,"apply_url":"url","documents":["doc1","doc2"],"tips":["tip1"],"visa_on_arrival":false,"visa_free_days":0}}""")
        return jsonify({'success':True,'visa':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/detect-theme', methods=['POST'])
@optional_auth
@limiter.limit("60 per hour")
def api_detect_theme():
    try:
        destination = clean((request.get_json() or {}).get('destination',''))
        result = groq_json(f"""Visual theme for travel app destination "{destination}".
Return JSON: {{"destination_type":"Beach/Mountain/City","theme":{{"primary_color":"#hex","secondary_color":"#hex","gradient_start":"#hex","gradient_end":"#hex","mood":"description","emoji":"emoji","vibe_words":["word1","word2"]}}}}""",
        temp=0.3, max_tok=400)
        return jsonify({'success':True,'theme':result.get('theme', result)})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/place-photo', methods=['POST'])
@optional_auth
@limiter.limit("60 per hour")
def api_place_photo():
    try:
        place_name = clean((request.get_json() or {}).get('place_name',''))
        if not UNSPLASH_KEY or not place_name: return jsonify({'success':False,'error':'No key or place'})
        r = req.get("https://api.unsplash.com/search/photos",
            params={'query':f"{place_name} travel landmark",'per_page':5,'orientation':'landscape','client_id':UNSPLASH_KEY}, timeout=8)
        results = r.json().get('results',[])
        photos = [x['urls']['regular'] for x in results if x.get('urls',{}).get('regular')]
        if photos: return jsonify({'success':True,'photos':photos})
        return jsonify({'success':False,'error':'No photos found'})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/journey', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_journey():
    try:
        data = request.get_json() or {}
        origin      = clean(data.get('origin',''))
        destination = clean(data.get('destination',''))
        if not origin or not destination: return jsonify({'success':False,'error':'Origin and destination required'})
        currency = clean(data.get('currency','INR'),3)
        travel_mode = clean(data.get('travel_mode','any'))
        result = groq_json(f"""Complete door-to-door journey from "{origin}" to "{destination}". Prices in {currency}. Mode: {travel_mode}.
Return JSON with: origin,destination,origin_has_airport,nearest_airports(name/city/code/distance_from_origin/ways_to_reach),
destination_airports,recommended_route(step1/step2/step3/step4/total_duration/total_cost),
flight_options(airline/duration/stops/price/class),alternative_routes(mode/description/cost),important_notes,documents_needed""",
        model="llama-3.3-70b-versatile", temp=0.2, max_tok=3000)
        return jsonify({'success':True,'journey':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/price-check', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def api_price_check():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['item','destination'])
        if not ok: return jsonify({'success':False,'error':err})
        result = groq_json(f"""Price check: "{clean(data.get('item',''),100)}" costs {clean(data.get('currency','INR'),3)} {clean(data.get('price',''),20)} in {clean(data.get('destination',''))}.
Return JSON: verdict,verdict_color,fair_price_range,local_price,tourist_price,overpaying_by,verdict_explanation,negotiation_tips,walk_away_price,local_phrase_to_say""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/scam-alerts', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_scam_alerts():
    try:
        destination = clean((request.get_json() or {}).get('destination',''))
        if not destination: return jsonify({'success':False,'error':'Destination required'})
        result = groq_json(f"""All tourist scams in {destination}. Return JSON: scam_risk_level,scams(name/category/severity/how_it_works/red_flags/how_to_avoid/what_to_say/icon),general_rules,safe_alternatives,emergency_if_robbed""",
        model="llama-3.3-70b-versatile", max_tok=3000)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/medical-translate', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_medical_translate():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['symptoms','destination'])
        if not ok: return jsonify({'success':False,'error':err})
        result = groq_json(f"""Medical translation: symptoms="{clean(data.get('symptoms',''),200)}", destination={clean(data.get('destination',''))}, language={clean(data.get('language','Japanese'))}.
Return JSON: severity,possible_conditions,translated_symptoms,pronunciation,say_to_doctor,immediate_actions,medicines_to_ask,emergency_number,medical_phrases""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/safety-check', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_safety_check():
    try:
        destination = clean((request.get_json() or {}).get('destination',''))
        if not destination: return jsonify({'success':False,'error':'Destination required'})
        result = groq_json(f"""You are a travel safety expert. Provide comprehensive safety information for {destination}.
Return ONLY valid JSON:
{{"safety_score":75,"safety_level":"Safe/Moderate/Caution/High Risk","crime_index":"Low/Medium/High","tourist_safety":"Safe/Generally Safe/Exercise Caution/Avoid","water_safe":true,"water_advice":"tap water advice","food_safety":"Safe/Mostly Safe/Be Careful","health_risks":["risk1","risk2"],"scams_to_avoid":["common scam 1","common scam 2","common scam 3"],"safe_areas":["safe area 1","safe area 2"],"avoid_areas":["area to avoid"],"emergency_embassy":"Indian embassy address","embassy_phone":"phone number","travel_advisory":"current advisory level","solo_female_safety":"Safe/Generally Safe/Take Precautions/Not Recommended","best_safety_tips":["tip1","tip2","tip3","tip4","tip5"]}}""",
        model="llama-3.3-70b-versatile", temp=0.1, max_tok=1200)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/local-laws', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_local_laws():
    try:
        destination = clean((request.get_json() or {}).get('destination',''))
        if not destination: return jsonify({'success':False,'error':'Destination required'})
        result = groq_json(f"""You are a legal travel expert. List important laws and rules for tourists visiting {destination}.
Return ONLY valid JSON:
{{"strict_laws":[{{"law":"law description","penalty":"penalty if broken","severity":"High/Medium/Low","icon":"emoji"}},{{"law":"law description","penalty":"penalty","severity":"High","icon":"emoji"}},{{"law":"law description","penalty":"penalty","severity":"Medium","icon":"emoji"}}],"photography_rules":["what you cannot photograph"],"dress_code_rules":["specific dress requirements"],"alcohol_rules":"alcohol laws","drug_laws":"drug law details","customs_limits":{{"cash":"max cash limit","cigarettes":"cigarette limit","alcohol":"alcohol limit","prohibited_items":["item1","item2"]}},"good_to_know":["cultural rule 1","cultural rule 2","cultural rule 3"],"legal_tip":"most important single tip for tourists"}}""",
        model="llama-3.3-70b-versatile", temp=0.1, max_tok=1500)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/jetlag', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_jetlag():
    try:
        data = request.get_json() or {}
        from_city   = clean(data.get('from_city',''))
        to_city     = clean(data.get('to_city',''))
        travel_date = clean(data.get('travel_date','upcoming'))
        if not from_city or not to_city: return jsonify({'success':False,'error':'Both cities required'})
        result = groq_json(f"""Calculate jet lag and recovery plan for travelling from {from_city} to {to_city} on {travel_date}.
Return ONLY valid JSON:
{{"from_timezone":"timezone + UTC offset","to_timezone":"timezone + UTC offset","time_difference":"X hours ahead/behind","jet_lag_severity":"Mild/Moderate/Severe","recovery_days":2,"direction":"Eastward/Westward","symptoms":["fatigue","insomnia"],"before_flight":[{{"action":"specific action","timing":"when","why":"reason"}}],"during_flight":[{{"action":"specific action","timing":"when","why":"reason"}}],"after_arrival":[{{"action":"specific action","timing":"when","why":"reason"}}],"sleep_schedule":{{"night_before_flight":"recommended bedtime","on_arrival":"recommended bedtime local time","day_2":"recommended schedule"}},"avoid":["avoid caffeine after X","avoid alcohol"],"recovery_tip":"single best tip for fast recovery"}}""",
        model="llama-3.3-70b-versatile", temp=0.1, max_tok=1200)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/festivals', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_festivals():
    try:
        data = request.get_json() or {}
        destination  = clean(data.get('destination',''))
        travel_date  = clean(data.get('travel_date','this month'))
        if not destination: return jsonify({'success':False,'error':'Destination required'})
        result = groq_json(f"""List festivals, public holidays and special events in {destination} around {travel_date}.
Return ONLY valid JSON:
{{"public_holidays":[{{"name":"holiday name","date":"date","impact":"what closes/opens","tourist_impact":"positive/negative/neutral"}}],"festivals":[{{"name":"festival name","dates":"date range","description":"brief desc","tourist_friendly":true,"special_tips":"tip","icon":"emoji"}}],"peak_season":true,"season_type":"Peak/Shoulder/Off-peak","price_impact":"prices X% higher/lower","crowd_level":"Very Crowded/Crowded/Moderate/Quiet","booking_advice":"book X weeks in advance","weather_this_month":"weather description","best_festival_tip":"top tip for experiencing local festivals"}}""",
        model="llama-3.3-70b-versatile", temp=0.2, max_tok=1200)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/budget-plan', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_budget_plan():
    try:
        data       = request.get_json() or {}
        destination = clean(data.get('destination',''))
        days        = int(data.get('days',5))
        people      = int(data.get('people',1))
        budget      = clean(data.get('budget','50000'))
        currency    = clean(data.get('currency','INR'),3)
        if not destination: return jsonify({'success':False,'error':'Destination required'})
        result = groq_json(f"""Create a detailed budget plan for {people} people visiting {destination} for {days} days. Total budget: {currency} {budget}.
Return ONLY valid JSON:
{{"total_budget":"{currency} {budget}","per_person":"{currency} X","per_day":"{currency} X","budget_tier":"Budget/Mid-range/Luxury","breakdown":{{"flights":{{"amount":"{currency} X","percentage":35,"tips":"book ahead"}},"accommodation":{{"amount":"{currency} X","percentage":25,"tips":"area tip"}},"food":{{"amount":"{currency} X","percentage":15,"tips":"where to eat"}},"transport":{{"amount":"{currency} X","percentage":10,"tips":"best transport"}},"activities":{{"amount":"{currency} X","percentage":10,"tips":"free vs paid"}},"shopping":{{"amount":"{currency} X","percentage":5,"tips":"what to buy"}}}},"daily_budget":{{"budget_day":"{currency} X - street food, hostel","comfort_day":"{currency} X - restaurant, hotel","splurge_day":"{currency} X - fine dining, premium"}},"money_saving_tips":["tip1","tip2","tip3","tip4","tip5"],"hidden_costs":["visa fee","airport tax","travel insurance"],"free_things":["free thing 1","free thing 2"],"worth_splurging":["experience worth extra"],"budget_verdict":"Is this budget realistic? One sentence."}}""",
        model="llama-3.3-70b-versatile", temp=0.1, max_tok=1500)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/passport-check', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_passport_check():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['expiry_date','travel_date','destination'])
        if not ok: return jsonify({'success':False,'error':err})
        from datetime import datetime as dt
        expiry = dt.strptime(data['expiry_date'], '%Y-%m-%d')
        travel = dt.strptime(data['travel_date'], '%Y-%m-%d')
        days_remaining    = (expiry - dt.now()).days
        days_after_travel = (expiry - travel).days
        destination = clean(data.get('destination',''))
        result = groq_json(f"""A traveller wants to visit {destination}. Passport expiry: {data['expiry_date']}. Travel date: {data['travel_date']}. Days passport is valid after travel date: {days_after_travel}. Most countries require passport valid for 6 months beyond travel date.
Return ONLY valid JSON:
{{"is_valid":true,"validity_status":"Safe/Warning/Critical/Invalid","days_remaining":{days_remaining},"days_after_travel":{days_after_travel},"destination_requirement":"X months required by {destination}","verdict":"one clear sentence verdict","action_needed":"what they must do if any","renewal_urgency":"Immediate/Soon/Not needed","renewal_time":"how long passport renewal takes in India","renewal_cost":"approximate cost in INR","tatkal_available":true,"tatkal_time":"days for tatkal","tatkal_cost":"INR amount","tips":["tip1","tip2"]}}""",
        model="llama-3.3-70b-versatile", temp=0.1, max_tok=800)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/luggage-check', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_luggage_check():
    try:
        data        = request.get_json() or {}
        airline     = clean(data.get('airline',''))
        cabin_class = clean(data.get('cabin_class','Economy'))
        destination = clean(data.get('destination',''))
        if not airline or not destination: return jsonify({'success':False,'error':'Airline and destination required'})
        result = groq_json(f"""Provide complete luggage and duty free allowance information for {airline} airline flying to {destination} in {cabin_class} class.
Return ONLY valid JSON:
{{"airline":"{airline}","cabin_class":"{cabin_class}","carry_on":{{"weight":"kg limit","dimensions":"cm","pieces":1}},"checked_baggage":{{"weight":"kg limit","dimensions":"cm","pieces":1,"extra_cost":"cost per extra kg"}},"prohibited_items":["item1","item2","item3"],"liquid_rules":"100ml rule details","duty_free_allowance":{{"alcohol":"limit","cigarettes":"limit","cash":"limit","gifts":"value limit"}},"packing_tips":["tip1","tip2","tip3"],"pro_tip":"best single tip for this airline/route"}}""",
        model="llama-3.3-70b-versatile", temp=0.1, max_tok=1000)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/emergency-card', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_emergency_card():
    try:
        data        = request.get_json() or {}
        name        = clean(data.get('name',''))
        blood_group = clean(data.get('blood_group',''))
        allergies   = clean(data.get('allergies','none'))
        destination = clean(data.get('destination',''))
        if not name or not destination: return jsonify({'success':False,'error':'Name and destination required'})
        result = groq_json(f"""Create a complete emergency contact card for {name} (blood group: {blood_group}, allergies: {allergies}) visiting {destination}.
Return ONLY valid JSON:
{{"emergency_numbers":{{"police":"local number","ambulance":"local number","fire":"local number","tourist_helpline":"local number"}},"indian_embassy":{{"address":"full embassy address","phone":"main phone number","emergency_phone":"24hr emergency number","email":"email"}},"nearest_hospitals":[{{"name":"hospital name","type":"Government/Private","phone":"number","specialty":"specialty"}}],"medical_phrases":[{{"english":"I need help","local":"translation","pronunciation":"how to say it"}},{{"english":"Call an ambulance","local":"translation","pronunciation":"pronunciation"}},{{"english":"I am allergic to {allergies}","local":"translation","pronunciation":"pronunciation"}},{{"english":"My blood group is {blood_group}","local":"translation","pronunciation":"pronunciation"}}],"what_to_do_if_robbed":["step1","step2","step3"],"what_to_do_if_sick":["step1","step2","step3"],"what_to_do_if_lost":["step1","step2","step3"]}}""",
        model="llama-3.3-70b-versatile", temp=0.1, max_tok=1500)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/allergy-card', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_allergy_card():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['name','destination'])
        if not ok: return jsonify({'success':False,'error':err})
        allergies_str = ', '.join([clean(a,50) for a in data.get('allergies',[])][:10])
        result = groq_json(f"""Allergy card for {clean(data.get('name',''),50)} allergic to {allergies_str} visiting {clean(data.get('destination',''))}. Return JSON: allergy_card_text,dangerous_dishes,safe_dishes,hidden_allergens,phrases_to_say,restaurant_tips,emergency_protocol,medicines_to_carry""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/flight-rights', methods=['POST'])
@require_auth
@limiter.limit("15 per hour")
def api_flight_rights():
    try:
        data = request.get_json() or {}
        result = groq_json(f"""Flight rights: {clean(data.get('airline',''))} on {clean(data.get('route',''))}, issue: {clean(data.get('issue',''))}, delay: {data.get('delay_hours',0)}h. Return JSON: entitled_to_compensation,compensation_amount,your_rights,immediate_actions,documents_to_collect,what_airline_must_provide,how_to_claim,exact_phrases_to_say,claim_template""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/safe-route', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_safe_route():
    try:
        data = request.get_json() or {}
        result = groq_json(f"""Safe route from {clean(data.get('from_location',''))} to {clean(data.get('to_location',''))} in {clean(data.get('destination',''))} at {clean(data.get('time_of_day',''))} for {clean(data.get('traveller_type',''))}. Return JSON: route_safety,safety_score,recommended_transport,areas_to_avoid,if_harassed,trusted_contacts,pro_tips""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/immigration-help', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_immigration_help():
    try:
        data = request.get_json() or {}
        result = groq_json(f"""Immigration guide for {clean(data.get('passport','India'))} entering {clean(data.get('destination',''))} for {clean(data.get('purpose','Tourism'))}. Return JSON: common_questions,documents_to_keep_ready,declaration_items,common_mistakes,if_stopped_for_questioning,pro_tips""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/trip-journal', methods=['POST'])
@require_auth
@limiter.limit("5 per hour")
def api_trip_journal():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success':False,'error':err})
        result = groq_json(f"""Write vivid personal travel journal. Destination:{clean(data.get('destination',''))}, {data.get('days',5)} days, with {clean(data.get('travel_with','solo'))}, vibe:{clean(data.get('vibe','adventure'))}, highlights:{clean(data.get('highlights','amazing trip'),300)}. Write in FIRST PERSON.
Return JSON: title,tagline,opening,chapters(array day/title/story/highlight/emotion/emoji),closing,best_memory,lesson_learned,quote,would_return,rating,tags""",
        model="llama-3.3-70b-versatile", temp=0.7, max_tok=3000)
        trip_id = data.get('trip_id')
        if trip_id: save_journal(trip_id, g.user_id, result)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/expense-summary', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_expense_summary():
    try:
        data     = request.get_json() or {}
        expenses = data.get('expenses',[])[:100]
        total    = sum(float(e.get('amount',0)) for e in expenses)
        by_cat   = {}
        for e in expenses:
            cat = clean(e.get('category','Other'),30)
            by_cat[cat] = by_cat.get(cat,0) + float(e.get('amount',0))
        result = groq_json(f"""Analyse trip expenses. Destination:{clean(data.get('destination',''))}, budget:{clean(data.get('currency','INR'),3)} {clean(data.get('budget',''))}, spent:{total:.0f}, categories:{by_cat}. Return JSON: total_spent,budget,status,per_person,verdict,insights,money_tips_next_trip""")
        result['by_category'] = by_cat
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/split-bill', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def api_split_bill():
    try:
        data     = request.get_json() or {}
        people   = data.get('people',[])[:20]
        expenses = data.get('expenses',[])[:100]
        currency = clean(data.get('currency','INR'),3)
        balances = {p:0 for p in people}; total = 0
        for exp in expenses:
            amount = float(exp.get('amount',0))
            paid_by = exp.get('paid_by', people[0] if people else '')
            split_between = exp.get('split_between', people) or people
            total += amount; share = amount / max(len(split_between),1)
            if paid_by in balances: balances[paid_by] += amount
            for p in split_between:
                if p in balances: balances[p] -= share
        settlements = []
        pos = sorted([(k,v) for k,v in balances.items() if v>0.01], key=lambda x:-x[1])
        neg = sorted([(k,v) for k,v in balances.items() if v<-0.01], key=lambda x:x[1])
        i=j=0
        while i<len(pos) and j<len(neg):
            creditor,credit=pos[i]; debtor,debt=neg[j]; amount=min(credit,-debt)
            if amount>0.01: settlements.append({'from':debtor,'to':creditor,'amount':round(amount,2),'currency':currency})
            pos[i]=(creditor,credit-amount); neg[j]=(debtor,debt+amount)
            if pos[i][1]<0.01: i+=1
            if neg[j][1]>-0.01: j+=1
        return jsonify({'success':True,'data':{'total':round(total,2),'per_person':round(total/max(len(people),1),2),'balances':{k:round(v,2) for k,v in balances.items()},'settlements':settlements,'currency':currency,'all_settled':len(settlements)==0}})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/trip-stats', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_trip_stats():
    try:
        data = request.get_json() or {}
        result = groq_json(f"""Fun viral trip stats for {clean(data.get('destination',''))}, {data.get('days',5)} days, {clean(data.get('travel_with','solo'))}, vibes:{clean(data.get('vibes','adventure'))}.
Return JSON: traveller_type,traveller_description,fun_stats(array label/value/icon),achievements(array title/description/icon/rarity),travel_score(0-100),instagram_caption""",
        temp=0.5, max_tok=1500)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/review-generator', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_review_generator():
    try:
        data = request.get_json() or {}
        ok, err = validate(data, ['place','experience'])
        if not ok: return jsonify({'success':False,'error':err})
        result = groq_json(f"""Write genuine {clean(data.get('platform','Google'))} review for {clean(data.get('place',''))}, rated {data.get('rating',5)}/5. Experience:{clean(data.get('experience',''),500)}. Return JSON: review_title,review_body,pros,cons,best_for,tip,short_version,hashtags""",
        temp=0.6, max_tok=1000)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/next-trip', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_next_trip():
    try:
        data = request.get_json() or {}
        result = groq_json(f"""Next trip after {clean(data.get('past_destination',''))}. Loved:{clean(data.get('loved',''),200)}. Budget:{clean(data.get('budget',''))}. Month:{clean(data.get('travel_month',''))}. Passport:{clean(data.get('passport','India'))}.
Return JSON: recommendations(destination/why_perfect/similarity_score/best_time/budget_level/estimated_cost/unique_experience/vibe/emoji/visa_for_india),travel_pattern,bucket_list_suggestion""",
        temp=0.4, max_tok=1500)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/currency-leftover', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_currency_leftover():
    try:
        data = request.get_json() or {}
        result = groq_json(f"""Options for {clean(data.get('currency',''),10)} {clean(data.get('amount',''),20)} leftover currency. Home:{clean(data.get('home_currency','INR'),3)}. Return JSON: options(option/description/estimated_value/rating/pros/cons),best_option,tips""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/packing', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_packing():
    try:
        data = request.get_json() or {}
        result = groq_json(f"""Packing list for {data.get('days',5)} days in {clean(data.get('destination',''))}. Weather:{clean(data.get('weather','moderate'))}. Style:{clean(data.get('vibe','adventure'))}.
Return JSON: essentials,clothing,toiletries,electronics,documents,health,destination_specific""")
        return jsonify({'success':True,'packing':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

# ── DISCOVER / IDENTIFY ──

@app.route('/api/identify', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_identify():
    try:
        data = request.get_json() or {}
        image_base64 = data.get('image','')
        if not image_base64: return jsonify({'success':False,'error':'No image provided'})
        if len(image_base64) > 5*1024*1024: return jsonify({'success':False,'error':'Image too large (max 5MB)'})

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_base64}"}},
                {"type":"text","text":"""Identify this location. Return ONLY valid JSON:
{"place_name":"exact name","city":"city","country":"country","confidence":92,"place_type":"type","description":"2-3 sentences","tags":["tag1"],"best_time":"months","climate":"type","budget_level":"Budget/Mid-range/Luxury","avg_daily_cost":"USD X/day","language":"language","currency":"currency","nearest_airport":"airport","why_famous":"reason","nearby":[{"name":"place","distance":"X km","type":"Attraction","icon":"emoji"}],"similar_places":[{"name":"place","country":"country","why_similar":"reason","emoji":"flag"}],"travel_tips":["tip1"],"best_food":["dish1"]}"""}
            ]}],
            temperature=0.1, max_tokens=1500
        )
        result_text = response.choices[0].message.content.strip()
        if '```' in result_text:
            for p in result_text.split('```'):
                if '{' in p: result_text = p[4:] if p.startswith('json') else p; break
        start = result_text.find('{'); end = result_text.rfind('}')+1
        if start != -1: result_text = result_text[start:end]
        result = json.loads(result_text)
        log_action(g.user_id,'identify_place',request.remote_addr)
        return jsonify({'success':True,'result':result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/identify-text', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_identify_text():
    try:
        description = clean((request.get_json() or {}).get('description',''), 300)
        if not description: return jsonify({'success':False,'error':'Description required'})
        result = groq_json(f"""Someone saw this place on social media: "{description}". Identify the exact location.
Return ONLY this JSON structure (fill in real values, no placeholders):
{{"place_name":"exact name","city":"city","country":"country","continent":"continent","confidence":88,"place_type":"Natural Wonder/Historical Site/City/Beach/Temple/etc","description":"2-3 sentences about the place","tags":["tag1","tag2","tag3"],"best_time":"best months to visit","climate":"climate type","budget_level":"Budget/Mid-range/Luxury","avg_daily_cost":"USD X/day","language":"local language","currency":"currency name + code","nearest_airport":"nearest airport name","airport_code":"IATA code","why_famous":"why this place is famous","nearby":[{{"name":"actual nearby place","distance":"X km","type":"Attraction","icon":"🏛️","description":"brief description"}},{{"name":"actual nearby place 2","distance":"X km","type":"Restaurant","icon":"🍽️","description":"brief description"}},{{"name":"actual nearby place 3","distance":"X km","type":"Hidden Gem","icon":"💎","description":"brief description"}}],"similar_places":[{{"name":"similar place name","country":"country","why_similar":"reason why similar","emoji":"🇫🇷"}},{{"name":"similar place name 2","country":"country","why_similar":"reason why similar","emoji":"🇮🇹"}},{{"name":"similar place name 3","country":"country","why_similar":"reason why similar","emoji":"🇬🇷"}}],"travel_tips":["tip1","tip2","tip3"],"best_food":["dish1","dish2","dish3"]}}""",
        model="llama-3.3-70b-versatile", temp=0.3, max_tok=2000)
        return jsonify({'success':True,'result':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

# ── CAMERA SCAN ──

def _translate_blocks_batch(blocks, target_lang, src_lang=None):
    """Translate a list of {text, vertices} blocks. Returns [{original,translated,vertices}]."""
    if not blocks: return []
    texts = [b['text'] for b in blocks]
    tgt = target_lang.lower()[:2] if len(target_lang) >= 2 else target_lang
    deepl_code = DEEPL_LANGS.get(tgt) or DEEPL_LANGS.get(target_lang)
    translated_texts = []
    if deepl_code and deepl_client:
        try:
            src = None
            if src_lang and src_lang not in ('unknown','auto',None,''):
                src = src_lang.upper()[:2]
            results = deepl_client.translate_text(texts, target_lang=deepl_code, source_lang=src)
            translated_texts = [r.text for r in results]
        except Exception as e: print(f"[DeepL batch] {e}")
    if not translated_texts:
        lang_name = LANG_NAMES.get(target_lang, LANG_NAMES.get(tgt.upper(), 'English'))
        sep = "\n[|||]\n"
        combined = sep.join(texts)
        try:
            r = groq_client.chat.completions.create(
                model='llama-3.1-8b-instant',
                messages=[
                    {'role':'system','content':f'Translate each segment to {lang_name}. Keep [|||] separators exactly. Return ONLY translations separated by [|||].'},
                    {'role':'user','content':combined}
                ], temperature=0.1, max_tokens=1500
            )
            parts = r.choices[0].message.content.strip().split('[|||]')
            translated_texts = [p.strip() for p in parts]
        except Exception as e:
            print(f"[Groq batch] {e}")
            translated_texts = texts  # fallback: no translation
    return [
        {'original': b['text'], 'translated': translated_texts[i].strip() if i < len(translated_texts) else b['text'], 'vertices': b['vertices']}
        for i, b in enumerate(blocks)
    ]

@app.route('/scan', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def scan():
    try:
        data = request.get_json() or {}
        image_data  = data.get('image','')
        target_lang = clean(data.get('target_lang','EN'),5).upper()
        if ',' in image_data: image_data = image_data.split(',')[1]
        if not image_data: return jsonify({'success':False,'error':'No image provided'})
        if len(image_data) > 5*1024*1024: return jsonify({'success':False,'error':'Image too large (max 5MB)'})

        vision_result=[None]; groq_result=[None]
        vision_done=threading.Event(); groq_done=threading.Event()

        def run_vision():
            try:
                if not GOOGLE_VISION_KEY: vision_done.set(); return
                r = req.post(
                    f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_KEY}",
                    json={"requests":[{"image":{"content":image_data},"features":[{"type":"DOCUMENT_TEXT_DETECTION"}]}]},
                    timeout=8
                )
                resp0 = r.json().get('responses',[{}])[0]
                full_text = resp0.get('fullTextAnnotation',{}).get('text','')
                if not full_text:
                    anns = resp0.get('textAnnotations',[])
                    full_text = anns[0].get('description','') if anns else ''
                pages = resp0.get('fullTextAnnotation',{}).get('pages',[])
                lang = 'unknown'
                if pages:
                    langs = pages[0].get('property',{}).get('detectedLanguages',[])
                    if langs: lang = langs[0].get('languageCode','unknown')
                # Extract paragraph-level blocks with bounding boxes
                raw_blocks = []
                for page in pages:
                    for block in page.get('blocks', []):
                        parts = []
                        for para in block.get('paragraphs', []):
                            words = []
                            for word in para.get('words', []):
                                w = ''.join(s.get('text','') for s in word.get('symbols',[]))
                                words.append(w)
                            parts.append(' '.join(words))
                        block_text = '\n'.join(parts).strip()
                        if not block_text: continue
                        verts = block.get('boundingBox',{}).get('vertices',[])
                        vertices = [[v.get('x',0), v.get('y',0)] for v in verts]
                        if vertices: raw_blocks.append({'text': block_text, 'vertices': vertices})
                # Fallback to word-level annotations if no blocks extracted
                if not raw_blocks and resp0.get('textAnnotations'):
                    for ann in resp0['textAnnotations'][1:]:
                        t = ann.get('description','').strip()
                        verts = ann.get('boundingPoly',{}).get('vertices',[])
                        vertices = [[v.get('x',0), v.get('y',0)] for v in verts]
                        if t and vertices: raw_blocks.append({'text': t, 'vertices': vertices})
                if full_text.strip():
                    vision_result[0] = (full_text.strip(), lang, raw_blocks)
            except Exception as e: print(f"[Vision] {e}")
            vision_done.set()

        def run_groq_vision():
            try:
                response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role":"user","content":[
                        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_data}"}},
                        {"type":"text","text":"Extract ALL text in this image exactly as written. Preserve line breaks. Return ONLY the raw text. If no text, return NO_TEXT."}
                    ]}], temperature=0.0, max_tokens=800
                )
                text = response.choices[0].message.content.strip()
                if text and text != 'NO_TEXT': groq_result[0] = (text, 'unknown', [])
            except Exception as e: print(f"[Groq Vision] {e}")
            groq_done.set()

        threading.Thread(target=run_vision, daemon=True).start()
        threading.Thread(target=run_groq_vision, daemon=True).start()

        extracted_text=''; detected_lang='unknown'; raw_blocks=[]
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if vision_done.is_set() and vision_result[0]:
                extracted_text, detected_lang, raw_blocks = vision_result[0]; break
            if groq_done.is_set() and groq_result[0]:
                extracted_text, detected_lang, raw_blocks = groq_result[0]; break
            time.sleep(0.05)

        if not extracted_text:
            vision_done.wait(2); groq_done.wait(2)
            if vision_result[0]: extracted_text, detected_lang, raw_blocks = vision_result[0]
            elif groq_result[0]: extracted_text, detected_lang, raw_blocks = groq_result[0]

        if not extracted_text:
            return jsonify({'success':False,'error':'No text found. Try pointing at clearer text.'})

        translated_text, engine = translate(extracted_text, target_lang, detected_lang)
        text_blocks = _translate_blocks_batch(raw_blocks, target_lang, detected_lang)
        return jsonify({
            'success':True,
            'original_text':extracted_text,
            'translated_text':translated_text,
            'detected_lang':detected_lang,
            'engine':engine,
            'text_blocks':text_blocks
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/retranslate-blocks', methods=['POST'])
@require_auth
@limiter.limit("60 per hour")
def api_retranslate_blocks():
    try:
        data = request.get_json() or {}
        blocks = data.get('blocks', [])
        target_lang = clean(data.get('target_lang','EN'), 5).upper()
        src_lang = data.get('src_lang', None)
        if not blocks: return jsonify({'success':False,'error':'No blocks provided'})
        text_blocks = _translate_blocks_batch(blocks, target_lang, src_lang)
        return jsonify({'success':True, 'text_blocks':text_blocks})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

# ══════════════════════════════════════
# WEBSOCKETS
# ══════════════════════════════════════

def process_stream(ws, audio_bytes, target_lang, src_lang, sentence_id):
    t_start = time.time()
    try:
        safe_send(ws, {'type':'status','message':'🎯 Listening...'})
        wav = audio_to_wav(bytes(audio_bytes))
        text, detected, _ = transcribe(wav, src_lang if src_lang != 'auto' else None)
        if not is_valid(text): safe_send(ws, {'type':'ready'}); return
        safe_send(ws, {'type':'transcript','text':text,'lang':detected,'id':sentence_id})
        safe_send(ws, {'type':'status','message':'🌍 Translating...'})
        translated, engine = translate(text, target_lang, detected)
        safe_send(ws, {'type':'translation','text':translated,'engine':engine,'lang':target_lang,'id':sentence_id})
        safe_send(ws, {'type':'status','message':'🔊 Speaking...'})
        audio_data = tts(translated, target_lang)
        safe_send(ws, {'type':'audio','data':base64.b64encode(audio_data).decode(),'id':sentence_id})
        safe_send(ws, {'type':'ready'})
        print(f"[Stream #{sentence_id}] {time.time()-t_start:.2f}s")
    except Exception as e:
        safe_send(ws, {'type':'error','message':str(e)})
        safe_send(ws, {'type':'ready'})

@sock.route('/stream')
def stream_ws(ws):
    target_lang='HI'; src_lang='auto'
    audio_buffer=bytearray(); silent_chunks=0; speaking=False; sentence_id=0; processing=False
    SILENCE_THRESHOLD=450; SILENCE_CHUNKS_NEEDED=2; MIN_BYTES=int(16000*2*0.25)
    while True:
        try:
            msg = ws.receive()
            if msg is None: break
            if isinstance(msg, str):
                try:
                    cfg = json.loads(msg)
                    if 'target_lang' in cfg: target_lang = cfg['target_lang']
                    if 'src_lang' in cfg: src_lang = cfg['src_lang']
                except: pass
                continue
            chunk=bytes(msg); rms=get_rms(chunk)
            safe_send(ws, {'type':'volume','level':min(100,int(rms/35))})
            if rms >= SILENCE_THRESHOLD:
                if not speaking: speaking=True; safe_send(ws, {'type':'speaking','status':True})
                silent_chunks=0; audio_buffer.extend(chunk)
            elif speaking:
                silent_chunks+=1; audio_buffer.extend(chunk)
                if silent_chunks >= SILENCE_CHUNKS_NEEDED:
                    if len(audio_buffer) >= MIN_BYTES and not processing:
                        sentence_id+=1; processing=True; buf_copy=bytearray(audio_buffer)
                        t = threading.Thread(target=process_stream, args=(ws,buf_copy,target_lang,src_lang,sentence_id), daemon=True)
                        t.start(); t.join(); processing=False
                    audio_buffer=bytearray(); silent_chunks=0; speaking=False
                    safe_send(ws, {'type':'speaking','status':False})
        except Exception as e: print(f"[Stream] {e}"); break

@sock.route('/convo-ws')
def convo_ws(ws):
    lang_a='en'; lang_b='hi'; active_speaker='A'
    audio_buffer=bytearray(); silent_chunks=0; speaking=False; msg_id=0
    SILENCE_THRESHOLD=450; MIN_BYTES=int(16000*2*0.4)
    while True:
        try:
            msg = ws.receive()
            if msg is None: break
            if isinstance(msg, str):
                try:
                    cfg = json.loads(msg)
                    if 'lang_a' in cfg: lang_a=cfg['lang_a'].lower()[:2]
                    if 'lang_b' in cfg: lang_b=cfg['lang_b'].lower()[:2]
                    if 'speaker' in cfg:
                        active_speaker=cfg['speaker']; audio_buffer=bytearray(); silent_chunks=0; speaking=False
                        safe_send(ws, {'type':'speaker_changed','speaker':active_speaker})
                except: pass
                continue
            chunk=bytes(msg); rms=get_rms(chunk)
            safe_send(ws, {'type':'volume','level':min(100,int(rms/35)),'speaker':active_speaker})
            if rms >= SILENCE_THRESHOLD:
                if not speaking: speaking=True; safe_send(ws, {'type':'speaking','status':True,'speaker':active_speaker})
                silent_chunks=0; audio_buffer.extend(chunk)
            elif speaking:
                silent_chunks+=1; audio_buffer.extend(chunk)
                src = lang_a if active_speaker=='A' else lang_b
                silence_needed = 5 if src in SLOW_LANGS else 3
                if silent_chunks >= silence_needed:
                    if len(audio_buffer) >= MIN_BYTES:
                        msg_id+=1; tgt = lang_b if active_speaker=='A' else lang_a
                        try:
                            safe_send(ws, {'type':'status','message':'🎯 Listening...'})
                            wav = audio_to_wav(bytes(audio_buffer))
                            text, detected, _ = transcribe(wav, WHISPER_LANG.get(src))
                            if is_valid(text):
                                safe_send(ws, {'type':'transcript','text':text,'speaker':active_speaker,'lang':detected,'id':msg_id})
                                safe_send(ws, {'type':'status','message':'🌍 Translating...'})
                                translated, engine = translate(text, tgt, src)
                                safe_send(ws, {'type':'translation','text':translated,'speaker':active_speaker,'engine':engine,'id':msg_id})
                                safe_send(ws, {'type':'status','message':'🔊 Speaking...'})
                                audio_data = tts(translated, tgt)
                                safe_send(ws, {'type':'audio','data':base64.b64encode(audio_data).decode(),'speaker':active_speaker,'id':msg_id})
                        except Exception as e: safe_send(ws, {'type':'error','message':str(e)})
                        safe_send(ws, {'type':'ready'})
                    audio_buffer=bytearray(); silent_chunks=0; speaking=False
                    safe_send(ws, {'type':'speaking','status':False,'speaker':active_speaker})
        except Exception as e: print(f"[Convo] {e}"); break


# ══════════════════════════════════════════════════════════════
# PASTE THIS ENTIRE BLOCK INTO app.py
# Right before:  if __name__ == '__main__':
# ══════════════════════════════════════════════════════════════

import json as _json


def clean_json(text):
    """Clean and parse JSON from Groq response."""
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
    return _json.loads(text)


# ──────────────────────────────────────────────────────────────
# MULTI-CITY TRIP PLANNER
# ──────────────────────────────────────────────────────────────
@app.route('/api/multi-city-plan', methods=['POST'])
def multi_city_plan():
    try:
        data         = request.get_json()
        origin       = data.get('origin', 'India')
        cities       = data.get('cities', [])
        total_budget = data.get('total_budget', 100000)
        currency     = data.get('currency', 'INR')
        vibe         = data.get('vibe', 'adventure')
        people       = data.get('people', 1)
        start_date   = data.get('start_date', '')
        passport     = data.get('passport', 'India')

        if not cities or len(cities) < 2:
            return jsonify({'success': False, 'error': 'Please add at least 2 cities.'})

        if len(cities) > 6:
            return jsonify({'success': False, 'error': 'Maximum 6 cities supported.'})

        total_days = sum(int(c.get('days', 3)) for c in cities)
        city_list  = ', '.join(c['name'] for c in cities)
        import json as _json

        prompt = (
            "You are the world's best multi-city trip planner.\n\n"
            "Plan an epic multi-city trip:\n"
            "- Origin: " + str(origin) + "\n"
            "- Cities in order: " + str(city_list) + "\n"
            "- Cities data: " + str(_json.dumps(cities)) + "\n"
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
            "5. Activities must match the " + str(vibe) + " travel style\n"
            "6. Include hidden gems, must eat, must do for each city\n\n"
            "Return ONLY valid JSON with this exact structure:\n"
            '{\n'
            '  "trip_title": "catchy name for this trip",\n'
            '  "origin": "' + str(origin) + '",\n'
            '  "total_days": ' + str(total_days) + ',\n'
            '  "total_budget": ' + str(total_budget) + ',\n'
            '  "currency": "' + str(currency) + '",\n'
            '  "cities_count": ' + str(len(cities)) + ',\n'
            '  "route_overview": "one paragraph describing the full journey",\n'
            '  "smart_suggestions": ["tip1", "tip2", "tip3", "tip4"],\n'
            '  "budget_split": {\n'
            '    "flights_and_transit": 0,\n'
            '    "accommodation": 0,\n'
            '    "food": 0,\n'
            '    "activities": 0,\n'
            '    "local_transport": 0,\n'
            '    "shopping": 0,\n'
            '    "miscellaneous": 0\n'
            '  },\n'
            '  "cities": [\n'
            '    {\n'
            '      "city_number": 1,\n'
            '      "city": "city name",\n'
            '      "country": "country name",\n'
            '      "days": 3,\n'
            '      "city_budget": 0,\n'
            '      "city_vibe": "what makes this city special",\n'
            '      "best_area_to_stay": "neighbourhood recommendation",\n'
            '      "weather_note": "weather during travel period",\n'
            '      "language": "local language",\n'
            '      "local_currency": "currency name and code",\n'
            '      "currency_tip": "best way to get local currency",\n'
            '      "itinerary": [\n'
            '        {\n'
            '          "day": 1,\n'
            '          "day_label": "Day 1 - City Name",\n'
            '          "theme": "theme for this day",\n'
            '          "morning": {"activity": "", "location": "", "cost": "", "tip": ""},\n'
            '          "afternoon": {"activity": "", "location": "", "cost": "", "tip": ""},\n'
            '          "evening": {"activity": "", "location": "", "cost": "", "tip": ""},\n'
            '          "lunch": {"restaurant": "", "cuisine": "", "cost": ""},\n'
            '          "dinner": {"restaurant": "", "cuisine": "", "cost": ""},\n'
            '          "accommodation": {"name": "", "area": "", "cost": ""}\n'
            '        }\n'
            '      ],\n'
            '      "hidden_gems": [{"name": "", "why": "", "cost": "", "best_time": ""}],\n'
            '      "must_eat": ["dish1", "dish2", "dish3"],\n'
            '      "must_do": ["activity1", "activity2", "activity3"],\n'
            '      "local_tips": ["tip1", "tip2", "tip3"]\n'
            '    }\n'
            '  ],\n'
            '  "transit_plans": [\n'
            '    {\n'
            '      "from": "city1",\n'
            '      "to": "city2",\n'
            '      "transit_day": "day number",\n'
            '      "options": [\n'
            '        {\n'
            '          "mode": "Flight/Train/Bus",\n'
            '          "operator": "operator name",\n'
            '          "duration": "travel time",\n'
            '          "cost": "per person cost",\n'
            '          "total_cost": "cost for all people",\n'
            '          "comfort": "Comfortable/Basic/Luxury",\n'
            '          "recommended": true,\n'
            '          "reason": "why recommended",\n'
            '          "booking_tip": "where to book"\n'
            '        }\n'
            '      ],\n'
            '      "transit_tip": "specific tip for this transit"\n'
            '    }\n'
            '  ],\n'
            '  "sim_strategy": {\n'
            '    "recommendation": "best SIM strategy for this route",\n'
            '    "options": [\n'
            '      {\n'
            '        "type": "Global eSIM/Local SIM/Roaming",\n'
            '        "providers": ["provider1", "provider2"],\n'
            '        "cost": "approximate total cost",\n'
            '        "coverage": "which cities covered",\n'
            '        "recommended": true,\n'
            '        "why": "reason"\n'
            '      }\n'
            '    ]\n'
            '  },\n'
            '  "packing_for_route": {\n'
            '    "weather_variation": "temperature range across all cities",\n'
            '    "key_items": ["item1", "item2", "item3", "item4", "item5"],\n'
            '    "luggage_tip": "advice on luggage for this specific trip"\n'
            '  },\n'
            '  "money_saving_tips": ["tip1", "tip2", "tip3"]\n'
            '}\n'
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are the world's best multi-city travel planner. Return ONLY valid JSON. No markdown. No backticks. No extra text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=8000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            parts = result.split('```')
            for part in parts:
                if '{' in part:
                    result = part
                    if result.startswith('json'):
                        result = result[4:]
                    break
        start = result.find('{')
        end   = result.rfind('}') + 1
        if start != -1 and end > start:
            result = result[start:end]

        plan = _json.loads(result)
        return jsonify({'success': True, 'plan': plan})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ──────────────────────────────────────────────────────────────
# LOCAL SIM GUIDE
# ──────────────────────────────────────────────────────────────
@app.route('/api/sim-guide', methods=['POST'])
def sim_guide():
    try:
        data             = request.get_json()
        destination      = data.get('destination', '')
        origin           = data.get('origin', 'India')
        days             = data.get('days', 7)
        data_needs       = data.get('data_needs', 'moderate')
        phone_type       = data.get('phone_type', 'unlocked')
        budget_conscious = data.get('budget_conscious', True)
        countries        = data.get('countries', [destination])

        if not destination:
            return jsonify({'success': False, 'error': 'Destination required'})

        countries_str = ', '.join(countries) if isinstance(countries, list) else destination

        prompt = """You are an expert in international mobile connectivity and SIM cards.

A traveller from {origin} is visiting {countries_str} for {days} days.
Phone type: {phone_type}
Data usage: {data_needs} (light=social media only, moderate=maps and streaming, heavy=video calls and remote work)
Budget conscious: {budget_conscious}

Give a COMPLETE, ACCURATE, ACTIONABLE SIM card guide with real carrier names and real prices.

Return ONLY valid JSON:
{{
  "destination": "{destination}",
  "days": {days},
  "data_recommendation": "how much data they need for {days} days with {data_needs} usage",
  "top_recommendation": {{
    "name": "specific SIM plan name",
    "provider": "real carrier name",
    "type": "Physical SIM/eSIM",
    "cost": "exact price in local currency and INR",
    "data": "exact GB",
    "validity": "days",
    "calls": "calls included or not",
    "why_best": "why this is best for their situation",
    "where_to_buy": "exact location - airport terminal, convenience store, carrier store",
    "activation": "how to activate step by step",
    "coverage": "4G/5G coverage quality",
    "hotspot": "hotspot allowed or not",
    "esim_compatible": true
  }},
  "all_options": [
    {{
      "rank": 1,
      "name": "plan name",
      "provider": "carrier",
      "type": "Physical/eSIM",
      "cost": "price",
      "data": "GB",
      "validity": "days",
      "best_for": "who this is best for",
      "buy_at": "where to buy",
      "pros": ["pro1", "pro2"],
      "cons": ["con1"],
      "score": 9
    }}
  ],
  "esim_options": [
    {{
      "provider": "Airalo/Holafly/Nomad",
      "plan_name": "plan name",
      "cost_usd": "USD price",
      "cost_inr": "INR price",
      "data": "GB",
      "validity": "days",
      "recommended": true,
      "setup_time": "minutes to set up",
      "note": "important note"
    }}
  ],
  "airport_buying_guide": {{
    "available_at_airport": true,
    "airport_terminal": "which terminal or exit",
    "price_difference": "how much more expensive at airport",
    "recommendation": "buy at airport or wait for city",
    "timing": "when to buy"
  }},
  "roaming_option": {{
    "worth_it": false,
    "jio_international": "Jio pack details and cost in INR",
    "airtel_international": "Airtel pack details and cost in INR",
    "vi_international": "Vi pack details and cost in INR",
    "verdict": "roaming vs local SIM comparison"
  }},
  "connectivity_tips": [
    "tip1", "tip2", "tip3", "tip4", "tip5"
  ],
  "data_saving_tips": [
    "tip1", "tip2", "tip3"
  ],
  "offline_essentials": [
    {{"app": "Google Maps", "action": "download {destination} map offline before you leave", "data_saved": "saves data"}},
    {{"app": "app name", "action": "action", "data_saved": "saving"}}
  ],
  "emergency_connectivity": {{
    "if_sim_fails": "what to do",
    "free_wifi_spots": "where to find free wifi in {destination}",
    "emergency_call": "can you call without SIM"
  }},
  "phone_unlock_check": "how to check if phone is unlocked",
  "budget_summary": {{
    "cheapest_option": "option name and price",
    "best_value": "option name and price",
    "premium_option": "option name and price",
    "recommended_for_this_trip": "option name"
  }}
}}""".format(
            origin=origin,
            countries_str=countries_str,
            days=days,
            phone_type=phone_type,
            data_needs=data_needs,
            budget_conscious=budget_conscious,
            destination=destination
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert in international SIM cards. Always give specific real carrier names and real prices. Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=3000
        )

        guide = clean_json(response.choices[0].message.content)
        return jsonify({'success': True, 'guide': guide})

    except Exception as e:
        print(f"SIM guide error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ──────────────────────────────────────────────────────────────
# OFFLINE ITINERARY DOWNLOAD
# ──────────────────────────────────────────────────────────────
@app.route('/download-itinerary', methods=['POST'])
def download_itinerary():
    try:
        from flask import make_response
        data       = request.get_json()
        plan       = data.get('plan', {})
        multi_city = data.get('multi_city', False)

        if multi_city:
            html = _build_multi_city_html(plan)
        else:
            html = _build_single_city_html(plan)

        dest     = plan.get('destination') or plan.get('trip_title', 'Trip')
        filename = 'yaply_' + dest.replace(' ', '_').lower() + '_itinerary.html'

        response = make_response(html)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Content-Disposition'] = 'attachment; filename="' + filename + '"'
        return response

    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({'success': False, 'error': str(e)})


def _slot(emoji, label, slot):
    if not slot or not slot.get('activity'):
        return ''
    tip_html = '<div style="font-size:11px;color:#1A8A72;margin-top:4px;font-style:italic;border-left:2px solid #1A8A72;padding-left:8px;">&#128161; ' + slot.get('tip', '') + '</div>' if slot.get('tip') else ''
    return (
        '<div style="background:#F7F6F2;border-radius:10px;padding:12px;margin-bottom:8px;">'
        '<div style="font-size:10px;font-weight:700;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">' + emoji + ' ' + label + '</div>'
        '<div style="font-weight:700;font-size:14px;">' + slot.get('activity', '') + '</div>'
        '<div style="font-size:12px;color:#1A8A72;margin-top:2px;">&#128205; ' + slot.get('location', '') + '</div>'
        '<div style="font-size:12px;font-weight:700;color:#28B06A;margin-top:4px;">&#128176; ' + slot.get('cost', '') + ' &middot; &#9200; ' + slot.get('duration', '') + '</div>'
        + tip_html +
        '</div>'
    )


def _meal(emoji, label, meal):
    if not meal or not meal.get('restaurant'):
        return ''
    return (
        '<div style="background:#FFF7ED;border-radius:10px;padding:10px 12px;margin-bottom:8px;">'
        '<div style="font-size:10px;font-weight:700;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">' + emoji + ' ' + label + '</div>'
        '<div style="font-weight:700;font-size:14px;">' + meal.get('restaurant', '') + '</div>'
        '<div style="font-size:12px;color:#6B6860;">&#127869;&#65039; ' + meal.get('cuisine', '') + ' &middot; &#128176; ' + meal.get('cost', '') + '</div>'
        '</div>'
    )


def _stay(stay):
    if not stay or not stay.get('name'):
        return ''
    return (
        '<div style="background:#EFF6FF;border-radius:10px;padding:10px 12px;margin-bottom:8px;">'
        '<div style="font-size:10px;font-weight:700;color:#3A6BC8;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">&#127968; Stay</div>'
        '<div style="font-weight:700;font-size:14px;">' + stay.get('name', '') + '</div>'
        '<div style="font-size:12px;color:#6B6860;">&#128205; ' + stay.get('area', '') + ' &middot; &#128176; ' + stay.get('cost', '') + ' / night</div>'
        '</div>'
    )


def _build_single_city_html(plan):
    destination = plan.get('destination', 'Your Trip')
    days        = plan.get('days', 0)
    currency    = plan.get('currency', '')
    itinerary   = plan.get('itinerary', [])
    budget_tips = plan.get('budget_tips', [])
    packing     = plan.get('packing_list', [])
    phrases     = plan.get('local_phrases', [])
    gems        = plan.get('hidden_gems', [])
    emergency   = plan.get('emergency_numbers', {})
    tips        = plan.get('tips', [])
    visa        = plan.get('visa_info', {})
    flight      = plan.get('flight_info', {})
    sim         = plan.get('sim_internet', {})
    cultural    = plan.get('cultural_guide', {})

    # Day cards
    day_cards_html = ''
    for day in itinerary:
        day_cards_html += (
            '<div style="background:white;border-radius:16px;padding:20px;margin-bottom:12px;border-left:4px solid #1A8A72;box-shadow:0 2px 8px rgba(0,0,0,0.06);">'
            '<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">'
            '<div style="width:36px;height:36px;border-radius:50%;background:#1A8A72;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;color:white;flex-shrink:0;">' + str(day.get('day', '')) + '</div>'
            '<div style="font-weight:700;font-size:16px;">' + (day.get('title', '')) + '</div>'
            '</div>'
            + _slot('&#127749;', 'Morning', day.get('morning'))
            + _meal('&#9728;&#65039;', 'Lunch', day.get('lunch'))
            + _slot('&#9728;&#65039;', 'Afternoon', day.get('afternoon'))
            + _slot('&#127750;', 'Evening', day.get('evening'))
            + _meal('&#127769;', 'Dinner', day.get('dinner'))
            + _stay(day.get('accommodation'))
            + '</div>'
        )

    # Packing tags
    packing_html = ''.join(
        '<span style="display:inline-block;background:#EFF6FF;color:#1A8A72;border-radius:20px;padding:4px 12px;font-size:12px;margin:3px;font-weight:500;">' + item + '</span>'
        for item in packing
    )

    # Phrases
    phrase_html = ''
    for p in phrases:
        phrase_html += (
            '<div style="display:flex;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid #F0EBE0;flex-wrap:wrap;">'
            '<span style="font-weight:600;min-width:120px;font-size:13px;">' + p.get('phrase', '') + '</span>'
            '<span style="color:#1A8A72;font-size:14px;font-weight:600;">' + p.get('translation', '') + '</span>'
            '<span style="color:#6B6860;font-size:12px;font-style:italic;">(' + p.get('pronunciation', '') + ')</span>'
            '</div>'
        )

    # Gems
    gems_html = ''
    for g in gems:
        gems_html += (
            '<div style="background:#F7F6F2;border-radius:12px;padding:14px;margin-bottom:10px;">'
            '<div style="font-weight:700;font-size:14px;margin-bottom:4px;">&#128142; ' + g.get('name', '') + '</div>'
            '<div style="font-size:12px;color:#3D3730;">' + g.get('description', '') + '</div>'
            '<div style="font-size:11px;color:#6B6860;margin-top:6px;">&#128205; ' + g.get('location', '') + ' &middot; &#8987; ' + g.get('best_time', '') + ' &middot; &#128176; ' + g.get('cost', '') + '</div>'
            '</div>'
        )

    # Cultural
    dos   = ''.join('<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">&#9989; ' + d + '</li>' for d in cultural.get('dos', []))
    donts = ''.join('<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">&#10060; ' + d + '</li>' for d in cultural.get('donts', []))

    # Budget tips
    btips = ''.join('<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">&#128161; ' + t + '</li>' for t in budget_tips)

    # Tips
    tips_html = ''.join('<li style="padding:6px 0;font-size:13px;border-bottom:1px solid #F0EBE0;">&#127919; ' + t + '</li>' for t in tips)

    # Emergency
    em_html = ''
    for k, num in emergency.items():
        em_html += (
            '<div style="background:#FEF2F2;border-radius:10px;padding:12px;text-align:center;">'
            '<div style="font-size:20px;font-weight:700;color:#D84C3E;">' + str(num) + '</div>'
            '<div style="font-size:11px;color:#6B6860;margin-top:2px;">' + k.replace('_', ' ').title() + '</div>'
            '</div>'
        )

    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{dest} &#8212; Yaply Offline Itinerary</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#F7F6F2; color:#2C2B28; }}
.header {{ background:linear-gradient(135deg,#1A8A72,#3A6BC8); color:white; padding:32px 24px; text-align:center; }}
.header h1 {{ font-size:28px; font-weight:800; letter-spacing:-1px; }}
.badge {{ display:inline-block; background:rgba(255,255,255,.2); border-radius:20px; padding:4px 12px; font-size:12px; margin:6px 4px 0; }}
.container {{ max-width:800px; margin:0 auto; padding:20px 16px 60px; }}
.section {{ background:white; border-radius:16px; padding:20px; margin-bottom:14px; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
.section-title {{ font-size:15px; font-weight:800; color:#1A8A72; margin-bottom:14px; }}
.info-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.info-item {{ background:#F7F6F2; border-radius:10px; padding:12px; }}
.info-label {{ font-size:10px; color:#6B6860; text-transform:uppercase; letter-spacing:1px; margin-bottom:3px; }}
.info-value {{ font-size:13px; font-weight:700; }}
ul {{ padding:0; list-style:none; }}
.offline-note {{ background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px; padding:12px 16px; margin-bottom:16px; font-size:12px; color:#1D4ED8; text-align:center; }}
.footer {{ text-align:center; padding:24px; color:#6B6860; font-size:12px; }}
@media print {{ .header {{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }} }}
</style>
</head>
<body>
<div class="header">
  <div style="font-size:11px;opacity:.6;letter-spacing:2px;margin-bottom:4px;">YAPLY &middot; AI TRAVEL OS</div>
  <h1>&#9992;&#65039; {dest}</h1>
  <div style="opacity:.8;font-size:13px;margin-top:6px;">{days}-Day Complete Itinerary</div>
  <div>
    <span class="badge">&#128197; {days} Days</span>
    <span class="badge">&#128176; {currency}</span>
    <span class="badge">&#127760; Offline Ready</span>
  </div>
</div>
<div class="container">
  <div class="offline-note">&#128241; Works completely offline &mdash; save to your phone before you travel. Open in any browser, no internet needed.</div>

  <div class="section">
    <div class="section-title">&#8505;&#65039; Trip Essentials</div>
    <div class="info-grid">
      <div class="info-item"><div class="info-label">Best Time</div><div class="info-value">{best_time}</div></div>
      <div class="info-item"><div class="info-label">Language</div><div class="info-value">{language}</div></div>
      <div class="info-item"><div class="info-label">Timezone</div><div class="info-value">{timezone}</div></div>
      <div class="info-item"><div class="info-label">Currency</div><div class="info-value">{local_currency}</div></div>
    </div>
  </div>

  {flight_section}
  {visa_section}
  {sim_section}

  <div style="font-size:14px;font-weight:800;color:#2C2B28;margin-bottom:12px;">&#128197; Day by Day Itinerary</div>
  {day_cards}

  {gems_section}

  {phrases_section}

  {cultural_section}

  {emergency_section}

  {packing_section}

  {tips_section}

  {btips_section}

</div>
<div class="footer">Generated by <strong>Yaply</strong> &mdash; Your Complete Travel OS &mdash; <a href="https://yaply.live" style="color:#1A8A72;">yaply.live</a></div>
</body>
</html>'''.format(
        dest=destination,
        days=days,
        currency=currency,
        best_time=plan.get('best_time_to_visit', ''),
        language=plan.get('language', ''),
        timezone=plan.get('timezone', ''),
        local_currency=plan.get('currency', ''),
        flight_section=(
            '<div class="section"><div class="section-title">&#9992;&#65039; Flight Info</div><div class="info-grid">'
            '<div class="info-item"><div class="info-label">Cost</div><div class="info-value">' + flight.get('estimated_cost','') + '</div></div>'
            '<div class="info-item"><div class="info-label">Duration</div><div class="info-value">' + flight.get('flight_duration','') + '</div></div>'
            '<div class="info-item"><div class="info-label">Airlines</div><div class="info-value">' + ', '.join(flight.get('best_airlines',[])) + '</div></div>'
            '<div class="info-item"><div class="info-label">Book Ahead</div><div class="info-value">' + flight.get('best_time_to_book','') + '</div></div>'
            '</div></div>'
        ) if flight else '',
        visa_section=(
            '<div class="section"><div class="section-title">&#128706; Visa</div><div class="info-grid">'
            '<div class="info-item"><div class="info-label">Required</div><div class="info-value">' + ('Yes' if visa.get('required') else 'No') + '</div></div>'
            '<div class="info-item"><div class="info-label">Type</div><div class="info-value">' + visa.get('type','') + '</div></div>'
            '<div class="info-item"><div class="info-label">Cost</div><div class="info-value">' + visa.get('cost','') + '</div></div>'
            '<div class="info-item"><div class="info-label">Processing</div><div class="info-value">' + visa.get('processing_time','') + '</div></div>'
            '</div></div>'
        ) if visa else '',
        sim_section=(
            '<div class="section"><div class="section-title">&#128241; SIM Card</div><div class="info-grid">'
            '<div class="info-item"><div class="info-label">Best Option</div><div class="info-value">' + sim.get('best_option','') + '</div></div>'
            '<div class="info-item"><div class="info-label">Cost</div><div class="info-value">' + sim.get('cost','') + '</div></div>'
            '<div class="info-item"><div class="info-label">Data</div><div class="info-value">' + sim.get('data','') + '</div></div>'
            '<div class="info-item"><div class="info-label">Where to Buy</div><div class="info-value">' + sim.get('where_to_buy','') + '</div></div>'
            '</div></div>'
        ) if sim else '',
        day_cards=day_cards_html,
        gems_section=(
            '<div class="section"><div class="section-title">&#128142; Hidden Gems</div>' + gems_html + '</div>'
        ) if gems_html else '',
        phrases_section=(
            '<div class="section"><div class="section-title">&#128172; Essential Phrases</div>' + phrase_html + '</div>'
        ) if phrase_html else '',
        cultural_section=(
            '<div class="section"><div class="section-title">&#127758; Cultural Guide</div>'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:12px;">'
            '<div><div style="font-size:11px;font-weight:700;color:#28B06A;margin-bottom:8px;">DOS</div><ul>' + dos + '</ul></div>'
            '<div><div style="font-size:11px;font-weight:700;color:#D84C3E;margin-bottom:8px;">DON\'TS</div><ul>' + donts + '</ul></div>'
            '</div>'
            '<div class="info-grid">'
            '<div class="info-item"><div class="info-label">Dress Code</div><div class="info-value" style="font-size:12px;">' + cultural.get('dress_code','') + '</div></div>'
            '<div class="info-item"><div class="info-label">Tipping</div><div class="info-value" style="font-size:12px;">' + cultural.get('tipping','') + '</div></div>'
            '</div></div>'
        ) if cultural else '',
        emergency_section=(
            '<div class="section"><div class="section-title">&#128680; Emergency Numbers</div>'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;">' + em_html + '</div></div>'
        ) if em_html else '',
        packing_section=(
            '<div class="section"><div class="section-title">&#129523; Packing List</div>' + packing_html + '</div>'
        ) if packing_html else '',
        tips_section=(
            '<div class="section"><div class="section-title">&#127919; Pro Tips</div><ul>' + tips_html + '</ul></div>'
        ) if tips_html else '',
        btips_section=(
            '<div class="section"><div class="section-title">&#128176; Budget Tips</div><ul>' + btips + '</ul></div>'
        ) if btips else '',
    )


def _build_multi_city_html(plan):
    title    = plan.get('trip_title', 'Multi-City Trip')
    cities   = plan.get('cities', [])
    transits = plan.get('transit_plans', [])
    currency = plan.get('currency', 'INR')
    sim      = plan.get('sim_strategy', {})
    packing  = plan.get('packing_for_route', {})
    suggs    = plan.get('smart_suggestions', [])
    budget   = plan.get('budget_split', {})
    MC_COLS  = ['#1A8A72','#3A6BC8','#E8823A','#8B5CF6','#28B06A','#F0B429']

    sugg_html = ''.join(
        '<li style="padding:6px 0;font-size:13px;border-bottom:1px solid rgba(26,138,114,.1);">&#129504; ' + s + '</li>'
        for s in suggs
    )

    budget_html = ''
    for k, v in budget.items():
        budget_html += (
            '<div style="background:#F7F6F2;border-radius:10px;padding:12px;">'
            '<div style="font-size:10px;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">' + k.replace('_', ' ').title() + '</div>'
            '<div style="font-size:13px;font-weight:700;color:#1A8A72;">' + currency + ' ' + str(int(v or 0)) + '</div>'
            '</div>'
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
                '<div style="width:26px;height:26px;border-radius:50%;background:' + col + ';display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:white;flex-shrink:0;">' + str(day.get('day', '')) + '</div>'
                '<div style="font-size:14px;font-weight:700;">' + (day.get('day_label') or day.get('theme') or 'Day ' + str(day.get('day', ''))) + '</div>'
                '</div>'
                + _slot('&#127749;', 'Morning', day.get('morning'))
                + _meal('&#9728;&#65039;', 'Lunch', day.get('lunch'))
                + _slot('&#9728;&#65039;', 'Afternoon', day.get('afternoon'))
                + _slot('&#127750;', 'Evening', day.get('evening'))
                + _meal('&#127769;', 'Dinner', day.get('dinner'))
                + '</div>'
            )

        gems_html = ''
        for g in gems:
            gems_html += (
                '<div style="background:#F7F6F2;border-radius:10px;padding:11px;margin-bottom:7px;">'
                '<div style="font-weight:700;font-size:13px;">&#128142; ' + g.get('name', '') + '</div>'
                '<div style="font-size:12px;color:#3D3730;margin-top:3px;">' + (g.get('why') or g.get('description', '')) + '</div>'
                '</div>'
            )

        tips_html = ''
        for i, tip in enumerate(tips):
            tips_html += (
                '<div style="display:flex;gap:8px;padding:8px 0;border-bottom:1px solid #F0EBE0;">'
                '<div style="width:20px;height:20px;border-radius:50%;background:#2C2B28;color:white;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;">' + str(i+1) + '</div>'
                '<div style="font-size:13px;">' + tip + '</div>'
                '</div>'
            )

        cities_html += (
            '<div style="background:white;border-radius:16px;margin-bottom:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.06);">'
            '<div style="background:' + col + ';padding:18px 20px;">'
            '<div style="font-size:11px;color:rgba(255,255,255,.6);letter-spacing:1px;margin-bottom:4px;">CITY ' + str(idx+1) + '</div>'
            '<div style="font-size:20px;font-weight:800;color:white;">' + city.get('city', '') + ', ' + city.get('country', '') + '</div>'
            '<div style="font-size:12px;color:rgba(255,255,255,.6);margin-top:4px;">' + str(city.get('days', '')) + ' days &middot; ' + currency + ' ' + str(int(city.get('city_budget', 0))) + ' &middot; ' + city.get('best_area_to_stay', '') + '</div>'
            '</div>'
            '<div style="padding:16px 18px;">'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">'
            '<div style="background:#F7F6F2;border-radius:10px;padding:10px;"><div style="font-size:10px;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Language</div><div style="font-size:13px;font-weight:700;">' + city.get('language', '-') + '</div></div>'
            '<div style="background:#F7F6F2;border-radius:10px;padding:10px;"><div style="font-size:10px;color:#6B6860;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Currency</div><div style="font-size:13px;font-weight:700;">' + city.get('local_currency', '-') + '</div></div>'
            '</div>'
            + (('<div style="background:#EFF6FF;border-radius:10px;padding:9px 12px;margin-bottom:12px;font-size:12px;color:#3A6BC8;">&#127780;&#65039; ' + city.get('weather_note', '') + '</div>') if city.get('weather_note') else '')
            + (('<div style="font-size:11px;font-weight:700;color:#6B6860;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;">Daily Plan</div>' + day_html) if day_html else '')
            + (('<div style="font-size:11px;font-weight:700;color:#6B6860;text-transform:uppercase;letter-spacing:.8px;margin:10px 0 8px;">Hidden Gems</div>' + gems_html) if gems_html else '')
            + (('<div style="font-size:11px;font-weight:700;color:#6B6860;text-transform:uppercase;letter-spacing:.8px;margin:10px 0 8px;">Local Tips</div>' + tips_html) if tips_html else '')
            + '</div></div>'
        )

        # Transit after this city
        transit = next((t for t in transits if t.get('from') and (t['from'] == city.get('city') or city.get('city','').split(',')[0] in t['from'])), None)
        if transit and idx < len(cities) - 1:
            opts = transit.get('options', [])
            opt_html = ''
            for opt in opts[:3]:
                opt_html += (
                    '<div style="background:' + ('#EFF6FF' if opt.get('recommended') else '#F7F6F2') + ';border-radius:10px;padding:10px 12px;margin-bottom:6px;display:flex;align-items:center;gap:10px;">'
                    '<div style="flex:1;">'
                    '<div style="font-size:13px;font-weight:700;">' + opt.get('mode', '') + ' &middot; ' + opt.get('operator', '') + '</div>'
                    '<div style="font-size:11px;color:#6B6860;">&#9201; ' + opt.get('duration', '') + ' &middot; ' + opt.get('comfort', '') + '</div>'
                    '</div>'
                    + ('<span style="background:#1A8A72;color:white;font-size:10px;font-weight:700;padding:2px 8px;border-radius:100px;">BEST</span>' if opt.get('recommended') else '')
                    + '<div style="font-size:14px;font-weight:800;color:#1A8A72;">' + currency + ' ' + str(opt.get('total_cost', opt.get('cost', '?'))) + '</div>'
                    '</div>'
                )
            cities_html += (
                '<div style="background:white;border-radius:12px;border:1.5px solid #EFEDE8;padding:14px 16px;margin-bottom:12px;">'
                '<div style="font-size:12px;font-weight:700;margin-bottom:8px;">&#9992;&#65039; ' + transit.get('from', '') + ' &#8594; ' + transit.get('to', '') + '</div>'
                + opt_html
                + (('<div style="background:#EFEDE8;border-radius:8px;padding:8px 10px;font-size:12px;color:#2C2B28;margin-top:6px;">&#128161; ' + transit.get('transit_tip', '') + '</div>') if transit.get('transit_tip') else '')
                + '</div>'
            )

    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} &#8212; Yaply Offline</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#F7F6F2; color:#2C2B28; }}
.header {{ background:linear-gradient(135deg,#1e2a3a,#0d1520); color:white; padding:28px 24px; text-align:center; }}
.container {{ max-width:800px; margin:0 auto; padding:20px 16px 60px; }}
.section {{ background:white; border-radius:16px; padding:20px; margin-bottom:14px; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
.section-title {{ font-size:15px; font-weight:800; color:#1A8A72; margin-bottom:14px; }}
.offline-note {{ background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px; padding:12px 16px; margin-bottom:16px; font-size:12px; color:#1D4ED8; text-align:center; }}
.footer {{ text-align:center; padding:24px; color:#6B6860; font-size:12px; }}
</style>
</head>
<body>
<div class="header">
  <div style="font-size:11px;opacity:.5;letter-spacing:2px;margin-bottom:6px;">YAPLY &middot; AI TRAVEL OS</div>
  <h1 style="font-size:24px;font-weight:800;letter-spacing:-1px;">&#128507; {title}</h1>
  <div style="opacity:.6;font-size:13px;margin-top:6px;">{total_days} days &middot; {city_count} cities &middot; {currency} {total_budget}</div>
</div>
<div class="container">
  <div class="offline-note">&#128241; Works completely offline &mdash; save to your phone before you travel.</div>

  {sugg_section}
  {budget_section}
  {cities_html}
  {sim_section}
  {packing_section}

</div>
<div class="footer">Generated by <strong>Yaply</strong> &mdash; <a href="https://yaply.live" style="color:#1A8A72;">yaply.live</a></div>
</body>
</html>'''.format(
        title=title,
        total_days=plan.get('total_days', ''),
        city_count=len(cities),
        currency=currency,
        total_budget=str(int(plan.get('total_budget', 0))),
        sugg_section=(
            '<div class="section"><div class="section-title">&#129504; Smart Suggestions</div><ul style="padding:0;list-style:none;">' + sugg_html + '</ul></div>'
        ) if sugg_html else '',
        budget_section=(
            '<div class="section"><div class="section-title">&#128176; Budget Split</div>'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">' + budget_html + '</div></div>'
        ) if budget_html else '',
        cities_html=cities_html,
        sim_section=(
            '<div class="section"><div class="section-title">&#128241; SIM Strategy</div>'
            '<div style="background:#EFF6FF;border-radius:10px;padding:10px 12px;font-size:13px;color:#1A8A72;">' + sim.get('recommendation', '') + '</div></div>'
        ) if sim.get('recommendation') else '',
        packing_section=(
            '<div class="section"><div class="section-title">&#129523; Packing for This Route</div>'
            '<div style="font-size:13px;color:#2C2B28;margin-bottom:10px;">&#127777;&#65039; ' + packing.get('weather_variation', '') + '</div>'
            '<div>' + ''.join('<span style="display:inline-block;background:#EFF6FF;color:#1A8A72;border-radius:20px;padding:4px 12px;font-size:12px;margin:3px;font-weight:500;">' + item + '</span>' for item in packing.get('key_items', [])) + '</div>'
            '<div style="font-size:12px;color:#6B6860;margin-top:10px;">&#129523; ' + packing.get('luggage_tip', '') + '</div></div>'
        ) if packing.get('key_items') else '',
    )

@app.route('/diary')
def diary_page():
    return render_template('yaply_diary.html')

@app.route('/groups')
@app.route('/groups/<int:group_id>')
def groups_page(group_id=None):
    return render_template('yaply_groups.html')

# ══════════════════════════════════════
# STARTUP
# ══════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=port,
        allow_unsafe_werkzeug=True
    )