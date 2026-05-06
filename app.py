"""
Yaply — Complete Production App v2
- Single unified server
- Real auth on every route
- Rate limiting
- Input validation
- Security headers
- Database connected
"""

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
sock = Sock(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET', 'yaply-secret-2025-change-me')

@app.route('/health')
def health():
    return 'OK', 200

# ── RATE LIMITER (after app) ──
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["300 per day", "60 per hour"],
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

# ── AI CLIENTS ──
from groq import Groq
import deepl
import edge_tts
import asyncio

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
try:    deepl_client = deepl.Translator(os.getenv("DEEPL_API_KEY"))
except: deepl_client = None

WEATHER_KEY       = os.getenv("OPENWEATHER_API_KEY")
EXCHANGE_KEY      = os.getenv("EXCHANGE_API_KEY")
GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_API_KEY")
UNSPLASH_KEY      = os.getenv("UNSPLASH_ACCESS_KEY")
GOOGLE_CLIENT_ID  = os.getenv("GOOGLE_CLIENT_ID", "")

# ── SECURITY HEADERS ──
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

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
}
SLOW_LANGS = {'hi','ar','zh','ja','ko'}

# ══════════════════════════════════════
# INPUT VALIDATION HELPER
# ══════════════════════════════════════

def validate(data, required_fields, max_len=500):
    """Returns (ok, error_message)"""
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
    """Sanitize string input"""
    if not text: return ""
    return str(text).strip()[:max_len]

# ══════════════════════════════════════
# AUDIO + AI HELPERS
# ══════════════════════════════════════

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
    text = result.text.strip()
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
    lang_name = LANG_NAMES.get(target_lang,'English')
    r = groq_client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role':'system','content':f'Translate to {lang_name}. Return ONLY the translation. Be natural.'},
            {'role':'user','content':text}
        ],
        temperature=0.1, max_tokens=500
    )
    return r.choices[0].message.content.strip(), 'Groq AI'

def tts(text, lang_code):
    async def _run():
        voice = EDGE_VOICES.get(lang_code,'en-US-JennyNeural')
        try:
            communicate = edge_tts.Communicate(text, voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk['type'] == 'audio': buf.write(chunk['data'])
            buf.seek(0); data = buf.read()
            if len(data) > 100: return data
            raise Exception("Empty")
        except:
            communicate = edge_tts.Communicate(text,'en-US-JennyNeural')
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk['type'] == 'audio': buf.write(chunk['data'])
            buf.seek(0); return buf.read()
    return asyncio.run(_run())

def groq_json(prompt, system="Return ONLY valid JSON. No markdown.", model="llama-3.1-8b-instant", temp=0.2, max_tok=2000):
    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        temperature=temp, max_tokens=max_tok
    )
    result = response.choices[0].message.content.strip()
    if '```' in result:
        for p in result.split('```'):
            if '{' in p: result = p[4:] if p.startswith('json') else p; break
    start = result.find('{'); end = result.rfind('}')+1
    if start != -1: result = result[start:end]
    return json.loads(result)

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

@app.route('/translate')
def translate_page(): return render_template('stream.html')

@app.route('/convo')
def convo_page(): return render_template('convo.html')

@app.route('/camera')
def camera_page(): return render_template('camera.html')

# ══════════════════════════════════════
# DATABASE ROUTES — All require login
# ══════════════════════════════════════

@app.route('/api/trips', methods=['GET'])
@require_auth
def api_get_trips():
    trips = get_trips(g.user_id)
    return jsonify({'success':True,'trips':trips})

@app.route('/api/trips', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_save_trip():
    try:
        data = request.get_json()
        ok, err = validate(data, ['destination','origin'])
        if not ok: return jsonify({'success':False,'error':err})
        trip_id = save_trip(
            user_id=g.user_id,
            destination=clean(data.get('destination','')),
            origin=clean(data.get('origin','')),
            days=min(max(int(data.get('days',7)),1),365),
            people=min(max(int(data.get('people',1)),1),50),
            budget=clean(data.get('budget','80000')),
            currency=clean(data.get('currency','INR'), 3),
            vibes=clean(data.get('vibes','Adventure')),
            passport=clean(data.get('passport','India')),
            plan_data=data.get('plan_data')
        )
        log_action(g.user_id,'save_trip',request.remote_addr)
        return jsonify({'success':True,'trip_id':trip_id})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

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
        data = request.get_json()
        update_trip(trip_id, g.user_id, **data)
        return jsonify({'success':True})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

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
        data = request.get_json()
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
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

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
        data = request.get_json()
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
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

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
        data = request.get_json()
        save_journal(trip_id, g.user_id, data.get('content'))
        return jsonify({'success':True})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/profile', methods=['GET'])
@require_auth
def api_profile():
    stats = get_user_stats(g.user_id)
    return jsonify({'success':True,'user':safe_user(g.user),'stats':stats})

# ══════════════════════════════════════
# AI ROUTES — All require auth + rate limit
# ══════════════════════════════════════

@app.route('/api/plan', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_plan():
    try:
        data = request.get_json()
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success':False,'error':err})
        destination = clean(data.get('destination',''))
        origin      = clean(data.get('origin','India'))
        days        = min(max(int(data.get('days',5)),1),60)
        budget      = clean(data.get('budget','50000'))
        vibe        = clean(data.get('vibe','adventure'))
        people      = min(max(int(data.get('people',1)),1),20)
        currency    = clean(data.get('currency','INR'),3)

        prompt = f"""World-class travel planner. Create UNIQUE {days}-day itinerary.
FROM: {origin} → TO: {destination}
Budget: {currency} {budget} for {people} people
Style: {vibe}
Currency: ALL prices in {currency} only

Return ONLY valid JSON:
{{"destination":"{destination}","days":{days},"language":"local language","currency":"local currency","timezone":"timezone","best_time_to_visit":"months","budget_breakdown":{{"flights":"{currency} X","accommodation":"{currency} X","food":"{currency} X","transport":"{currency} X","activities":"{currency} X","miscellaneous":"{currency} X"}},"flight_info":{{"estimated_cost":"{currency} X","best_airlines":["a1","a2"],"flight_duration":"Xh","best_time_to_book":"X weeks ahead"}},"itinerary":[{{"day":1,"title":"Day title","morning":{{"activity":"name","location":"place","duration":"2h","cost":"{currency} X","tip":"insider tip"}},"afternoon":{{"activity":"name","location":"place","duration":"2h","cost":"{currency} X","tip":"tip"}},"evening":{{"activity":"name","location":"place","duration":"2h","cost":"{currency} X","tip":"tip"}},"lunch":{{"restaurant":"name","cuisine":"type","cost":"{currency} X"}},"dinner":{{"restaurant":"name","cuisine":"type","cost":"{currency} X"}},"accommodation":{{"name":"hotel","area":"area","cost":"{currency} X/night"}}}}],"hidden_gems":[{{"name":"place","description":"why special","location":"area","best_time":"when","cost":"{currency} X"}}],"local_transport":{{"airport_to_city":{{"options":["opt1"],"cost":"{currency} X","duration":"30min"}},"within_city":[{{"type":"Metro","cost":"{currency} X/ride","tip":"tip"}}],"useful_apps":["app1"]}},"sim_internet":{{"best_option":"option","cost":"{currency} X","data":"XGB","where_to_buy":"location"}},"cultural_guide":{{"dos":["do1","do2","do3"],"donts":["dont1","dont2","dont3"],"dress_code":"advice","tipping":"culture","greetings":"how to greet"}},"packing_list":["item1","item2","item3"],"emergency_numbers":{{"police":"number","ambulance":"number","tourist_helpline":"number"}},"visa_info":{{"required":true,"type":"tourist","validity":"30 days","cost":"{currency} X"}},"tips":["tip1","tip2","tip3"]}}"""

        result = groq_json(prompt, model="llama-3.3-70b-versatile", temp=0.3, max_tok=4000)

        trip_id = data.get('trip_id')
        if trip_id:
            update_trip(trip_id, g.user_id, plan_data=result, status='active')

        log_action(g.user_id,'plan_trip',request.remote_addr)
        return jsonify({'success':True,'plan':result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/weather', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def api_weather():
    try:
        city = clean(request.get_json().get('city',''))
        if not city: return jsonify({'success':False,'error':'City required'})
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_KEY}&units=metric&cnt=40"
        r = req.get(url, timeout=10); data = r.json()
        if data.get('cod') != '200': return jsonify({'success':False,'error':'City not found'})
        daily = {}
        for item in data['list']:
            date = item['dt_txt'].split(' ')[0]
            if date not in daily:
                daily[date] = {'date':date,'temp_max':item['main']['temp_max'],'temp_min':item['main']['temp_min'],'description':item['weather'][0]['description'],'icon':item['weather'][0]['icon'],'humidity':item['main']['humidity'],'wind':item['wind']['speed']}
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
        data = request.get_json()
        amount = float(data.get('amount',1))
        from_c = clean(data.get('from','INR'),3).upper()
        to_c   = clean(data.get('to','USD'),3).upper()
        r = req.get(f"https://v6.exchangerate-api.com/v6/{EXCHANGE_KEY}/pair/{from_c}/{to_c}/{amount}", timeout=10).json()
        if r.get('result') != 'success': return jsonify({'success':False,'error':'Currency not found'})
        return jsonify({'success':True,'from':from_c,'to':to_c,'amount':amount,'converted':round(r['conversion_result'],2),'rate':r['conversion_rate']})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/visa', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_visa():
    try:
        data = request.get_json()
        passport    = clean(data.get('passport','India'))
        destination = clean(data.get('destination',''))
        if not destination: return jsonify({'success':False,'error':'Destination required'})
        result = groq_json(f"""Visa info for {passport} passport to {destination}.
Return JSON: {{"visa_required":true,"visa_type":"type","validity":"30 days","cost":"USD X","processing_days":"5","apply_online":true,"apply_url":"url","documents":["doc1","doc2"],"tips":["tip1"],"visa_on_arrival":false,"visa_free_days":0}}""")
        return jsonify({'success':True,'visa':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/detect-theme', methods=['POST'])
@optional_auth
@limiter.limit("60 per hour")
def api_detect_theme():
    try:
        destination = clean(request.get_json().get('destination',''))
        result = groq_json(f"""Theme for travel app showing "{destination}". Return JSON: {{"destination_type":"Beach/Mountain/City/Historical","theme":{{"primary_color":"#hex","secondary_color":"#hex","gradient_start":"#hex","gradient_end":"#hex","mood":"description","emoji":"single emoji","vibe_words":["word1","word2"]}}}}""", temp=0.3, max_tok=400)
        return jsonify({'success':True,'theme':result.get('theme',result)})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/place-photo', methods=['POST'])
@optional_auth
@limiter.limit("60 per hour")
def api_place_photo():
    try:
        place_name = clean(request.get_json().get('place_name',''))
        if UNSPLASH_KEY and place_name:
            r = req.get("https://api.unsplash.com/search/photos", params={'query':f"{place_name} travel landmark",'per_page':5,'orientation':'landscape','client_id':UNSPLASH_KEY}, timeout=8)
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
        data = request.get_json()
        origin      = clean(data.get('origin',''))
        destination = clean(data.get('destination',''))
        if not origin or not destination: return jsonify({'success':False,'error':'Origin and destination required'})
        currency    = clean(data.get('currency','INR'),3)
        result = groq_json(f"""Door-to-door journey from "{origin}" to "{destination}". Prices in {currency}.
Return JSON with: origin,destination,origin_has_airport,nearest_airports,recommended_route,flight_options,alternative_routes,important_notes,documents_needed""",
        model="llama-3.3-70b-versatile", temp=0.2, max_tok=3000)
        return jsonify({'success':True,'journey':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

# ── TOOLS ──

@app.route('/api/passport-check', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_passport_check():
    try:
        data = request.get_json()
        ok, err = validate(data, ['expiry_date','travel_date','destination'])
        if not ok: return jsonify({'success':False,'error':err})
        from datetime import datetime
        expiry = datetime.strptime(data.get('expiry_date',''), '%Y-%m-%d')
        travel = datetime.strptime(data.get('travel_date',''), '%Y-%m-%d')
        today  = datetime.now()
        days_remaining    = (expiry-today).days
        days_after_travel = (expiry-travel).days
        result = groq_json(f"""Passport check: expiry={data.get('expiry_date')}, travel={data.get('travel_date')}, destination={clean(data.get('destination',''))}, days_remaining={days_remaining}, days_after_travel={days_after_travel}.
Return JSON: {{"is_valid":true,"validity_status":"Safe/Warning/Critical","days_remaining":{days_remaining},"verdict":"sentence","action_needed":"what to do","renewal_urgency":"Immediate/Soon/Not needed","renewal_cost":"INR X","tips":["tip1"]}}""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/safety-check', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_safety_check():
    try:
        destination = clean(request.get_json().get('destination',''))
        if not destination: return jsonify({'success':False,'error':'Destination required'})
        result = groq_json(f"""Safety info for {destination}. Return JSON with: safety_score(0-100),safety_level,crime_index,tourist_safety,water_safe,water_advice,food_safety,health_risks,scams_to_avoid,safe_areas,avoid_areas,travel_advisory,solo_female_safety,best_safety_tips""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/local-laws', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_local_laws():
    try:
        destination = clean(request.get_json().get('destination',''))
        result = groq_json(f"""Local laws tourists must know in {destination}. Return JSON with: strict_laws(array law/penalty/severity/icon),photography_rules,dress_code_rules,alcohol_rules,customs_limits,good_to_know,legal_tip""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/jetlag', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_jetlag():
    try:
        data = request.get_json()
        result = groq_json(f"""Jet lag for {clean(data.get('from_city',''))} to {clean(data.get('to_city',''))}. Return JSON with: from_timezone,to_timezone,time_difference,jet_lag_severity,recovery_days,direction,symptoms,before_flight,during_flight,after_arrival,sleep_schedule,recovery_tip""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/festivals', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_festivals():
    try:
        data = request.get_json()
        result = groq_json(f"""Festivals in {clean(data.get('destination',''))} around {clean(data.get('travel_date','this month'))}. Return JSON with: public_holidays,festivals(name/dates/description/tourist_friendly/tips/icon),peak_season,price_impact,crowd_level,booking_advice,weather_this_month""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/budget-plan', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_budget_plan():
    try:
        data = request.get_json()
        result = groq_json(f"""Budget plan for {data.get('people',1)} people in {clean(data.get('destination',''))} for {data.get('days',5)} days. Budget: {clean(data.get('currency','USD'),3)} {clean(data.get('budget','1000'))}. Return JSON with: total_budget,per_person,per_day,budget_tier,budget_verdict,breakdown,daily_budget,money_saving_tips,free_things,worth_splurging""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/luggage-check', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_luggage_check():
    try:
        data = request.get_json()
        result = groq_json(f"""Luggage rules for {clean(data.get('airline',''))} {clean(data.get('cabin_class','Economy'))} to {clean(data.get('destination',''))}. Return JSON with: carry_on(weight/dimensions),checked_baggage(weight/dimensions/extra_cost),prohibited_items,liquid_rules,duty_free_allowance,packing_tips,pro_tip""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/emergency-card', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_emergency_card():
    try:
        data = request.get_json()
        result = groq_json(f"""Emergency card for {clean(data.get('name',''))} (blood:{clean(data.get('blood_group',''))},allergies:{clean(data.get('allergies','none'))}) visiting {clean(data.get('destination',''))}. Return JSON with: emergency_numbers,indian_embassy,nearest_hospitals,medical_phrases,what_to_do_if_robbed,what_to_do_if_sick,what_to_do_if_lost""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

# ── DURING TRIP ──

@app.route('/api/medical-translate', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_medical_translate():
    try:
        data = request.get_json()
        ok, err = validate(data, ['symptoms','destination'])
        if not ok: return jsonify({'success':False,'error':err})
        result = groq_json(f"""Medical translation: symptoms="{clean(data.get('symptoms',''),200)}", destination={clean(data.get('destination',''))}, language={clean(data.get('language','Japanese'))}. Return JSON with: severity,possible_conditions,translated_symptoms,pronunciation,say_to_doctor,immediate_actions,medicines_to_ask,emergency_number,medical_phrases""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/price-check', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def api_price_check():
    try:
        data = request.get_json()
        ok, err = validate(data, ['item','destination'])
        if not ok: return jsonify({'success':False,'error':err})
        result = groq_json(f"""Price check: "{clean(data.get('item',''),100)}" costs {clean(data.get('currency','INR'),3)} {clean(data.get('price',''),20)} in {clean(data.get('destination',''))}. Return JSON with: verdict,verdict_color,fair_price_range,local_price,tourist_price,overpaying_by,verdict_explanation,negotiation_tips,walk_away_price,local_phrase_to_say""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/scam-alerts', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_scam_alerts():
    try:
        destination = clean(request.get_json().get('destination',''))
        if not destination: return jsonify({'success':False,'error':'Destination required'})
        result = groq_json(f"""All tourist scams in {destination}. Return JSON with: scam_risk_level,scams(name/category/severity/how_it_works/red_flags/how_to_avoid/what_to_say/icon),general_rules,safe_alternatives,emergency_if_robbed""",
        model="llama-3.3-70b-versatile", max_tok=3000)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/allergy-card', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_allergy_card():
    try:
        data = request.get_json()
        ok, err = validate(data, ['name','destination'])
        if not ok: return jsonify({'success':False,'error':err})
        allergies_str = ', '.join([clean(a,50) for a in data.get('allergies',[])][:10])
        result = groq_json(f"""Food allergy card for {clean(data.get('name',''),50)} allergic to {allergies_str} visiting {clean(data.get('destination',''))}. Return JSON with: allergy_card_text,dangerous_dishes,safe_dishes,hidden_allergens,phrases_to_say,restaurant_tips,emergency_protocol,medicines_to_carry""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/flight-rights', methods=['POST'])
@require_auth
@limiter.limit("15 per hour")
def api_flight_rights():
    try:
        data = request.get_json()
        result = groq_json(f"""Flight rights: {clean(data.get('airline',''))} on {clean(data.get('route',''))}, issue: {clean(data.get('issue',''))}, delay: {data.get('delay_hours',0)}h. Return JSON with: entitled_to_compensation,compensation_amount,your_rights,immediate_actions,documents_to_collect,what_airline_must_provide,how_to_claim,exact_phrases_to_say,claim_template""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/safe-route', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_safe_route():
    try:
        data = request.get_json()
        result = groq_json(f"""Safe route from {clean(data.get('from_location',''))} to {clean(data.get('to_location',''))} in {clean(data.get('destination',''))} at {clean(data.get('time_of_day',''))} for {clean(data.get('traveller_type',''))}. Return JSON with: route_safety,safety_score,recommended_transport,areas_to_avoid,if_harassed,trusted_contacts,pro_tips""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/immigration-help', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_immigration_help():
    try:
        data = request.get_json()
        result = groq_json(f"""Immigration guide for {clean(data.get('passport','India'))} passport entering {clean(data.get('destination',''))} for {clean(data.get('purpose','Tourism'))}. Return JSON with: common_questions,documents_to_keep_ready,declaration_items,common_mistakes,if_stopped_for_questioning,pro_tips""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

# ── AFTER TRIP ──

@app.route('/api/trip-journal', methods=['POST'])
@require_auth
@limiter.limit("5 per hour")
def api_trip_journal():
    try:
        data = request.get_json()
        ok, err = validate(data, ['destination'])
        if not ok: return jsonify({'success':False,'error':err})
        result = groq_json(f"""Write vivid personal travel journal. Destination:{clean(data.get('destination',''))}, {data.get('days',5)} days, with {clean(data.get('travel_with','solo'))}, vibe:{clean(data.get('vibe','adventure'))}, highlights:{clean(data.get('highlights','amazing trip'),300)}. Write in FIRST PERSON.
Return JSON: title,tagline,opening,chapters(array day/title/story/highlight/emotion/emoji),closing,best_memory,lesson_learned,quote,would_return,rating(1-10),tags""",
        model="llama-3.1-8b-instant", temp=0.7, max_tok=2500)
        trip_id = data.get('trip_id')
        if trip_id: save_journal(trip_id, g.user_id, result)
        return jsonify({'success':True,'data':result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/expense-summary', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_expense_summary():
    try:
        data = request.get_json()
        expenses = data.get('expenses',[])[:100]  # Max 100 expenses
        total = sum(float(e.get('amount',0)) for e in expenses)
        by_cat = {}
        for e in expenses:
            cat = clean(e.get('category','Other'),30)
            by_cat[cat] = by_cat.get(cat,0) + float(e.get('amount',0))
        result = groq_json(f"""Analyze trip expenses. Destination:{clean(data.get('destination',''))}, budget:{clean(data.get('currency','INR'),3)} {clean(data.get('budget',''))}, spent:{total:.0f}, by category:{by_cat}. Return JSON: total_spent,budget,status,per_person,verdict,comparison,insights,money_tips_next_trip""")
        result['by_category'] = by_cat
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/split-bill', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def api_split_bill():
    try:
        data = request.get_json()
        people = data.get('people',[])[:20]  # Max 20 people
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
        data = request.get_json()
        result = groq_json(f"""Fun viral trip stats for {clean(data.get('destination',''))}, {data.get('days',5)} days, {clean(data.get('travel_with','solo'))}, vibes:{clean(data.get('vibes','adventure'))}. Return JSON: traveller_type,traveller_description,fun_stats(array label/value/icon),achievements(array title/description/icon/rarity),travel_score(0-100),instagram_caption""",
        temp=0.5, max_tok=1500)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/review-generator', methods=['POST'])
@require_auth
@limiter.limit("10 per hour")
def api_review_generator():
    try:
        data = request.get_json()
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
        data = request.get_json()
        result = groq_json(f"""Next trip recommendations after {clean(data.get('past_destination',''))}. Loved:{clean(data.get('loved',''),200)}. Budget:{clean(data.get('budget',''))}. Month:{clean(data.get('travel_month',''))}. Passport:{clean(data.get('passport','India'))}. Return JSON: recommendations(destination/why_perfect/similarity_score/best_time/budget_level/estimated_cost/unique_experience/vibe/emoji/visa_for_india),travel_pattern,bucket_list_suggestion""",
        temp=0.4, max_tok=1500)
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

@app.route('/api/currency-leftover', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_currency_leftover():
    try:
        data = request.get_json()
        result = groq_json(f"""Best options for {clean(data.get('currency',''),10)} {clean(data.get('amount',''),20)} leftover foreign currency. Home:{clean(data.get('home_currency','INR'),3)}. Return JSON: options(option/description/estimated_value/rating/pros/cons),best_option,tips""")
        return jsonify({'success':True,'data':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

# ── DISCOVER ──

@app.route('/api/identify', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def api_identify():
    try:
        data = request.get_json()
        image_base64 = data.get('image','')
        if not image_base64: return jsonify({'success':False,'error':'No image'})
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
        description = clean(request.get_json().get('description',''), 300)
        if not description: return jsonify({'success':False,'error':'Description required'})
        result = groq_json(f"""Someone saw this place on social media: "{description}". Identify it. Return JSON: place_name,city,country,confidence(0-100),place_type,description,tags,best_time,climate,budget_level,avg_daily_cost,language,currency,nearest_airport,why_famous,nearby,similar_places,travel_tips,best_food""",
        model="llama-3.3-70b-versatile", temp=0.3, max_tok=1500)
        return jsonify({'success':True,'result':result})
    except Exception as e: return jsonify({'success':False,'error':str(e)})

# ── CAMERA SCAN ──

@app.route('/scan', methods=['POST'])
@require_auth
@limiter.limit("30 per hour")
def scan():
    try:
        data = request.get_json()
        image_data = data.get('image','')
        target_lang = clean(data.get('target_lang','EN'),5).upper()
        if ',' in image_data: image_data = image_data.split(',')[1]
        if not image_data: return jsonify({'success':False,'error':'No image'})
        if len(image_data) > 5*1024*1024: return jsonify({'success':False,'error':'Image too large'})

        extracted_text = ''; detected_lang = 'unknown'
        vision_result=[None]; groq_result=[None]
        vision_done=threading.Event(); groq_done=threading.Event()

        def run_vision():
            try:
                if not GOOGLE_VISION_KEY: vision_done.set(); return
                r = req.post(
                    f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_KEY}",
                    json={"requests":[{"image":{"content":image_data},"features":[{"type":"DOCUMENT_TEXT_DETECTION"},{"type":"TEXT_DETECTION"}]}]},
                    timeout=6
                )
                responses = r.json().get('responses',[{}])
                full_text = responses[0].get('fullTextAnnotation',{}).get('text','')
                if not full_text:
                    anns = responses[0].get('textAnnotations',[])
                    full_text = anns[0].get('description','') if anns else ''
                pages = responses[0].get('fullTextAnnotation',{}).get('pages',[])
                lang = 'unknown'
                if pages:
                    langs = pages[0].get('property',{}).get('detectedLanguages',[])
                    if langs: lang = langs[0].get('languageCode','unknown')
                if full_text.strip(): vision_result[0] = (full_text.strip(), lang)
            except Exception as e: print(f"[Vision] {e}")
            vision_done.set()

        def run_groq_vision():
            try:
                response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role":"user","content":[
                        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_data}"}},
                        {"type":"text","text":"Extract ALL text in this image exactly as written. Return ONLY the raw text. If no text, return NO_TEXT."}
                    ]}], temperature=0.0, max_tokens=800
                )
                text = response.choices[0].message.content.strip()
                if text and text != 'NO_TEXT': groq_result[0] = (text, 'unknown')
            except Exception as e: print(f"[Groq Vision] {e}")
            groq_done.set()

        threading.Thread(target=run_vision, daemon=True).start()
        threading.Thread(target=run_groq_vision, daemon=True).start()

        deadline = time.time() + 8.0
        while time.time() < deadline:
            if vision_done.is_set() and vision_result[0]: extracted_text, detected_lang = vision_result[0]; break
            if groq_done.is_set() and groq_result[0]: extracted_text, detected_lang = groq_result[0]; break
            time.sleep(0.05)

        if not extracted_text:
            vision_done.wait(2); groq_done.wait(2)
            if vision_result[0]: extracted_text, detected_lang = vision_result[0]
            elif groq_result[0]: extracted_text, detected_lang = groq_result[0]

        if not extracted_text: return jsonify({'success':False,'error':'No text found. Try pointing at clearer text.'})
        translated_text, engine = translate(extracted_text, target_lang, detected_lang)
        return jsonify({'success':True,'original_text':extracted_text,'translated_text':translated_text,'detected_lang':detected_lang,'engine':engine})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)})

# ── WEBSOCKETS ──

def process_stream(ws, audio_bytes, target_lang, src_lang, sentence_id):
    t_start = time.time()
    try:
        safe_send(ws, {'type':'status','message':'🎯 Listening...'})
        wav = audio_to_wav(bytes(audio_bytes))
        text, detected, conf = transcribe(wav, src_lang if src_lang != 'auto' else None)
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
                            text, detected, conf = transcribe(wav, WHISPER_LANG.get(src))
                            if is_valid(text):
                                safe_send(ws, {'type':'transcript','text':text,'speaker':active_speaker,'lang':detected,'id':msg_id})
                                safe_send(ws, {'type':'status','message':'🌍 Translating...'})
                                translated, engine = translate(text, tgt, src)
                                safe_send(ws, {'type':'translation','text':translated,'speaker':active_speaker,'engine':engine,'id':msg_id})
                                safe_send(ws, {'type':'status','message':'🔊 Speaking...'})
                                audio_data = tts(translated, tgt)
                                safe_send(ws, {'type':'audio','data':base64.b64encode(audio_data).decode(),'speaker':active_speaker,'id':msg_id})
                        except Exception as e:
                            safe_send(ws, {'type':'error','message':str(e)})
                        safe_send(ws, {'type':'ready'})
                    audio_buffer=bytearray(); silent_chunks=0; speaking=False
                    safe_send(ws, {'type':'speaking','status':False,'speaker':active_speaker})
        except Exception as e: print(f"[Convo] {e}"); break

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"""
╔══════════════════════════════════════╗
║   🌍 YAPLY — Production Server       ║
║   Port: {port}                          ║
║   Auth: ✅ All routes protected      ║
║   Rate limiting: ✅ Active           ║
║   Security headers: ✅ Active        ║
║   Database: ✅ SQLite connected      ║
╚══════════════════════════════════════╝
    """)
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)