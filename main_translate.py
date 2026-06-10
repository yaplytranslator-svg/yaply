"""
Yaply — Translation Engine v3
Upgrades:
  - JWT auth on all routes (same secret as main app)
  - Usage limit enforcement (translation, voice, identify)
  - Daily usage tracking via shared SQLite DB
  - Rate limiting per user
  - Proper error responses matching main app format
  - Scout model for identify, 70b for translation fallback
  - CORS locked to yaply.live
"""
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from flask_sock import Sock
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from app import check_feature_limit
import deepl, edge_tts, asyncio
import os, io, base64, json, wave, struct, threading, time
import requests as req
from dotenv import load_dotenv
import jwt as pyjwt
import sqlite3

load_dotenv()

# ══════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app, origins=[
    'https://www.yaply.live',
    'https://yaply.live',
    'http://localhost:5000',
    'http://localhost:5001',
])
sock = Sock(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per hour"],
    storage_uri="memory://"
)

# ══════════════════════════════════════════════════════════
# CLIENTS
# ══════════════════════════════════════════════════════════
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
try:
    deepl_client = deepl.Translator(os.getenv("DEEPL_API_KEY"))
except:
    deepl_client = None

GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_API_KEY")
JWT_SECRET        = os.getenv("JWT_SECRET", "yaply_secret_key_change_in_prod")
DB_PATH           = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yaply.db')

# ══════════════════════════════════════════════════════════
# MODEL CONSTANTS
# ══════════════════════════════════════════════════════════
SCOUT    = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_70B = "llama-3.3-70b-versatile"
WHISPER_TURBO = "whisper-large-v3-turbo"
WHISPER_LARGE = "whisper-large-v3"

# ══════════════════════════════════════════════════════════
# FREE / PRO LIMITS (mirrors main app)
# ══════════════════════════════════════════════════════════
FREE_LIMITS = {
    'translations_day': 10,
    'voice_day':         5,
    'identify_day':      3,
}
PRO_LIMITS = {
    'translations_day': 100,
    'voice_day':         50,
    'identify_day':      20,
}

# ══════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_user_from_token(token):
    """Decode JWT and return user row from DB"""
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        user_id = payload.get('user_id')
        if not user_id:
            return None
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
        conn.close()
        return dict(user) if user else None
    except:
        return None

def get_token_from_request():
    """Extract Bearer token from Authorization header or query param"""
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        return auth[7:]
    return request.args.get('token', '')

def get_user_plan(user):
    """Check if user is pro and not expired"""
    from datetime import datetime
    if not user:
        return 'free'
    plan = user.get('plan_type', 'free')
    expires = user.get('pro_expires_at')
    if plan == 'pro' and expires:
        if datetime.now().isoformat() > expires:
            # Auto expire
            try:
                conn = get_db()
                conn.execute(
                    "UPDATE users SET plan_type='free', is_pro=0 WHERE id=?",
                    (user['id'],)
                )
                conn.commit()
                conn.close()
            except:
                pass
            return 'free'
    return plan

def reset_daily_if_needed(user_id):
    """Reset daily counters if new day"""
    from datetime import date
    today = date.today().isoformat()
    try:
        conn = get_db()
        row = conn.execute(
            'SELECT usage_reset_date FROM users WHERE id=?', (user_id,)
        ).fetchone()
        if row and dict(row).get('usage_reset_date') != today:
            conn.execute(
                """UPDATE users SET
                   translations_today=0, voice_today=0,
                   identify_today=0, tools_today=0,
                   usage_reset_date=?
                   WHERE id=?""",
                (today, user_id)
            )
            conn.commit()
        conn.close()
    except:
        pass

def check_and_increment(user_id, feature, plan):
    """
    Returns (allowed: bool, used: int, limit: int)
    Increments counter if allowed.
    """
    limits = PRO_LIMITS if plan == 'pro' else FREE_LIMITS
    col_map = {
        'translation': ('translations_today', limits['translations_day']),
        'voice':       ('voice_today',         limits['voice_day']),
        'identify':    ('identify_today',      limits['identify_day']),
    }
    if feature not in col_map:
        return True, 0, 999

    col, limit = col_map[feature]
    try:
        conn = get_db()
        reset_daily_if_needed(user_id)
        user = dict(conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone())
        used = user.get(col, 0) or 0
        if used >= limit:
            conn.close()
            return False, used, limit
        # Increment
        conn.execute(f'UPDATE users SET {col}={col}+1 WHERE id=?', (user_id,))
        conn.commit()
        conn.close()
        return True, used + 1, limit
    except:
        return True, 0, 999  # Fail open — don't block user on DB error

def limit_error_response(feature, used, limit, plan):
    """Standard limit error response"""
    NAMES = {
        'translation': 'Translations',
        'voice':       'Voice TTS',
        'identify':    'Place Identifier',
    }
    return jsonify({
        'success':     False,
        'error':       'limit_reached',
        'feature':     feature,
        'used':        used,
        'limit':       limit,
        'plan':        plan,
        'message':     f"You've used {used}/{limit} free {NAMES.get(feature, feature)} today. Upgrade to Pro for more.",
        'upgrade_url': '/pricing'
    }), 429

# ══════════════════════════════════════════════════════════
# AUTH DECORATOR
# ══════════════════════════════════════════════════════════
from functools import wraps

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()
        if not token:
            return jsonify({'success': False, 'error': 'Authentication required', 'code': 'NO_TOKEN'}), 401
        user = get_user_from_token(token)
        if not user:
            return jsonify({'success': False, 'error': 'Session expired. Please log in again.', 'code': 'EXPIRED'}), 401
        g.user    = user
        g.user_id = user['id']
        g.plan    = get_user_plan(user)
        return f(*args, **kwargs)
    return decorated

def optional_auth(f):
    """Auth is optional — sets g.user if token exists, else None"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()
        if token:
            user = get_user_from_token(token)
            if user:
                g.user    = user
                g.user_id = user['id']
                g.plan    = get_user_plan(user)
                return f(*args, **kwargs)
        g.user    = None
        g.user_id = None
        g.plan    = 'free'
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════════════════════
# LANGUAGE DATA
# ══════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════
# AUDIO HELPERS
# ══════════════════════════════════════════════════════════
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
        normalized = [max(-32768,min(32767,int(s*factor))) for s in samples]
        return struct.pack('<'+'h'*len(normalized), *normalized)
    except: return raw_bytes

def audio_to_wav(raw_bytes, sample_rate=16000):
    normalized = normalize_audio(raw_bytes)
    buf = io.BytesIO()
    with wave.open(buf,'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sample_rate); wf.writeframes(normalized)
    buf.seek(0); return buf.read()

def is_valid(text):
    if not text: return False
    t = text.strip()
    if len(t)<3 or t in HALLUCINATIONS: return False
    alpha = sum(c.isalpha() for c in t)
    return alpha >= len(t)*0.25

def safe_send(ws, data):
    try: ws.send(json.dumps(data))
    except: pass

# ══════════════════════════════════════════════════════════
# TRANSCRIPTION
# ══════════════════════════════════════════════════════════
def transcribe(wav_data, lang_hint=None):
    return transcribe_with_model(wav_data, lang_hint, WHISPER_TURBO)

def transcribe_with_model(wav_data, lang_hint=None, model=WHISPER_TURBO, prompt_override=None):
    kwargs = {
        'file': ('audio.wav', wav_data),
        'model': model,
        'response_format': 'verbose_json',
        'temperature': 0.0,
    }
    if lang_hint and lang_hint not in ('auto','unknown',None,''):
        wc = WHISPER_LANG.get(lang_hint)
        if wc:
            kwargs['language'] = wc
            prompt = prompt_override or WHISPER_PROMPTS.get(wc,'')
            if prompt: kwargs['prompt'] = prompt
    t0 = time.time()
    result = groq_client.audio.transcriptions.create(**kwargs)
    text = result.text.strip()
    detected = getattr(result,'language','unknown')
    segments = getattr(result,'segments',[])
    conf = sum(abs(s.get('avg_logprob',-1)) for s in segments)/max(len(segments),1) if segments else 0.0
    print(f"[{model.split('/')[-1]} {time.time()-t0:.2f}s] '{text[:50]}' | lang={detected}")
    return text, detected, conf

# ══════════════════════════════════════════════════════════
# TRANSLATION
# ══════════════════════════════════════════════════════════
_TRANSLATE_PREFIXES = (
    'translation:', 'translated:', 'in english:', 'in hindi:',
    'here is the translation:', 'here\'s the translation:',
    'the translation is:', 'sure,', 'sure!', 'certainly,',
)

def _strip_translation_noise(text):
    import re
    t = text.strip()
    lower = t.lower()
    for prefix in _TRANSLATE_PREFIXES:
        if lower.startswith(prefix):
            t = t[len(prefix):].lstrip(' \n')
            lower = t.lower()
    t = re.sub(r'\s*\(Note:[^)]*\)\s*$', '', t, flags=re.IGNORECASE).strip()
    t = re.sub(r'\s*\[Note:[^\]]*\]\s*$', '', t, flags=re.IGNORECASE).strip()
    return t

def translate(text, target_lang, src_lang=None):
    tgt = target_lang.lower()[:2] if len(target_lang)>=2 else target_lang
    deepl_code = DEEPL_LANGS.get(tgt) or DEEPL_LANGS.get(target_lang)
    if deepl_code and deepl_client:
        try:
            src = None
            if src_lang and src_lang not in ('unknown','auto',None,''):
                src = src_lang.upper()[:2]
                if src.lower()==tgt.lower(): src=None
            t0=time.time()
            result = deepl_client.translate_text(text, target_lang=deepl_code, source_lang=src)
            print(f"[DeepL {time.time()-t0:.2f}s]")
            return result.text,'DeepL'
        except Exception as e: print(f"[DeepL error] {e}")

    tgt_name = LANG_NAMES.get(tgt) or LANG_NAMES.get(target_lang,'English')
    src_name = LANG_NAMES.get(src_lang,'') if src_lang and src_lang not in ('unknown','auto',None,'') else ''
    src_clause = f' from {src_name}' if src_name else ''
    t0=time.time()
    r = groq_client.chat.completions.create(
        model=MODEL_70B,
        messages=[
            {'role':'system','content':(
                f'You are a professional translator. Translate text{src_clause} into {tgt_name}.\n'
                'RULES:\n'
                '• Output ONLY the translated text. No labels, no explanations.\n'
                '• Do NOT answer questions — translate them as questions.\n'
                '• Never say "Translation:", "Sure,", "Here is", or any prefix.\n'
                '• Preserve meaning, tone, register, and punctuation exactly.'
            )},
            {'role':'user','content':f'"""\n{text}\n"""'}
        ],
        temperature=0.05, max_tokens=600
    )
    raw = r.choices[0].message.content.strip()
    result = _strip_translation_noise(raw)
    print(f"[Groq translate {time.time()-t0:.2f}s]")
    return result,'Groq AI'

# ══════════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════════
def tts(text, lang_code):
    async def _run():
        voice = EDGE_VOICES.get(lang_code,'en-US-JennyNeural')
        try:
            communicate = edge_tts.Communicate(text, voice)
            buf=io.BytesIO()
            async for chunk in communicate.stream():
                if chunk['type']=='audio': buf.write(chunk['data'])
            buf.seek(0); data=buf.read()
            if len(data)>100: return data
            raise Exception("Empty audio")
        except:
            communicate = edge_tts.Communicate(text,'en-US-JennyNeural')
            buf=io.BytesIO()
            async for chunk in communicate.stream():
                if chunk['type']=='audio': buf.write(chunk['data'])
            buf.seek(0); return buf.read()
    t0=time.time(); result=asyncio.run(_run())
    print(f"[TTS {time.time()-t0:.2f}s]")
    return result

# ══════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════
@app.route('/')
def stream_page(): return render_template('stream.html')

@app.route('/convo')
def convo_page(): return render_template('convo.html')

@app.route('/camera')
def camera_page(): return render_template('camera.html')

@app.route('/translate')
def translate_page(): return render_template('stream.html')

# ══════════════════════════════════════════════════════════
# REST TRANSLATION API
# ══════════════════════════════════════════════════════════
@app.route('/api/translate/text', methods=['POST'])
@require_auth
@limiter.limit("60 per minute")
@check_feature_limit('translation')
def api_translate_text():
    """Translate text — enforces daily limit"""
    allowed, used, limit = check_and_increment(g.user_id, 'translation', g.plan)
    if not allowed:
        return limit_error_response('translation', used, limit, g.plan)
    try:
        data        = request.get_json() or {}
        text        = (data.get('text') or '').strip()
        target_lang = (data.get('target_lang') or 'en').strip()
        src_lang    = (data.get('src_lang') or None)
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        translated, engine = translate(text, target_lang, src_lang)
        return jsonify({
            'success':    True,
            'translated': translated,
            'engine':     engine,
            'used':       used,
            'limit':      limit,
            'plan':       g.plan,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/translate/tts', methods=['POST'])
@require_auth
@limiter.limit("30 per minute")
@check_feature_limit('voice')
def api_tts():
    """Text to speech — enforces daily voice limit"""
    allowed, used, limit = check_and_increment(g.user_id, 'voice', g.plan)
    if not allowed:
        return limit_error_response('voice', used, limit, g.plan)
    try:
        data      = request.get_json() or {}
        text      = (data.get('text') or '').strip()
        lang_code = (data.get('lang') or 'en').strip()
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        audio_data = tts(text, lang_code)
        return jsonify({
            'success': True,
            'audio':   base64.b64encode(audio_data).decode(),
            'used':    used,
            'limit':   limit,
            'plan':    g.plan,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ══════════════════════════════════════════════════════════
# CAMERA SCAN — with auth + identify limit
# ══════════════════════════════════════════════════════════
def _translate_blocks_batch(blocks, target_lang, src_lang=None):
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
                model=SCOUT,
                messages=[
                    {'role':'system','content':f'Translate each segment to {lang_name}. Keep [|||] separators exactly. Return ONLY translations.'},
                    {'role':'user','content':combined}
                ], temperature=0.1, max_tokens=1500
            )
            parts = r.choices[0].message.content.strip().split('[|||]')
            translated_texts = [p.strip() for p in parts]
        except Exception as e:
            print(f"[Groq batch] {e}")
            translated_texts = texts
    return [
        {
            'original':   b['text'],
            'translated': translated_texts[i].strip() if i < len(translated_texts) else b['text'],
            'vertices':   b['vertices']
        }
        for i, b in enumerate(blocks)
    ]

@app.route('/scan', methods=['POST'])
@optional_auth
@limiter.limit("20 per minute")
def scan():
    """Camera scan + translate — enforces identify limit"""
    user_id = g.user_id
    plan    = g.plan

    if user_id:
        allowed, used, limit = check_and_increment(user_id, 'identify', plan)
        if not allowed:
            return limit_error_response('identify', used, limit, plan)

    try:
        data        = request.get_json()
        image_data  = data.get('image','')
        target_lang = data.get('target_lang','EN').upper()
        if ',' in image_data: image_data = image_data.split(',')[1]
        if not image_data: return jsonify({'success':False,'error':'No image'})

        vision_result=[None]; groq_result=[None]
        vision_done=threading.Event(); groq_done=threading.Event()

        def run_vision():
            try:
                if not GOOGLE_VISION_KEY: vision_done.set(); return
                t0=time.time()
                r = req.post(
                    f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_KEY}",
                    json={"requests":[{"image":{"content":image_data},"features":[
                        {"type":"DOCUMENT_TEXT_DETECTION"}
                    ]}]},
                    timeout=8
                )
                resp0 = r.json().get('responses',[{}])[0]
                full_text = resp0.get('fullTextAnnotation',{}).get('text','')
                if not full_text:
                    anns = resp0.get('textAnnotations',[])
                    full_text = anns[0].get('description','') if anns else ''
                pages = resp0.get('fullTextAnnotation',{}).get('pages',[])
                lang='unknown'
                if pages:
                    langs=pages[0].get('property',{}).get('detectedLanguages',[])
                    if langs: lang=langs[0].get('languageCode','unknown')
                raw_blocks=[]
                for page in pages:
                    for block in page.get('blocks',[]):
                        parts=[]
                        for para in block.get('paragraphs',[]):
                            words=[]
                            for word in para.get('words',[]):
                                w=''.join(s.get('text','') for s in word.get('symbols',[]))
                                words.append(w)
                            parts.append(' '.join(words))
                        block_text='\n'.join(parts).strip()
                        if not block_text: continue
                        verts=block.get('boundingBox',{}).get('vertices',[])
                        vertices=[[v.get('x',0),v.get('y',0)] for v in verts]
                        if vertices: raw_blocks.append({'text':block_text,'vertices':vertices})
                if not raw_blocks and resp0.get('textAnnotations'):
                    for ann in resp0['textAnnotations'][1:]:
                        t=ann.get('description','').strip()
                        verts=ann.get('boundingPoly',{}).get('vertices',[])
                        vertices=[[v.get('x',0),v.get('y',0)] for v in verts]
                        if t and vertices: raw_blocks.append({'text':t,'vertices':vertices})
                print(f"[Vision {time.time()-t0:.2f}s] '{full_text[:50]}'")
                if full_text.strip(): vision_result[0]=(full_text.strip(), lang, raw_blocks)
            except Exception as e: print(f"[Vision error] {e}")
            vision_done.set()

        def run_groq_vision():
            try:
                t0=time.time()
                response = groq_client.chat.completions.create(
                    model=SCOUT,
                    messages=[{"role":"user","content":[
                        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_data}"}},
                        {"type":"text","text":"Extract ALL text in this image exactly as written. Preserve line breaks. Return ONLY the raw text. If no text, return NO_TEXT."}
                    ]}],
                    temperature=0.0, max_tokens=800
                )
                text=response.choices[0].message.content.strip()
                print(f"[Scout Vision {time.time()-t0:.2f}s] '{text[:50]}'")
                if text and text!='NO_TEXT': groq_result[0]=(text,'unknown',[])
            except Exception as e: print(f"[Scout Vision error] {e}")
            groq_done.set()

        threading.Thread(target=run_vision, daemon=True).start()
        threading.Thread(target=run_groq_vision, daemon=True).start()

        extracted_text=''; detected_lang='unknown'; raw_blocks=[]
        deadline=time.time()+8.0
        while time.time()<deadline:
            if vision_done.is_set() and vision_result[0]:
                extracted_text,detected_lang,raw_blocks=vision_result[0]; break
            if groq_done.is_set() and groq_result[0]:
                extracted_text,detected_lang,raw_blocks=groq_result[0]; break
            time.sleep(0.05)

        if not extracted_text:
            vision_done.wait(2); groq_done.wait(2)
            if vision_result[0]: extracted_text,detected_lang,raw_blocks=vision_result[0]
            elif groq_result[0]: extracted_text,detected_lang,raw_blocks=groq_result[0]

        if not extracted_text:
            return jsonify({'success':False,'error':'No text found. Try pointing at clearer text.'})

        translated_text,engine = translate(extracted_text,target_lang,detected_lang)
        text_blocks = _translate_blocks_batch(raw_blocks, target_lang, detected_lang)

        return jsonify({
            'success':         True,
            'original_text':   extracted_text,
            'translated_text': translated_text,
            'detected_lang':   detected_lang,
            'engine':          engine,
            'text_blocks':     text_blocks,
            'used':            used if user_id else 0,
            'limit':           limit if user_id else 999,
            'plan':            plan,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/retranslate-blocks', methods=['POST'])
@optional_auth
def api_retranslate_blocks():
    try:
        data = request.get_json() or {}
        blocks      = data.get('blocks',[])
        target_lang = data.get('target_lang','EN').upper()
        src_lang    = data.get('src_lang', None)
        if not blocks: return jsonify({'success':False,'error':'No blocks provided'})
        text_blocks = _translate_blocks_batch(blocks, target_lang, src_lang)
        return jsonify({'success':True,'text_blocks':text_blocks})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

# ══════════════════════════════════════════════════════════
# STREAM WEBSOCKET — with per-session limit tracking
# ══════════════════════════════════════════════════════════
def _get_user_from_ws_token(token):
    """Get user for WebSocket (token passed in first message)"""
    if not token: return None, 'free'
    user = get_user_from_token(token)
    if not user: return None, 'free'
    return user, get_user_plan(user)

def process_stream(ws, audio_bytes, target_lang, src_lang, sentence_id, user_id, plan):
    t_start=time.time()
    try:
        # Check translation limit
        if user_id:
            allowed, used, limit = check_and_increment(user_id, 'translation', plan)
            if not allowed:
                safe_send(ws, {
                    'type':    'limit_reached',
                    'feature': 'translation',
                    'used':    used,
                    'limit':   limit,
                    'plan':    plan,
                    'message': f"Daily translation limit reached ({used}/{limit}). Upgrade to Pro.",
                    'upgrade_url': '/pricing'
                })
                safe_send(ws, {'type':'ready'})
                return

        safe_send(ws,{'type':'status','message':'🎯 Listening...'})
        wav=audio_to_wav(bytes(audio_bytes))
        text,detected,conf=transcribe(wav, src_lang if src_lang!='auto' else None)

        if not is_valid(text):
            safe_send(ws,{'type':'ready'}); return

        safe_send(ws,{'type':'transcript','text':text,'lang':detected,'id':sentence_id})
        safe_send(ws,{'type':'status','message':'🌍 Translating...'})
        translated,engine=translate(text,target_lang,detected)
        safe_send(ws,{'type':'translation','text':translated,'engine':engine,'lang':target_lang,'id':sentence_id})

        # TTS limit check
        tts_allowed = True
        if user_id:
            tts_ok, _, _ = check_and_increment(user_id, 'voice', plan)
            tts_allowed = tts_ok

        if tts_allowed:
            safe_send(ws,{'type':'status','message':'🔊 Speaking...'})
            audio_data=tts(translated,target_lang)
            safe_send(ws,{'type':'audio','data':base64.b64encode(audio_data).decode(),'id':sentence_id})
        else:
            safe_send(ws,{'type':'status','message':'Voice limit reached — text only'})

        safe_send(ws,{'type':'ready'})
        print(f"[Stream #{sentence_id}] TOTAL {time.time()-t_start:.2f}s")
    except Exception as e:
        print(f"[Stream #{sentence_id}] Error: {e}")
        safe_send(ws,{'type':'error','message':str(e)})
        safe_send(ws,{'type':'ready'})

@sock.route('/stream')
def stream_ws(ws):
    print("✅ Stream connected")
    target_lang='HI'; src_lang='auto'
    audio_buffer=bytearray(); silent_chunks=0; speaking=False
    sentence_id=0; processing=False
    user_id=None; plan='free'
    SILENCE_THRESHOLD=450
    SILENCE_CHUNKS_NEEDED=2
    MIN_BYTES=int(16000*2*0.25)

    while True:
        try:
            msg=ws.receive()
            if msg is None: break
            if isinstance(msg,str):
                try:
                    cfg=json.loads(msg)
                    if 'target_lang' in cfg: target_lang=cfg['target_lang']
                    if 'src_lang'    in cfg: src_lang=cfg['src_lang']
                    # Auth via token in first config message
                    if 'token' in cfg:
                        user, plan = _get_user_from_ws_token(cfg['token'])
                        if user:
                            user_id = user['id']
                            safe_send(ws, {
                                'type':  'auth_ok',
                                'plan':  plan,
                                'name':  user.get('name',''),
                            })
                        else:
                            safe_send(ws, {'type':'auth_failed'})
                except: pass
                continue
            chunk=bytes(msg); rms=get_rms(chunk)
            safe_send(ws,{'type':'volume','level':min(100,int(rms/35))})
            if rms>=SILENCE_THRESHOLD:
                if not speaking:
                    speaking=True
                    safe_send(ws,{'type':'speaking','status':True})
                silent_chunks=0; audio_buffer.extend(chunk)
            elif speaking:
                silent_chunks+=1; audio_buffer.extend(chunk)
                if silent_chunks>=SILENCE_CHUNKS_NEEDED:
                    if len(audio_buffer)>=MIN_BYTES and not processing:
                        sentence_id+=1; processing=True
                        buf_copy=bytearray(audio_buffer)
                        t=threading.Thread(
                            target=process_stream,
                            args=(ws,buf_copy,target_lang,src_lang,sentence_id,user_id,plan),
                            daemon=True
                        )
                        t.start(); t.join(); processing=False
                    audio_buffer=bytearray(); silent_chunks=0; speaking=False
                    safe_send(ws,{'type':'speaking','status':False})
        except Exception as e:
            print(f"[Stream WS] {e}"); break
    print("❌ Stream disconnected")

# ══════════════════════════════════════════════════════════
# CONVERSATION WEBSOCKET
# ══════════════════════════════════════════════════════════
@sock.route('/convo-ws')
def convo_ws(ws):
    print("✅ Convo connected")
    lang_a='en'; lang_b='hi'; active_speaker='A'
    audio_buffer=bytearray(); silent_chunks=0; speaking=False; msg_id=0
    mode='fast'
    overlap_buf=bytearray()
    context_a=''; context_b=''
    user_id=None; plan='free'
    OVERLAP_BYTES=int(16000*2*0.3)

    MODES = {
        'fast': {
            'model':       WHISPER_TURBO,
            'threshold':   400,
            'silence_fast': 2,
            'silence_slow': 4,
            'min_bytes':   int(16000*2*0.25),
        },
        'advanced': {
            'model':       WHISPER_LARGE,
            'threshold':   450,
            'silence_fast': 3,
            'silence_slow': 5,
            'min_bytes':   int(16000*2*0.35),
        },
    }

    while True:
        try:
            msg=ws.receive()
            if msg is None: break
            if isinstance(msg,str):
                try:
                    cfg=json.loads(msg)
                    if 'lang_a'   in cfg: lang_a=cfg['lang_a'].lower().strip()[:2]
                    if 'lang_b'   in cfg: lang_b=cfg['lang_b'].lower().strip()[:2]
                    if 'mode'     in cfg:
                        mode=cfg['mode']
                        safe_send(ws,{'type':'mode_ack','mode':mode})
                    if 'speaker'  in cfg:
                        active_speaker=cfg['speaker']
                        audio_buffer=bytearray(); silent_chunks=0
                        speaking=False; overlap_buf=bytearray()
                        safe_send(ws,{'type':'speaker_changed','speaker':active_speaker})
                    if 'token'    in cfg:
                        user, plan = _get_user_from_ws_token(cfg['token'])
                        if user:
                            user_id = user['id']
                            safe_send(ws, {'type':'auth_ok','plan':plan})
                except Exception as e: print(f"[Convo config] {e}")
                continue

            mc=MODES.get(mode,MODES['fast'])
            chunk=bytes(msg); rms=get_rms(chunk)
            safe_send(ws,{'type':'volume','level':min(100,int(rms/35)),'speaker':active_speaker})

            if rms>=mc['threshold']:
                if not speaking:
                    speaking=True
                    if mode=='advanced' and overlap_buf:
                        audio_buffer.extend(overlap_buf)
                    safe_send(ws,{'type':'speaking','status':True,'speaker':active_speaker})
                silent_chunks=0; audio_buffer.extend(chunk)
            elif speaking:
                silent_chunks+=1; audio_buffer.extend(chunk)
                src=lang_a if active_speaker=='A' else lang_b
                silence_needed=mc['silence_slow'] if src in SLOW_LANGS else mc['silence_fast']
                if silent_chunks>=silence_needed:
                    if len(audio_buffer)>=mc['min_bytes']:
                        msg_id+=1
                        tgt=lang_b if active_speaker=='A' else lang_a

                        # Check translation limit
                        if user_id:
                            allowed, used, limit = check_and_increment(user_id, 'translation', plan)
                            if not allowed:
                                safe_send(ws, {
                                    'type':    'limit_reached',
                                    'feature': 'translation',
                                    'used':    used,
                                    'limit':   limit,
                                    'plan':    plan,
                                    'message': f"Daily limit reached ({used}/{limit}). Upgrade to Pro.",
                                    'upgrade_url': '/pricing'
                                })
                                safe_send(ws,{'type':'ready'})
                                audio_buffer=bytearray(); silent_chunks=0; speaking=False
                                safe_send(ws,{'type':'speaking','status':False,'speaker':active_speaker})
                                continue

                        try:
                            status_msg='⚡ Processing...' if mode=='fast' else '🔍 High-accuracy...'
                            safe_send(ws,{'type':'status','message':status_msg})
                            wav=audio_to_wav(bytes(audio_buffer))
                            whisper_hint=WHISPER_LANG.get(src)
                            if mode=='advanced':
                                ctx = context_a if active_speaker=='A' else context_b
                                base_prompt = WHISPER_PROMPTS.get(WHISPER_LANG.get(src,'en'),'')
                                prompt_ctx = (base_prompt+' '+ctx[-120:]).strip() if ctx else base_prompt or None
                                text,detected,_ = transcribe_with_model(wav, whisper_hint, mc['model'], prompt_ctx)
                            else:
                                text,detected,_ = transcribe_with_model(wav, whisper_hint, mc['model'])

                            if is_valid(text):
                                if active_speaker=='A': context_a=text
                                else: context_b=text
                                safe_send(ws,{'type':'transcript','text':text,'speaker':active_speaker,
                                              'lang':detected,'id':msg_id,'mode':mode})
                                safe_send(ws,{'type':'status','message':'🌍 Translating...'})
                                translated,engine=translate(text,tgt,src)
                                safe_send(ws,{'type':'translation','text':translated,'speaker':active_speaker,
                                              'engine':engine,'src_lang':src,'tgt_lang':tgt,'id':msg_id})
                                # TTS
                                if user_id:
                                    tts_ok, _, _ = check_and_increment(user_id, 'voice', plan)
                                else:
                                    tts_ok = True
                                if tts_ok:
                                    safe_send(ws,{'type':'status','message':'🔊 Speaking...'})
                                    audio_data=tts(translated,tgt)
                                    safe_send(ws,{'type':'audio',
                                                  'data':base64.b64encode(audio_data).decode(),
                                                  'speaker':active_speaker,'id':msg_id})
                        except Exception as e:
                            print(f"[Convo error] {e}")
                            safe_send(ws,{'type':'error','message':str(e)})
                        safe_send(ws,{'type':'ready'})
                    audio_buffer=bytearray(); silent_chunks=0; speaking=False
                    safe_send(ws,{'type':'speaking','status':False,'speaker':active_speaker})
        except Exception as e:
            print(f"[Convo WS] {e}"); break
    print("❌ Convo disconnected")

# ══════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════
@app.route('/health')
def health():
    return jsonify({
        'status':  'ok',
        'service': 'yaply-translation-engine',
        'version': '3.0',
        'deepl':   deepl_client is not None,
        'vision':  GOOGLE_VISION_KEY is not None,
    })

if __name__=='__main__':
    port=int(os.environ.get('PORT',5001))
    print(f"🚀 Yaply Translation Engine v3 — port {port}")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)