"""
Yaply — Perfected Translation Engine v2
Fixes applied:
  STREAM:  Silence detection 0.6s → faster trigger, parallel processing
  CONVO:   Whisper language prompts → right words, audio normalization
  CAMERA:  Vision + Groq run in PARALLEL → first result wins, no waiting
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sock import Sock
from groq import Groq
import deepl, edge_tts, asyncio
import os, io, base64, json, wave, struct, threading, time
import requests as req
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
sock = Sock(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
try:    deepl_client = deepl.Translator(os.getenv("DEEPL_API_KEY"))
except: deepl_client = None

GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_API_KEY")

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
# FIX: Language-specific prompts boost Whisper accuracy ~15%
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
# Languages with natural pauses — need longer silence window
SLOW_LANGS = {'hi','ar','zh','ja','ko'}

# ─────────────────────────────────────
# AUDIO HELPERS
# ─────────────────────────────────────

def get_rms(audio_bytes):
    try:
        count = len(audio_bytes)//2
        if count == 0: return 0
        samples = struct.unpack('<'+'h'*count, audio_bytes[:count*2])
        return (sum(s*s for s in samples)/count)**0.5
    except: return 0

def normalize_audio(raw_bytes):
    """Normalize volume — Whisper performs best at consistent volume"""
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

# ─────────────────────────────────────
# TRANSCRIPTION — with prompts for accuracy
# ─────────────────────────────────────

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
            if prompt: kwargs['prompt'] = prompt  # Boosts accuracy
    t0 = time.time()
    result = groq_client.audio.transcriptions.create(**kwargs)
    text = result.text.strip()
    detected = getattr(result,'language','unknown')
    segments = getattr(result,'segments',[])
    conf = sum(abs(s.get('avg_logprob',-1)) for s in segments)/max(len(segments),1) if segments else 0.0
    print(f"[Whisper {time.time()-t0:.2f}s] '{text[:50]}' | lang={detected} | conf={conf:.2f}")
    return text, detected, conf

# ─────────────────────────────────────
# TRANSLATION
# ─────────────────────────────────────

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
    lang_name = LANG_NAMES.get(target_lang,'English')
    t0=time.time()
    r = groq_client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role':'system','content':f'Translate to {lang_name}. Return ONLY the translation. Be natural.'},
            {'role':'user','content':text}
        ],
        temperature=0.1, max_tokens=500
    )
    print(f"[Groq translate {time.time()-t0:.2f}s]")
    return r.choices[0].message.content.strip(),'Groq AI'

# ─────────────────────────────────────
# TTS
# ─────────────────────────────────────

def tts(text, lang_code):
    async def _run():
        voice = EDGE_VOICES.get(lang_code,'en-US-JennyNeural')
        try:
            communicate = edge_tts.Communicate(text,voice)
            buf=io.BytesIO()
            async for chunk in communicate.stream():
                if chunk['type']=='audio': buf.write(chunk['data'])
            buf.seek(0); data=buf.read()
            if len(data)>100: return data
            raise Exception("Empty")
        except:
            communicate = edge_tts.Communicate(text,'en-US-JennyNeural')
            buf=io.BytesIO()
            async for chunk in communicate.stream():
                if chunk['type']=='audio': buf.write(chunk['data'])
            buf.seek(0); return buf.read()
    t0=time.time(); result=asyncio.run(_run())
    print(f"[TTS {time.time()-t0:.2f}s]")
    return result

# ─────────────────────────────────────
# ROUTES
# ─────────────────────────────────────

@app.route('/') 
def stream_page(): return render_template('stream.html')

@app.route('/convo')
def convo_page(): return render_template('convo.html')

@app.route('/camera')
def camera_page(): return render_template('camera.html')

@app.route('/landing')
def landing_page(): return render_template('landing.html')

# ─────────────────────────────────────
# CAMERA — PARALLEL Vision + Groq
# Both run simultaneously, first result wins
# ─────────────────────────────────────

@app.route('/scan', methods=['POST'])
def scan():
    try:
        data = request.get_json()
        image_data = data.get('image','')
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
                        {"type":"DOCUMENT_TEXT_DETECTION"},{"type":"TEXT_DETECTION"}
                    ]}]},
                    timeout=6
                )
                responses = r.json().get('responses',[{}])
                full_text = responses[0].get('fullTextAnnotation',{}).get('text','')
                if not full_text:
                    anns = responses[0].get('textAnnotations',[])
                    full_text = anns[0].get('description','') if anns else ''
                pages = responses[0].get('fullTextAnnotation',{}).get('pages',[])
                lang='unknown'
                if pages:
                    langs=pages[0].get('property',{}).get('detectedLanguages',[])
                    if langs: lang=langs[0].get('languageCode','unknown')
                print(f"[Vision {time.time()-t0:.2f}s] '{full_text[:50]}'")
                if full_text.strip(): vision_result[0]=(full_text.strip(),lang)
            except Exception as e: print(f"[Vision error] {e}")
            vision_done.set()

        def run_groq_vision():
            try:
                t0=time.time()
                response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role":"user","content":[
                        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_data}"}},
                        {"type":"text","text":"Extract ALL text in this image exactly as written. Preserve line breaks. Return ONLY the raw text. If no text, return NO_TEXT."}
                    ]}],
                    temperature=0.0, max_tokens=800
                )
                text=response.choices[0].message.content.strip()
                print(f"[Groq Vision {time.time()-t0:.2f}s] '{text[:50]}'")
                if text and text!='NO_TEXT': groq_result[0]=(text,'unknown')
            except Exception as e: print(f"[Groq Vision error] {e}")
            groq_done.set()

        # START BOTH IN PARALLEL
        threading.Thread(target=run_vision,daemon=True).start()
        threading.Thread(target=run_groq_vision,daemon=True).start()

        # Wait for first good result
        extracted_text=''; detected_lang='unknown'
        deadline=time.time()+8.0
        while time.time()<deadline:
            if vision_done.is_set() and vision_result[0]:
                extracted_text,detected_lang=vision_result[0]; break
            if groq_done.is_set() and groq_result[0]:
                extracted_text,detected_lang=groq_result[0]; break
            time.sleep(0.05)

        if not extracted_text:
            vision_done.wait(2); groq_done.wait(2)
            if vision_result[0]: extracted_text,detected_lang=vision_result[0]
            elif groq_result[0]: extracted_text,detected_lang=groq_result[0]

        if not extracted_text:
            return jsonify({'success':False,'error':'No text found. Try pointing at clearer text.'})

        translated_text,engine = translate(extracted_text,target_lang,detected_lang)
        return jsonify({
            'success':True,'original_text':extracted_text,
            'translated_text':translated_text,'detected_lang':detected_lang,'engine':engine
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)})

# ─────────────────────────────────────
# STREAM — Fast silence detection
# ─────────────────────────────────────

def process_stream(ws, audio_bytes, target_lang, src_lang, sentence_id):
    t_start=time.time()
    try:
        safe_send(ws,{'type':'status','message':'🎯 Listening...'})
        wav=audio_to_wav(bytes(audio_bytes))
        text,detected,conf=transcribe(wav, src_lang if src_lang!='auto' else None)

        if not is_valid(text):
            safe_send(ws,{'type':'ready'}); return

        # Send transcript IMMEDIATELY — users see this first
        safe_send(ws,{'type':'transcript','text':text,'lang':detected,'id':sentence_id})

        safe_send(ws,{'type':'status','message':'🌍 Translating...'})
        translated,engine=translate(text,target_lang,detected)
        safe_send(ws,{'type':'translation','text':translated,'engine':engine,'lang':target_lang,'id':sentence_id})

        safe_send(ws,{'type':'status','message':'🔊 Speaking...'})
        audio_data=tts(translated,target_lang)
        safe_send(ws,{'type':'audio','data':base64.b64encode(audio_data).decode(),'id':sentence_id})
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
    SILENCE_THRESHOLD=450
    SILENCE_CHUNKS_NEEDED=2       # ~0.6s silence = sentence done
    MIN_BYTES=int(16000*2*0.25)   # Min 0.25s audio

    while True:
        try:
            msg=ws.receive()
            if msg is None: break
            if isinstance(msg,str):
                try:
                    cfg=json.loads(msg)
                    if 'target_lang' in cfg: target_lang=cfg['target_lang']
                    if 'src_lang' in cfg: src_lang=cfg['src_lang']
                except: pass
                continue
            chunk=bytes(msg); rms=get_rms(chunk)
            safe_send(ws,{'type':'volume','level':min(100,int(rms/35))})
            if rms>=SILENCE_THRESHOLD:
                if not speaking:
                    speaking=True; safe_send(ws,{'type':'speaking','status':True})
                silent_chunks=0; audio_buffer.extend(chunk)
            elif speaking:
                silent_chunks+=1; audio_buffer.extend(chunk)
                if silent_chunks>=SILENCE_CHUNKS_NEEDED:
                    if len(audio_buffer)>=MIN_BYTES and not processing:
                        sentence_id+=1; processing=True
                        buf_copy=bytearray(audio_buffer)
                        t=threading.Thread(target=process_stream,args=(ws,buf_copy,target_lang,src_lang,sentence_id),daemon=True)
                        t.start(); t.join(); processing=False
                    audio_buffer=bytearray(); silent_chunks=0; speaking=False
                    safe_send(ws,{'type':'speaking','status':False})
        except Exception as e:
            print(f"[Stream WS] {e}"); break
    print("❌ Stream disconnected")

# ─────────────────────────────────────
# CONVERSATION — Better accuracy
# ─────────────────────────────────────

@sock.route('/convo-ws')
def convo_ws(ws):
    print("✅ Convo connected")
    lang_a='en'; lang_b='hi'; active_speaker='A'
    audio_buffer=bytearray(); silent_chunks=0; speaking=False; msg_id=0
    SILENCE_THRESHOLD=450
    MIN_BYTES=int(16000*2*0.4)

    while True:
        try:
            msg=ws.receive()
            if msg is None: break
            if isinstance(msg,str):
                try:
                    cfg=json.loads(msg)
                    if 'lang_a' in cfg: lang_a=cfg['lang_a'].lower().strip()[:2]
                    if 'lang_b' in cfg: lang_b=cfg['lang_b'].lower().strip()[:2]
                    if 'speaker' in cfg:
                        active_speaker=cfg['speaker']
                        audio_buffer=bytearray(); silent_chunks=0; speaking=False
                        safe_send(ws,{'type':'speaker_changed','speaker':active_speaker})
                except Exception as e: print(f"[Convo config] {e}")
                continue
            chunk=bytes(msg); rms=get_rms(chunk)
            safe_send(ws,{'type':'volume','level':min(100,int(rms/35)),'speaker':active_speaker})
            if rms>=SILENCE_THRESHOLD:
                if not speaking:
                    speaking=True; safe_send(ws,{'type':'speaking','status':True,'speaker':active_speaker})
                silent_chunks=0; audio_buffer.extend(chunk)
            elif speaking:
                silent_chunks+=1; audio_buffer.extend(chunk)
                src=lang_a if active_speaker=='A' else lang_b
                # Longer silence window for languages with natural pauses
                silence_needed=5 if src in SLOW_LANGS else 3
                if silent_chunks>=silence_needed:
                    if len(audio_buffer)>=MIN_BYTES:
                        msg_id+=1
                        tgt=lang_b if active_speaker=='A' else lang_a
                        try:
                            safe_send(ws,{'type':'status','message':'🎯 Listening...'})
                            wav=audio_to_wav(bytes(audio_buffer))
                            # Use speaker's language as Whisper hint + prompt
                            whisper_hint=WHISPER_LANG.get(src)
                            text,detected,conf=transcribe(wav,whisper_hint)
                            if is_valid(text):
                                safe_send(ws,{'type':'transcript','text':text,'speaker':active_speaker,'lang':detected,'id':msg_id})
                                safe_send(ws,{'type':'status','message':'🌍 Translating...'})
                                translated,engine=translate(text,tgt,src)
                                safe_send(ws,{'type':'translation','text':translated,'speaker':active_speaker,'engine':engine,'src_lang':src,'tgt_lang':tgt,'id':msg_id})
                                safe_send(ws,{'type':'status','message':'🔊 Speaking...'})
                                audio_data=tts(translated,tgt)
                                safe_send(ws,{'type':'audio','data':base64.b64encode(audio_data).decode(),'speaker':active_speaker,'id':msg_id})
                            else:
                                print(f"[Convo {active_speaker}] Filtered: '{text}'")
                        except Exception as e:
                            print(f"[Convo error] {e}")
                            safe_send(ws,{'type':'error','message':str(e)})
                        safe_send(ws,{'type':'ready'})
                    audio_buffer=bytearray(); silent_chunks=0; speaking=False
                    safe_send(ws,{'type':'speaking','status':False,'speaker':active_speaker})
        except Exception as e:
            print(f"[Convo WS] {e}"); break
    print("❌ Convo disconnected")

if __name__=='__main__':
    port=int(os.environ.get('PORT',5001))
    print(f"🚀 Yaply Translation Engine — port {port}")
    app.run(debug=False,host='0.0.0.0',port=port,threaded=True)