from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
from flask_sock import Sock
from groq import Groq
import deepl
import edge_tts
import asyncio
import os
import io
import base64
import json
import wave
import struct
import threading
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
sock = Sock(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

# ── CLIENTS ──
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
deepl_client = deepl.Translator(os.getenv("DEEPL_API_KEY"))
GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_API_KEY")

# ── CONSTANTS ──
EDGE_VOICES = {
    'EN': 'en-US-JennyNeural', 'ES': 'es-ES-ElviraNeural',
    'FR': 'fr-FR-DeniseNeural', 'DE': 'de-DE-KatjaNeural',
    'JA': 'ja-JP-NanamiNeural', 'ZH': 'zh-CN-XiaoxiaoNeural',
    'AR': 'ar-SA-ZariyahNeural', 'HI': 'hi-IN-SwaraNeural',
    'PT': 'pt-BR-FranciscaNeural', 'RU': 'ru-RU-SvetlanaNeural',
    'IT': 'it-IT-ElsaNeural', 'KO': 'ko-KR-SunHiNeural',
}

DEEPL_LANGS = {
    'EN': 'EN-US', 'ES': 'ES', 'FR': 'FR', 'DE': 'DE',
    'JA': 'JA', 'ZH': 'ZH', 'PT': 'PT-BR', 'RU': 'RU',
    'IT': 'IT', 'KO': 'KO', 'AR': None, 'HI': None,
}

LANG_NAMES = {
    'EN': 'English', 'ES': 'Spanish', 'FR': 'French',
    'DE': 'German', 'JA': 'Japanese', 'ZH': 'Chinese',
    'AR': 'Arabic', 'HI': 'Hindi', 'PT': 'Portuguese',
    'RU': 'Russian', 'IT': 'Italian', 'KO': 'Korean'
}

HALLUCINATIONS = {
    'you', 'You', 'the', 'The', 'Thank you.', 'Thank you',
    'Thanks', 'Thanks.', 'Bye', 'bye', 'Okay', 'okay',
    'OK', 'ok', '', ' ', '.', '...', 'Hmm', 'hmm',
    'Um', 'um', 'Uh', 'uh', 'Subtitles by', 'Subscribe',
}

# ── HELPERS ──
def get_rms(audio_bytes):
    try:
        count = len(audio_bytes) // 2
        if count == 0: return 0
        samples = struct.unpack('<' + 'h' * count, audio_bytes[:count*2])
        return (sum(s*s for s in samples) / count) ** 0.5
    except:
        return 0

def audio_to_wav(raw_bytes, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    buf.seek(0)
    return buf.read()

async def text_to_speech(text, lang_code):
    voice = EDGE_VOICES.get(lang_code.upper(), 'en-US-JennyNeural')
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            buf.write(chunk['data'])
    buf.seek(0)
    return buf.read()

def transcribe_audio(wav_data, src_lang=None):
    kwargs = {
        'file': ('audio.wav', wav_data),
        'model': 'whisper-large-v3-turbo',
        'response_format': 'verbose_json',
        'temperature': 0.0,
    }
    if src_lang and src_lang not in ('auto', 'unknown'):
        kwargs['language'] = src_lang.lower()
    result = groq_client.audio.transcriptions.create(**kwargs)
    return result.text.strip(), getattr(result, 'language', 'unknown')

def translate_text(text, target_lang, src_lang=None):
    target_lang = target_lang.upper()
    deepl_code = DEEPL_LANGS.get(target_lang)
    if deepl_code:
        try:
            result = deepl_client.translate_text(text, target_lang=deepl_code)
            return result.text, 'DeepL'
        except Exception as e:
            print(f"DeepL error: {e}")
    lang_name = LANG_NAMES.get(target_lang, 'English')
    response = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {'role': 'system', 'content': f'Translate to {lang_name}. Return ONLY the translation.'},
            {'role': 'user', 'content': text}
        ],
        temperature=0.1, max_tokens=500
    )
    return response.choices[0].message.content.strip(), 'Groq AI'

def safe_send(ws, data):
    try:
        ws.send(json.dumps(data))
    except:
        pass

def process_audio_chunk(ws, audio_bytes, target_lang, src_lang, sentence_id):
    try:
        safe_send(ws, {'type': 'status', 'message': '🎯 Transcribing...'})
        wav_data = audio_to_wav(bytes(audio_bytes))
        original_text, detected_lang = transcribe_audio(wav_data, src_lang)
        print(f"[#{sentence_id}] '{original_text}' ({detected_lang})")

        if not original_text or original_text in HALLUCINATIONS or len(original_text.strip()) < 4:
            safe_send(ws, {'type': 'ready'})
            return

        safe_send(ws, {
            'type': 'transcript',
            'text': original_text,
            'lang': detected_lang,
            'id': sentence_id
        })

        safe_send(ws, {'type': 'status', 'message': '🌍 Translating...'})
        translated_text, engine = translate_text(original_text, target_lang, detected_lang)
        print(f"[#{sentence_id}] → '{translated_text}' ({engine})")

        safe_send(ws, {
            'type': 'translation',
            'text': translated_text,
            'engine': engine,
            'lang': target_lang,
            'id': sentence_id
        })

        safe_send(ws, {'type': 'status', 'message': '🔊 Speaking...'})
        audio_data = asyncio.run(text_to_speech(translated_text, target_lang))
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        safe_send(ws, {'type': 'audio', 'data': audio_b64, 'id': sentence_id})
        safe_send(ws, {'type': 'ready'})

    except Exception as e:
        print(f"Error: {e}")
        safe_send(ws, {'type': 'error', 'message': str(e)})
        safe_send(ws, {'type': 'ready'})

# ── ROUTES ──
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def main_app():
    return render_template('stream.html')

@app.route('/convo')
def convo_page():
    return render_template('convo.html')

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/tools')
def tools_page():
    return render_template('tools.html')

# ── CAMERA SCAN API ──
@app.route('/scan', methods=['POST'])
def scan():
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        target_lang = data.get('target_lang', 'EN').upper()

        if ',' in image_data:
            image_data = image_data.split(',')[1]

        if not image_data:
            return jsonify({'success': False, 'error': 'No image provided'})

        # Google Vision OCR
        url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_KEY}"
        payload = {
            "requests": [{
                "image": {"content": image_data},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
            }]
        }
        response = requests.post(url, json=payload)
        result = response.json()

        responses = result.get('responses', [{}])
        full_text = responses[0].get('fullTextAnnotation', {}).get('text', '').strip()

        if not full_text:
            annotations = responses[0].get('textAnnotations', [])
            full_text = annotations[0].get('description', '').strip() if annotations else ''

        if not full_text:
            return jsonify({'success': False, 'error': 'No text found in image. Try pointing at clearer text.'})

        translated_text, engine = translate_text(full_text, target_lang)

        return jsonify({
            'success': True,
            'original_text': full_text,
            'translated_text': translated_text,
            'engine': engine
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ── VOICE STREAMING WEBSOCKET ──
@sock.route('/stream')
def stream_ws(ws):
    print("✅ Stream connected")
    target_lang = 'HI'
    src_lang = 'auto'
    audio_buffer = bytearray()
    silent_chunks = 0
    speaking = False
    sentence_id = 0
    processing = False
    SILENCE_THRESHOLD = 600
    SILENCE_CHUNKS_NEEDED = 2
    MIN_BYTES = int(16000 * 2 * 0.4)

    while True:
        try:
            message = ws.receive()
            if message is None:
                break

            if isinstance(message, str):
                try:
                    config = json.loads(message)
                    if 'target_lang' in config:
                        target_lang = config['target_lang'].upper()
                    if 'src_lang' in config:
                        src_lang = config['src_lang']
                except:
                    pass
                continue

            chunk = bytes(message)
            rms = get_rms(chunk)
            safe_send(ws, {'type': 'volume', 'level': min(100, int(rms/50))})

            if rms >= SILENCE_THRESHOLD:
                if not speaking:
                    speaking = True
                    safe_send(ws, {'type': 'speaking', 'status': True})
                silent_chunks = 0
                audio_buffer.extend(chunk)
            elif speaking:
                silent_chunks += 1
                audio_buffer.extend(chunk)
                if silent_chunks >= SILENCE_CHUNKS_NEEDED:
                    if len(audio_buffer) >= MIN_BYTES and not processing:
                        sentence_id += 1
                        processing = True
                        buf_copy = bytearray(audio_buffer)
                        t = threading.Thread(
                            target=process_audio_chunk,
                            args=(ws, buf_copy, target_lang, src_lang, sentence_id)
                        )
                        t.daemon = True
                        t.start()
                        t.join()
                        processing = False
                    audio_buffer = bytearray()
                    silent_chunks = 0
                    speaking = False
                    safe_send(ws, {'type': 'speaking', 'status': False})

        except Exception as e:
            print(f"Stream WS error: {e}")
            break

    print("❌ Stream disconnected")

# ── CONVERSATION WEBSOCKET ──
@sock.route('/convo-ws')
def convo_ws(ws):
    print("✅ Convo connected")
    lang_a = 'en'
    lang_b = 'hi'
    active_speaker = 'A'
    audio_buffer = bytearray()
    silent_chunks = 0
    speaking = False
    msg_id = 0
    SILENCE_THRESHOLD = 600
    SILENCE_CHUNKS_NEEDED = 2
    MIN_BYTES = int(16000 * 2 * 0.4)

    while True:
        try:
            message = ws.receive()
            if message is None:
                break

            if isinstance(message, str):
                try:
                    config = json.loads(message)
                    if 'lang_a' in config:
                        lang_a = config['lang_a'].lower()
                    if 'lang_b' in config:
                        lang_b = config['lang_b'].lower()
                    if 'speaker' in config:
                        active_speaker = config['speaker']
                        audio_buffer = bytearray()
                        silent_chunks = 0
                        speaking = False
                        safe_send(ws, {'type': 'speaker_changed', 'speaker': active_speaker})
                except:
                    pass
                continue

            chunk = bytes(message)
            rms = get_rms(chunk)
            safe_send(ws, {'type': 'volume', 'level': min(100, int(rms/50))})

            if rms >= SILENCE_THRESHOLD:
                if not speaking:
                    speaking = True
                    safe_send(ws, {'type': 'speaking', 'status': True, 'speaker': active_speaker})
                silent_chunks = 0
                audio_buffer.extend(chunk)
            elif speaking:
                silent_chunks += 1
                audio_buffer.extend(chunk)
                if silent_chunks >= SILENCE_CHUNKS_NEEDED:
                    if len(audio_buffer) >= MIN_BYTES:
                        msg_id += 1
                        try:
                            safe_send(ws, {'type': 'status', 'message': '🎯 Transcribing...'})
                            wav = audio_to_wav(bytes(audio_buffer))
                            src = lang_a if active_speaker == 'A' else lang_b
                            tgt = lang_b if active_speaker == 'A' else lang_a
                            text, detected = transcribe_audio(wav, src)
                            print(f"[{active_speaker}] '{text}'")

                            if text and text not in HALLUCINATIONS and len(text) > 3:
                                safe_send(ws, {
                                    'type': 'transcript',
                                    'text': text,
                                    'speaker': active_speaker,
                                    'id': msg_id
                                })
                                safe_send(ws, {'type': 'status', 'message': '🌍 Translating...'})
                                translated, engine = translate_text(text, tgt.upper(), src)
                                print(f"→ '{translated}'")

                                safe_send(ws, {
                                    'type': 'translation',
                                    'text': translated,
                                    'speaker': active_speaker,
                                    'engine': engine,
                                    'src_lang': src,
                                    'tgt_lang': tgt,
                                    'id': msg_id
                                })

                                safe_send(ws, {'type': 'status', 'message': '🔊 Speaking...'})
                                audio_data = asyncio.run(text_to_speech(translated, tgt.upper()))
                                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                                safe_send(ws, {'type': 'audio', 'data': audio_b64, 'id': msg_id})

                        except Exception as e:
                            print(f"Convo error: {e}")
                            safe_send(ws, {'type': 'error', 'message': str(e)})

                        safe_send(ws, {'type': 'ready'})

                    audio_buffer = bytearray()
                    silent_chunks = 0
                    speaking = False
                    safe_send(ws, {'type': 'speaking', 'status': False, 'speaker': active_speaker})

        except Exception as e:
            print(f"Convo WS error: {e}")
            break

    print("❌ Convo disconnected")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)