from flask import Flask, render_template
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
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
sock = Sock(app)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
deepl_client = deepl.Translator(os.getenv("DEEPL_API_KEY"))

EDGE_VOICES = {
    'EN': 'en-US-JennyNeural',
    'ES': 'es-ES-ElviraNeural',
    'FR': 'fr-FR-DeniseNeural',
    'DE': 'de-DE-KatjaNeural',
    'JA': 'ja-JP-NanamiNeural',
    'ZH': 'zh-CN-XiaoxiaoNeural',
    'AR': 'ar-SA-ZariyahNeural',
    'HI': 'hi-IN-SwaraNeural',
    'PT': 'pt-BR-FranciscaNeural',
    'RU': 'ru-RU-SvetlanaNeural',
    'IT': 'it-IT-ElsaNeural',
    'KO': 'ko-KR-SunHiNeural',
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
    'Um', 'um', 'Uh', 'uh', 'Ah', 'ah',
    'Subtitles by', 'Subscribe', 'subtitles',
    'www.mooji.org', 'Please subscribe',
}

@app.route('/landing')
def landing():
    return render_template('landing.html')

def get_rms(audio_bytes):
    """Calculate RMS volume of audio chunk"""
    try:
        count = len(audio_bytes) // 2
        if count == 0:
            return 0
        samples = struct.unpack('<' + 'h' * count, audio_bytes[:count*2])
        rms = (sum(s * s for s in samples) / count) ** 0.5
        return rms
    except:
        return 0

def is_silent(audio_bytes, threshold=600):
    return get_rms(audio_bytes) < threshold

def audio_to_wav(raw_bytes, sample_rate=16000):
    """Convert raw PCM int16 bytes to WAV"""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    buf.seek(0)
    return buf.read()

async def text_to_speech(text, lang_code):
    """Natural voice using Edge TTS"""
    voice = EDGE_VOICES.get(lang_code, 'en-US-JennyNeural')
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            buf.write(chunk['data'])
    buf.seek(0)
    return buf.read()

def transcribe_audio(wav_data, src_lang=None):
    """Transcribe using Groq Whisper Large V3 Turbo"""
    kwargs = {
        'file': ('audio.wav', wav_data),
        'model': 'whisper-large-v3-turbo',
        'response_format': 'verbose_json',
        'temperature': 0.0,
    }
    if src_lang and src_lang != 'auto':
        kwargs['language'] = src_lang

    result = groq_client.audio.transcriptions.create(**kwargs)
    text = result.text.strip()
    lang = getattr(result, 'language', 'unknown')

    # Get confidence if available
    segments = getattr(result, 'segments', [])
    avg_confidence = 0
    if segments:
        confidences = [abs(s.get('avg_logprob', -1)) for s in segments]
        avg_confidence = sum(confidences) / len(confidences)

    return text, lang, avg_confidence

def translate_text(text, target_lang, src_lang=None):
    """Translate with DeepL first, Groq as fallback"""
    deepl_code = DEEPL_LANGS.get(target_lang)

    if deepl_code:
        try:
            result = deepl_client.translate_text(
                text,
                target_lang=deepl_code,
                source_lang=src_lang.upper() if src_lang and src_lang != 'unknown' else None
            )
            return result.text, 'DeepL'
        except Exception as e:
            print(f"DeepL error: {e}")

    # Groq fallback
    lang_name = LANG_NAMES.get(target_lang, 'English')
    response = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {
                'role': 'system',
                'content': f'You are an expert translator. Translate to {lang_name}. Return ONLY the translated text. Be natural and accurate.'
            },
            {'role': 'user', 'content': text}
        ],
        temperature=0.1,
        max_tokens=500
    )
    return response.choices[0].message.content.strip(), 'Groq AI'

def process_sentence(ws, audio_bytes, target_lang, src_lang, sentence_id):
    """Process a complete sentence — transcribe, translate, speak"""
    try:
        # Step 1 — Transcribe
        safe_send(ws, {'type': 'status', 'message': '🎯 Transcribing...'})
        wav_data = audio_to_wav(bytes(audio_bytes))
        original_text, detected_lang, confidence = transcribe_audio(wav_data, src_lang)

        print(f"[#{sentence_id}] Transcribed: '{original_text}' | Lang: {detected_lang} | Conf: {confidence:.2f}")

        # Filter bad transcriptions
        if not original_text or original_text in HALLUCINATIONS or len(original_text.strip()) < 4:
            print(f"[#{sentence_id}] Filtered as hallucination")
            safe_send(ws, {'type': 'ready'})
            return

        # Send transcript immediately
        safe_send(ws, {
            'type': 'transcript',
            'text': original_text,
            'lang': detected_lang,
            'id': sentence_id
        })

        # Step 2 — Translate
        safe_send(ws, {'type': 'status', 'message': '🌍 Translating...'})
        translated_text, engine = translate_text(original_text, target_lang, detected_lang)
        print(f"[#{sentence_id}] Translated ({engine}): '{translated_text}'")

        # Send translation
        safe_send(ws, {
            'type': 'translation',
            'text': translated_text,
            'engine': engine,
            'lang': target_lang,
            'id': sentence_id
        })

        # Step 3 — Generate voice
        safe_send(ws, {'type': 'status', 'message': '🔊 Generating voice...'})
        audio_data = asyncio.run(text_to_speech(translated_text, target_lang))
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        safe_send(ws, {
            'type': 'audio',
            'data': audio_b64,
            'id': sentence_id
        })

        safe_send(ws, {'type': 'ready'})
        print(f"[#{sentence_id}] Done!")

    except Exception as e:
        print(f"[#{sentence_id}] Error: {e}")
        safe_send(ws, {'type': 'error', 'message': str(e)})
        safe_send(ws, {'type': 'ready'})

def safe_send(ws, data):
    """Thread-safe WebSocket send"""
    try:
        ws.send(json.dumps(data))
    except:
        pass

@app.route('/')
def index():
    return render_template('stream.html')

@sock.route('/stream')
def stream_ws(ws):
    print("✅ Client connected")

    # Session config
    target_lang = 'HI'
    src_lang = 'auto'

    # Audio buffering
    audio_buffer = bytearray()
    silent_chunks = 0
    speaking = False
    sentence_id = 0
    processing = False

    # Tuning params
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 0.3
    CHUNK_SIZE = int(SAMPLE_RATE * 2 * CHUNK_DURATION)
    SILENCE_THRESHOLD = 600
    SILENCE_CHUNKS_NEEDED = 2
    MIN_SPEECH_DURATION = 0.4
    MIN_BYTES = int(SAMPLE_RATE * 2 * MIN_SPEECH_DURATION)

    while True:
        try:
            message = ws.receive()
            if message is None:
                break

            # Config message
            if isinstance(message, str):
                try:
                    config = json.loads(message)
                    if 'target_lang' in config:
                        target_lang = config['target_lang'].upper()
                        print(f"Target lang: {target_lang}")
                    if 'src_lang' in config:
                        src_lang = config['src_lang']
                        print(f"Source lang: {src_lang}")
                except:
                    pass
                continue

            # Audio chunk
            chunk = bytes(message)
            rms = get_rms(chunk)
            chunk_silent = rms < SILENCE_THRESHOLD

            # Send volume level to UI
            safe_send(ws, {
                'type': 'volume',
                'level': min(100, int(rms / 50))
            })

            if not chunk_silent:
                if not speaking:
                    speaking = True
                    safe_send(ws, {'type': 'speaking', 'status': True})

                silent_chunks = 0
                audio_buffer.extend(chunk)

            elif speaking:
                silent_chunks += 1
                audio_buffer.extend(chunk)

                if silent_chunks >= SILENCE_CHUNKS_NEEDED:
                    # Sentence complete!
                    if len(audio_buffer) >= MIN_BYTES and not processing:
                        sentence_id += 1
                        processing = True
                        buf_copy = bytearray(audio_buffer)

                        # Process in background thread
                        t = threading.Thread(
                            target=lambda: (
                                process_sentence(ws, buf_copy, target_lang, src_lang, sentence_id),
                                setattr(threading.current_thread(), '_processing_done', True)
                            )
                        )
                        t.daemon = True
                        t.start()

                        # Wait for processing to finish
                        t.join()
                        processing = False

                    audio_buffer = bytearray()
                    silent_chunks = 0
                    speaking = False
                    safe_send(ws, {'type': 'speaking', 'status': False})

        except Exception as e:
            print(f"WebSocket error: {e}")
            break

    print("❌ Client disconnected")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)