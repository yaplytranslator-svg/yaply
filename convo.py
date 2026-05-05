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
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
sock = Sock(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
deepl_client = deepl.Translator(os.getenv("DEEPL_API_KEY"))

EDGE_VOICES = {
    'en': 'en-US-JennyNeural',
    'es': 'es-ES-ElviraNeural',
    'fr': 'fr-FR-DeniseNeural',
    'de': 'de-DE-KatjaNeural',
    'ja': 'ja-JP-NanamiNeural',
    'zh': 'zh-CN-XiaoxiaoNeural',
    'ar': 'ar-SA-ZariyahNeural',
    'hi': 'hi-IN-SwaraNeural',
    'pt': 'pt-BR-FranciscaNeural',
    'ru': 'ru-RU-SvetlanaNeural',
    'it': 'it-IT-ElsaNeural',
    'ko': 'ko-KR-SunHiNeural',
}

DEEPL_LANGS = {
    'en': 'EN-US', 'es': 'ES', 'fr': 'FR', 'de': 'DE',
    'ja': 'JA', 'zh': 'ZH', 'pt': 'PT-BR', 'ru': 'RU',
    'it': 'IT', 'ko': 'KO', 'ar': None, 'hi': None,
}

LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'ja': 'Japanese', 'zh': 'Chinese',
    'ar': 'Arabic', 'hi': 'Hindi', 'pt': 'Portuguese',
    'ru': 'Russian', 'it': 'Italian', 'ko': 'Korean'
}

HALLUCINATIONS = {
    'you', 'You', 'the', 'The', 'Thank you.', 'Thank you',
    'Thanks', 'Bye', 'bye', 'Okay', 'okay', '', ' ', '.', '...'
}

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

async def speak(text, lang):
    voice = EDGE_VOICES.get(lang, 'en-US-JennyNeural')
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            buf.write(chunk['data'])
    buf.seek(0)
    return buf.read()

def transcribe(wav_data, lang=None):
    kwargs = {
        'file': ('audio.wav', wav_data),
        'model': 'whisper-large-v3-turbo',
        'response_format': 'verbose_json',
        'temperature': 0.0,
    }
    if lang and lang != 'auto':
        kwargs['language'] = lang
    result = groq_client.audio.transcriptions.create(**kwargs)
    return result.text.strip(), getattr(result, 'language', 'unknown')

def translate(text, from_lang, to_lang):
    if from_lang == to_lang:
        return text, 'none'
    deepl_code = DEEPL_LANGS.get(to_lang)
    if deepl_code:
        try:
            result = deepl_client.translate_text(text, target_lang=deepl_code)
            return result.text, 'DeepL'
        except Exception as e:
            print(f"DeepL error: {e}")
    lang_name = LANG_NAMES.get(to_lang, 'English')
    response = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {'role': 'system', 'content': f'Translate to {lang_name}. Return ONLY translation.'},
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

@app.route('/')
def index():
    return render_template('convo.html')

@sock.route('/convo')
def convo_ws(ws):
    print("✅ Conversation started")
    
    # Two speakers config
    speaker_a_lang = 'en'
    speaker_b_lang = 'hi'
    active_speaker = 'A'

    audio_buffer = bytearray()
    silent_chunks = 0
    speaking = False
    msg_id = 0

    SILENCE_THRESHOLD = 600
    SILENCE_CHUNKS_NEEDED = 2
    MIN_BYTES = int(16000 * 2 * 0.5)

    while True:
        try:
            message = ws.receive()
            if message is None:
                break

            # Config
            if isinstance(message, str):
                try:
                    config = json.loads(message)
                    if 'lang_a' in config:
                        speaker_a_lang = config['lang_a']
                    if 'lang_b' in config:
                        speaker_b_lang = config['lang_b']
                    if 'speaker' in config:
                        active_speaker = config['speaker']
                        audio_buffer = bytearray()
                        silent_chunks = 0
                        speaking = False
                        safe_send(ws, {
                            'type': 'speaker_changed',
                            'speaker': active_speaker
                        })
                except:
                    pass
                continue

            # Audio
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

                            # Who is speaking?
                            src_lang = speaker_a_lang if active_speaker == 'A' else speaker_b_lang
                            tgt_lang = speaker_b_lang if active_speaker == 'A' else speaker_a_lang

                            text, detected = transcribe(wav, src_lang)
                            print(f"[{active_speaker}] '{text}' ({detected})")

                            if text and text not in HALLUCINATIONS and len(text) > 3:
                                safe_send(ws, {
                                    'type': 'transcript',
                                    'text': text,
                                    'speaker': active_speaker,
                                    'lang': detected,
                                    'id': msg_id
                                })

                                safe_send(ws, {'type': 'status', 'message': '🌍 Translating...'})
                                translated, engine = translate(text, src_lang, tgt_lang)
                                print(f"→ '{translated}' ({engine})")

                                safe_send(ws, {
                                    'type': 'translation',
                                    'text': translated,
                                    'speaker': active_speaker,
                                    'engine': engine,
                                    'src_lang': src_lang,
                                    'tgt_lang': tgt_lang,
                                    'id': msg_id
                                })

                                safe_send(ws, {'type': 'status', 'message': '🔊 Speaking...'})
                                audio_data = asyncio.run(speak(translated, tgt_lang))
                                audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                                safe_send(ws, {
                                    'type': 'audio',
                                    'data': audio_b64,
                                    'speaker': active_speaker,
                                    'id': msg_id
                                })

                        except Exception as e:
                            print(f"Error: {e}")
                            safe_send(ws, {'type': 'error', 'message': str(e)})

                        safe_send(ws, {'type': 'ready'})

                    audio_buffer = bytearray()
                    silent_chunks = 0
                    speaking = False
                    safe_send(ws, {'type': 'speaking', 'status': False, 'speaker': active_speaker})

        except Exception as e:
            print(f"WS error: {e}")
            break

    print("❌ Conversation ended")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)