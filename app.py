from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import deepl
import edge_tts
import asyncio
import os
import tempfile
import base64
import io
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

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
    'EN': 'EN-US',
    'ES': 'ES',
    'FR': 'FR',
    'DE': 'DE',
    'JA': 'JA',
    'ZH': 'ZH',
    'AR': None,
    'HI': None,
    'PT': 'PT-BR',
    'RU': 'RU',
    'IT': 'IT',
    'KO': 'KO',
}

LANG_NAMES = {
    'EN': 'English', 'ES': 'Spanish', 'FR': 'French',
    'DE': 'German', 'JA': 'Japanese', 'ZH': 'Chinese',
    'AR': 'Arabic', 'HI': 'Hindi', 'PT': 'Portuguese',
    'RU': 'Russian', 'IT': 'Italian', 'KO': 'Korean'
}


async def tts_edge(text, lang_code):
    voice = EDGE_VOICES.get(lang_code, 'en-US-JennyNeural')
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            buf.write(chunk['data'])
    buf.seek(0)
    return buf.read()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    tmp_path = None
    try:
        audio_file = request.files['audio']
        target_language = request.form.get('target_language', 'EN').upper()

        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        file_size = os.path.getsize(tmp_path)
        print(f"Audio file size: {file_size} bytes")

        if file_size < 500:
            return jsonify({
                'success': False,
                'error': 'Recording too short. Please try again.'
            })

        with open(tmp_path, 'rb') as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(tmp_path), f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                language="en",
                temperature=0.0
            )

        original_text = transcription.text.strip()
        detected_language = getattr(transcription, 'language', 'unknown')
        print(f"Transcribed: '{original_text}' | Language: {detected_language}")

        if not original_text or len(original_text) == 0:
            return jsonify({
                'success': False,
                'error': 'No speech detected. Please try again.'
            })

        translated_text = None
        engine = "DeepL"

        deepl_code = DEEPL_LANGS.get(target_language)
        if deepl_code:
            try:
                result = deepl_client.translate_text(original_text, target_lang=deepl_code)
                translated_text = result.text
            except Exception as e:
                print(f"DeepL error: {e}")
                translated_text = None

        if not translated_text:
            engine = "Groq AI"
            lang_name = LANG_NAMES.get(target_language, 'English')
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate to {lang_name}. Return ONLY the translated text."
                    },
                    {"role": "user", "content": original_text}
                ],
                temperature=0.1,
                max_tokens=500
            )
            translated_text = response.choices[0].message.content.strip()

        print(f"Translated ({engine}): '{translated_text}'")

        audio_data = asyncio.run(tts_edge(translated_text, target_language))
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        return jsonify({
            'success': True,
            'original_text': original_text,
            'translated_text': translated_text,
            'detected_language': detected_language,
            'translation_engine': engine,
            'audio_base64': audio_base64
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'success': False, 'error': str(e)})

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)