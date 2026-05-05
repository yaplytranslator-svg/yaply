from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import deepl
import base64
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

deepl_client = deepl.Translator(os.getenv("DEEPL_API_KEY"))
GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_API_KEY")

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

def extract_text_from_image(image_base64):
    """Extract text from image using Google Vision API"""
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_KEY}"
    payload = {
        "requests": [{
            "image": {"content": image_base64},
            "features": [
                {"type": "TEXT_DETECTION", "maxResults": 1},
                {"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 1}
            ]
        }]
    }
    response = requests.post(url, json=payload)
    result = response.json()

    if 'error' in result:
        raise Exception(result['error']['message'])

    responses = result.get('responses', [{}])
    if not responses:
        return '', 'unknown'

    full_text = responses[0].get('fullTextAnnotation', {}).get('text', '')
    if not full_text:
        text_annotations = responses[0].get('textAnnotations', [])
        full_text = text_annotations[0].get('description', '') if text_annotations else ''

    # Detect language from response
    pages = responses[0].get('fullTextAnnotation', {}).get('pages', [])
    detected_lang = 'unknown'
    if pages:
        prop = pages[0].get('property', {})
        langs = prop.get('detectedLanguages', [])
        if langs:
            detected_lang = langs[0].get('languageCode', 'unknown')

    return full_text.strip(), detected_lang

def translate_text(text, target_lang):
    """Translate with DeepL, fallback to none"""
    if not text:
        return text, 'none'

    deepl_code = DEEPL_LANGS.get(target_lang)
    if deepl_code:
        try:
            result = deepl_client.translate_text(text, target_lang=deepl_code)
            return result.text, 'DeepL'
        except Exception as e:
            print(f"DeepL error: {e}")

    return text, 'none'

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/scan', methods=['POST'])
def scan():
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        target_lang = data.get('target_lang', 'EN').upper()

        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        if not image_data:
            return jsonify({'success': False, 'error': 'No image provided'})

        # Extract text
        extracted_text, detected_lang = extract_text_from_image(image_data)
        print(f"Extracted: '{extracted_text}' | Lang: {detected_lang}")

        if not extracted_text:
            return jsonify({
                'success': False,
                'error': 'No text found in image. Try pointing at clear text.'
            })

        # Translate
        translated_text, engine = translate_text(extracted_text, target_lang)
        print(f"Translated ({engine}): '{translated_text}'")

        return jsonify({
            'success': True,
            'original_text': extracted_text,
            'translated_text': translated_text,
            'detected_lang': detected_lang,
            'engine': engine
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(debug=True, host='0.0.0.0', port=port)