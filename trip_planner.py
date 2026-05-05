from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
WEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
EXCHANGE_KEY = os.getenv("EXCHANGE_API_KEY")
GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_API_KEY")

# ── AI TRIP PLANNER ──
@app.route('/plan', methods=['POST'])
def plan_trip():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        origin = data.get('origin', 'India')
        days = data.get('days', 5)
        budget = data.get('budget', '50000')
        vibe = data.get('vibe', 'adventure')
        people = data.get('people', 1)
        currency = data.get('currency', 'INR')

        prompt = f"""You are a world-class travel planner with expertise in every destination globally.

Create a UNIQUE, HIGHLY DETAILED {days}-day trip plan for:
- FROM: {origin}
- TO: {destination}  
- Duration: {days} days
- Budget: {currency} {budget} total for {people} people
- Travel style: {vibe} — Make EVERY activity match this style specifically
- Travellers: {people}

CRITICAL RULES:
1. ALL prices must be in {currency} — not any other currency
2. Budget breakdown must add up to approximately {currency} {budget}
3. Every single activity, restaurant, hotel price must be in {currency}
4. Tailor activities 100% to the {vibe} travel style
5. Include HIDDEN GEMS — not just tourist spots
6. Include LOCAL TRANSPORT options specific to {destination}
7. Include internet/SIM card costs in {currency}
8. Include vaccination requirements for travellers from {origin}
9. Include cultural dos and donts
10. Include must have apps for {destination}

Make the itinerary 100% unique based on the travel style. Different vibes = completely different activities, restaurants and experiences.

Return ONLY a valid JSON object with this exact structure:
{{
  "destination": "{destination}",
  "days": {days},
  "budget_breakdown": {{
    "accommodation": "amount",
    "food": "amount", 
    "transport": "amount",
    "activities": "amount",
    "miscellaneous": "amount"
  }},
  "best_time_to_visit": "month range",
  "language": "local language",
  "currency": "local currency",
  "timezone": "timezone",
  "itinerary": [
    {{
      "day": 1,
      "title": "Day title",
      "morning": {{"activity": "name", "location": "place", "duration": "2 hours", "cost": "amount", "tip": "insider tip"}},
      "afternoon": {{"activity": "name", "location": "place", "duration": "2 hours", "cost": "amount", "tip": "insider tip"}},
      "evening": {{"activity": "name", "location": "place", "duration": "2 hours", "cost": "amount", "tip": "insider tip"}},
      "lunch": {{"restaurant": "name", "cuisine": "type", "cost": "amount"}},
      "dinner": {{"restaurant": "name", "cuisine": "type", "cost": "amount"}},
      "accommodation": {{"name": "hotel name", "area": "neighborhood", "cost": "per night"}}
    }}
  ],
  "packing_list": ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item10"],
  "local_phrases": [
    {{"phrase": "Hello", "translation": "local word", "pronunciation": "how to say"}},
    {{"phrase": "Thank you", "translation": "local word", "pronunciation": "how to say"}},
    {{"phrase": "Where is...?", "translation": "local word", "pronunciation": "how to say"}},
    {{"phrase": "How much?", "translation": "local word", "pronunciation": "how to say"}},
    {{"phrase": "Help!", "translation": "local word", "pronunciation": "how to say"}}
  ],
  "visa_info": {{
    "required": true,
    "type": "visa type for travellers from {origin}",
    "duration": "validity",
    "cost": "fee",
    "processing_time": "days",
    "apply_at": "where to apply"
  }},
  "emergency_numbers": {{
    "police": "number",
    "ambulance": "number",
    "fire": "number",
    "tourist_helpline": "number"
  }},
 "flight_info": {{
    "estimated_cost": "return flight cost in {currency}",
    "best_airlines": ["airline1", "airline2"],
    "flight_duration": "hours",
    "best_time_to_book": "how many days in advance"
}},
"hidden_gems": [
    {{"name": "place name", "description": "why its special", "location": "area", "best_time": "when to visit", "cost": "price in {currency}"}},
    {{"name": "place name", "description": "why its special", "location": "area", "best_time": "when to visit", "cost": "price in {currency}"}},
    {{"name": "place name", "description": "why its special", "location": "area", "best_time": "when to visit", "cost": "price in {currency}"}}
],
"local_transport": {{
    "airport_to_city": {{"options": ["option1", "option2"], "cost": "{currency} amount", "duration": "time"}},
    "within_city": [
        {{"type": "Metro/Subway", "cost": "{currency} per ride", "tip": "how to use"}},
        {{"type": "Bus", "cost": "{currency} per ride", "tip": "how to use"}},
        {{"type": "Taxi/Rideshare", "cost": "{currency} per km", "tip": "recommended apps"}}
    ],
    "useful_apps": ["app1", "app2", "app3"]
}},
"sim_internet": {{
    "best_option": "recommendation",
    "cost": "{currency} amount",
    "data": "GB amount",
    "where_to_buy": "location"
}},
"vaccinations": {{
    "required": ["vaccine1", "vaccine2"],
    "recommended": ["vaccine1", "vaccine2"],
    "note": "health advisory"
}},
"cultural_guide": {{
    "dos": ["do1", "do2", "do3", "do4", "do5"],
    "donts": ["dont1", "dont2", "dont3", "dont4", "dont5"],
    "dress_code": "what to wear",
    "tipping": "tipping culture",
    "greetings": "how to greet locals"
}},
"payment_info": {{
    "preferred": "cash or card",
    "credit_cards_accepted": true,
    "atm_availability": "common or rare",
    "notify_bank": true,
    "forex_tips": "best way to get local currency"
}},
"must_have_apps": [
    {{"name": "app name", "purpose": "what it does", "platform": "iOS/Android/Both"}},
    {{"name": "app name", "purpose": "what it does", "platform": "iOS/Android/Both"}},
    {{"name": "app name", "purpose": "what it does", "platform": "iOS/Android/Both"}}
],
"what_to_buy": ["item1", "item2", "item3", "item4"],
"what_to_avoid": ["item1", "item2", "item3"],
"power_plug": {{
    "type": "plug type",
    "voltage": "voltage",
    "adapter_needed": true
}},
"insurance": {{
    "recommended": true,
    "type": "travel insurance type",
    "estimated_cost": "{currency} amount",
    "must_cover": ["coverage1", "coverage2"]
}},
"tips": ["tip1", "tip2", "tip3", "tip4", "tip5"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert travel planner. Always return valid JSON only. No markdown, no backticks, no explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )

        result = response.choices[0].message.content.strip()
        # Clean JSON
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'):
                result = result[4:]
        
        import json
        plan = json.loads(result)
        return jsonify({'success': True, 'plan': plan})

    except Exception as e:
        print(f"Plan error: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ── WEATHER ──
@app.route('/weather', methods=['POST'])
def get_weather():
    try:
        data = request.get_json()
        city = data.get('city', '')
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_KEY}&units=metric&cnt=40"
        response = requests.get(url)
        weather_data = response.json()

        if weather_data.get('cod') != '200':
            return jsonify({'success': False, 'error': 'City not found'})

        # Process forecast
        daily = {}
        for item in weather_data['list']:
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

        forecast = list(daily.values())[:7]

        return jsonify({
            'success': True,
            'city': weather_data['city']['name'],
            'country': weather_data['city']['country'],
            'forecast': forecast
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ── CURRENCY ──
@app.route('/currency', methods=['POST'])
def convert_currency():
    try:
        data = request.get_json()
        amount = float(data.get('amount', 1))
        from_curr = data.get('from', 'INR').upper()
        to_curr = data.get('to', 'USD').upper()

        url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_KEY}/pair/{from_curr}/{to_curr}/{amount}"
        response = requests.get(url)
        result = response.json()

        if result.get('result') != 'success':
            return jsonify({'success': False, 'error': 'Currency not found'})

        return jsonify({
            'success': True,
            'from': from_curr,
            'to': to_curr,
            'amount': amount,
            'converted': round(result['conversion_result'], 2),
            'rate': result['conversion_rate']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ── VISA CHECK ──
@app.route('/visa', methods=['POST'])
def check_visa():
    try:
        data = request.get_json()
        passport = data.get('passport', 'India')
        destination = data.get('destination', '')

        prompt = f"""A traveller with a {passport} passport wants to visit {destination}.

Return ONLY a JSON object:
{{
  "visa_required": true or false,
  "visa_type": "type of visa",
  "validity": "how long",
  "cost": "approximate cost in USD",
  "processing_days": "number of days",
  "apply_online": true or false,
  "apply_url": "url if online",
  "documents": ["doc1", "doc2", "doc3"],
  "tips": ["tip1", "tip2"],
  "visa_on_arrival": true or false,
  "visa_free_days": 0
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return valid JSON only. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )

        import json
        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'):
                result = result[4:]

        visa_info = json.loads(result)
        return jsonify({'success': True, 'visa': visa_info})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ── PACKING LIST ──
@app.route('/packing', methods=['POST'])
def generate_packing():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        days = data.get('days', 5)
        vibe = data.get('vibe', 'adventure')
        weather = data.get('weather', 'moderate')

        prompt = f"""Generate a packing list for {days} days in {destination}.
Weather: {weather}, Trip style: {vibe}

Return ONLY JSON:
{{
  "essentials": ["item1", "item2", "item3", "item4", "item5"],
  "clothing": ["item1", "item2", "item3", "item4", "item5"],
  "toiletries": ["item1", "item2", "item3", "item4"],
  "electronics": ["item1", "item2", "item3"],
  "documents": ["item1", "item2", "item3", "item4"],
  "health": ["item1", "item2", "item3"],
  "destination_specific": ["item1", "item2", "item3"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )

        import json
        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'):
                result = result[4:]

        packing = json.loads(result)
        return jsonify({'success': True, 'packing': packing})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/journey', methods=['POST'])
def plan_journey():

    try:
        data = request.get_json()
        origin = data.get('origin', '')
        destination = data.get('destination', '')
        travel_mode = data.get('travel_mode', 'any')
        currency = data.get('currency', 'INR')

        prompt = f"""You are an expert travel logistics planner.

Someone wants to travel from "{origin}" to "{destination}".
Their preferred mode: {travel_mode}
Show all prices in {currency}

Analyze this journey completely and return ONLY valid JSON:
{{
  "origin": "{origin}",
  "destination": "{destination}",
  "origin_has_airport": true or false,
  "nearest_airports": [
    {{
      "name": "airport name",
      "city": "city name",
      "code": "IATA code",
      "distance_from_origin": "km",
      "ways_to_reach": [
        {{
          "mode": "Train/Bus/Cab/Metro",
          "operator": "operator name",
          "duration": "hours",
          "cost": "{currency} amount",
          "frequency": "how often",
          "booking": "where to book",
          "tip": "insider tip"
        }}
      ]
    }}
  ],
  "destination_airports": [
    {{
      "name": "airport name",
      "city": "city",
      "code": "IATA code",
      "distance_from_city": "km",
      "transport_to_city": [
        {{
          "mode": "mode",
          "cost": "{currency} amount",
          "duration": "time"
        }}
      ]
    }}
  ],
  "recommended_route": {{
    "step1": "exact step",
    "step2": "exact step",
    "step3": "exact step",
    "step4": "exact step",
    "total_duration": "total travel time",
    "total_cost": "{currency} amount"
  }},
  "flight_options": [
    {{
      "from_airport": "airport code",
      "to_airport": "airport code",
      "airlines": ["airline1", "airline2"],
      "duration": "flight time",
      "stops": 0,
      "estimated_cost": "{currency} amount",
      "best_booking_sites": ["site1", "site2"],
      "cheapest_days": "best days to fly",
      "book_advance": "how many days before"
    }}
  ],
  "alternative_routes": [
    {{
      "title": "route name",
      "description": "full description",
      "total_cost": "{currency} amount",
      "total_time": "hours",
      "pros": ["pro1", "pro2"],
      "cons": ["con1"]
    }}
  ],
  "important_notes": ["note1", "note2", "note3"],
  "documents_needed": ["doc1", "doc2", "doc3"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert travel logistics planner. Return ONLY valid JSON. No markdown, no backticks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=3000
        )

        import json
        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'):
                result = result[4:]

        journey = json.loads(result)
        return jsonify({'success': True, 'journey': journey})

    except Exception as e:
        print(f"Journey error: {e}")
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/discover')
def discover_page():
    return render_template('discover.html')

@app.route('/identify', methods=['POST'])
def identify_place():
    try:
        data = request.get_json()
        image_base64 = data.get('image', '')

        if not image_base64:
            return jsonify({'success': False, 'error': 'No image provided'})

        # Use Groq Vision directly — much more powerful
        import json

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """You are a world-class travel and geography expert with encyclopedic knowledge of every place on Earth.

Analyze this image carefully and identify the exact location shown.

Look for:
- Architectural style, building features, landmarks
- Natural features (mountains, beaches, water bodies)
- Vegetation, climate indicators
- Any text, signs, or symbols visible
- Distinctive cultural elements
- Sky, lighting conditions
- People's clothing and appearance

Return ONLY valid JSON with no markdown:
{
  "place_name": "exact place or landmark name",
  "city": "city name",
  "region": "state/province/region",
  "country": "country name",
  "continent": "continent",
  "confidence": 92,
  "place_type": "Natural Wonder/Historical Site/City/Beach/Mountain/Temple/etc",
  "description": "2-3 sentence vivid description of this place",
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "best_time": "months to visit",
  "climate": "weather type eg Tropical/Mediterranean/etc",
  "budget_level": "Budget/Mid-range/Luxury",
  "avg_daily_cost": "USD X per day",
  "language": "local language",
  "currency": "local currency name + code",
  "nearest_airport": "airport name",
  "airport_code": "IATA code",
  "why_famous": "what makes this place special in 1 sentence",
  "instagram_spots": ["best photo spot 1", "best photo spot 2", "best photo spot 3"],
  "nearby": [
    {"name": "place name", "distance": "X km", "description": "brief desc", "type": "Attraction", "icon": "🏛️"},
    {"name": "place name", "distance": "X km", "description": "brief desc", "type": "Hidden Gem", "icon": "💎"},
    {"name": "place name", "distance": "X km", "description": "brief desc", "type": "Restaurant", "icon": "🍽️"},
    {"name": "place name", "distance": "X km", "description": "brief desc", "type": "Activity", "icon": "🎯"}
  ],
  "similar_places": [
    {"name": "similar place", "country": "country", "why_similar": "short reason", "emoji": "🇫🇷"},
    {"name": "similar place", "country": "country", "why_similar": "short reason", "emoji": "🇮🇹"},
    {"name": "similar place", "country": "country", "why_similar": "short reason", "emoji": "🇬🇷"}
  ],
  "travel_tips": ["specific tip 1", "specific tip 2", "specific tip 3"],
  "best_food": ["local dish 1", "local dish 2", "local dish 3"]
}

If you cannot identify with 100% certainty, make your best educated guess. Never return unknown or blank fields."""
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content.strip()

        # Clean JSON
        if '```' in result_text:
            parts = result_text.split('```')
            for part in parts:
                if '{' in part:
                    result_text = part
                    if result_text.startswith('json'):
                        result_text = result_text[4:]
                    break

        # Find JSON object
        start = result_text.find('{')
        end = result_text.rfind('}') + 1
        if start != -1 and end > start:
            result_text = result_text[start:end]

        result = json.loads(result_text)
        print(f"Identified: {result.get('place_name')} in {result.get('country')}")
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        print(f"Identify error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/identify-text', methods=['POST'])
def identify_from_text():
    try:
        data = request.get_json()
        description = data.get('description', '').strip()

        if not description:
            return jsonify({'success': False, 'error': 'No description provided'})

        prompt = f"""You are an expert travel and geography AI.

Someone described a place they saw on social media: "{description}"

Identify the most likely place this description refers to and return ONLY valid JSON:
{{
  "place_name": "exact place or landmark name",
  "city": "city name",
  "region": "state/province/region",
  "country": "country name",
  "continent": "continent",
  "confidence": 88,
  "place_type": "Natural Wonder/Historical Site/City/Beach/Mountain/Temple/etc",
  "description": "2-3 sentence description of this place",
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "best_time": "months to visit",
  "climate": "weather description",
  "budget_level": "Budget/Mid-range/Luxury",
  "avg_daily_cost": "USD amount per day",
  "language": "local language",
  "currency": "local currency + code",
  "nearest_airport": "airport name",
  "airport_code": "IATA code",
  "why_famous": "what makes this place special",
  "instagram_spots": ["best photo spot 1", "best photo spot 2", "best photo spot 3"],
  "nearby": [
    {{"name": "place name", "distance": "X km", "description": "brief desc", "type": "Attraction", "icon": "emoji"}},
    {{"name": "place name", "distance": "X km", "description": "brief desc", "type": "Hidden Gem", "icon": "emoji"}},
    {{"name": "place name", "distance": "X km", "description": "brief desc", "type": "Restaurant", "icon": "emoji"}}
  ],
  "similar_places": [
    {{"name": "similar place", "country": "country", "why_similar": "reason", "emoji": "flag emoji"}},
    {{"name": "similar place", "country": "country", "why_similar": "reason", "emoji": "flag emoji"}},
    {{"name": "similar place", "country": "country", "why_similar": "reason", "emoji": "flag emoji"}}
  ],
  "travel_tips": ["tip1", "tip2", "tip3"],
  "best_food": ["local dish 1", "local dish 2", "local dish 3"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a world geography and travel expert. Return ONLY valid JSON. No markdown. No backticks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        import json
        result_text = response.choices[0].message.content.strip()
        if '```' in result_text:
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]

        result = json.loads(result_text)
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        print(f"Identify text error: {e}")
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/passport-check', methods=['POST'])
def passport_check():
    try:
        data = request.get_json()
        expiry_date = data.get('expiry_date', '')
        destination = data.get('destination', '')
        travel_date = data.get('travel_date', '')

        from datetime import datetime, timedelta
        import json

        expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
        travel = datetime.strptime(travel_date, '%Y-%m-%d')
        today = datetime.now()

        days_until_expiry = (expiry - today).days
        days_until_travel = (travel - today).days
        days_valid_after_travel = (expiry - travel).days

        prompt = f"""A traveller wants to visit {destination}.
Passport expiry: {expiry_date}
Travel date: {travel_date}
Days until passport expires from travel date: {days_valid_after_travel}

Most countries require passport valid for 6 months beyond travel date.

Return ONLY valid JSON:
{{
  "is_valid": true or false,
  "validity_status": "Safe/Warning/Critical/Invalid",
  "days_remaining": {days_until_expiry},
  "days_after_travel": {days_valid_after_travel},
  "destination_requirement": "X months required by {destination}",
  "verdict": "one clear sentence verdict",
  "action_needed": "what they must do if any",
  "renewal_urgency": "Immediate/Soon/Not needed",
  "renewal_time": "how long passport renewal takes in India",
  "renewal_cost": "approximate cost in INR",
  "tatkal_available": true or false,
  "tatkal_time": "days for tatkal",
  "tatkal_cost": "INR amount",
  "tips": ["tip1", "tip2"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=800
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]

        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/safety-check', methods=['POST'])
def safety_check():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        import json

        prompt = f"""You are a travel safety expert. Provide comprehensive safety information for {destination}.

Return ONLY valid JSON:
{{
  "safety_score": 75,
  "safety_level": "Safe/Moderate/Caution/High Risk",
  "safety_color": "green/yellow/orange/red",
  "crime_index": "Low/Medium/High",
  "tourist_safety": "Safe/Generally Safe/Exercise Caution/Avoid",
  "water_safe": true or false,
  "water_advice": "tap water advice",
  "food_safety": "Safe/Mostly Safe/Be Careful",
  "food_tips": ["tip1", "tip2"],
  "health_risks": ["risk1", "risk2"],
  "natural_disasters": ["earthquake season/flood season/etc or None"],
  "scams_to_avoid": ["common scam 1", "common scam 2", "common scam 3"],
  "safe_areas": ["safe area 1", "safe area 2"],
  "avoid_areas": ["area to avoid 1", "area to avoid 2"],
  "emergency_embassy": "Indian embassy address in {destination}",
  "embassy_phone": "phone number",
  "travel_advisory": "current travel advisory level",
  "advisory_source": "government source",
  "lgbtq_safety": "Safe/Moderate/Caution/Illegal",
  "solo_female_safety": "Safe/Generally Safe/Take Precautions/Not Recommended",
  "best_safety_tips": ["tip1", "tip2", "tip3", "tip4", "tip5"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=1200
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]

        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/local-laws', methods=['POST'])
def local_laws():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        import json

        prompt = f"""You are a legal travel expert. List important laws and rules for tourists visiting {destination}.

Return ONLY valid JSON:
{{
  "destination": "{destination}",
  "strict_laws": [
    {{"law": "law description", "penalty": "penalty if broken", "severity": "High/Medium/Low", "icon": "emoji"}},
    {{"law": "law description", "penalty": "penalty", "severity": "High", "icon": "emoji"}},
    {{"law": "law description", "penalty": "penalty", "severity": "High", "icon": "emoji"}},
    {{"law": "law description", "penalty": "penalty", "severity": "Medium", "icon": "emoji"}},
    {{"law": "law description", "penalty": "penalty", "severity": "Medium", "icon": "emoji"}}
  ],
  "photography_rules": ["what you cannot photograph", "restricted areas"],
  "dress_code_rules": ["specific dress requirements"],
  "alcohol_rules": "alcohol laws",
  "drug_laws": "very strict - details",
  "customs_limits": {{
    "cash": "max cash you can carry",
    "cigarettes": "limit",
    "alcohol": "limit",
    "prohibited_items": ["item1", "item2"]
  }},
  "good_to_know": ["cultural rule 1", "cultural rule 2", "cultural rule 3"],
  "legal_tip": "most important single tip for tourists"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=1500
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]

        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/jetlag', methods=['POST'])
def jetlag_calc():
    try:
        data = request.get_json()
        from_city = data.get('from_city', '')
        to_city = data.get('to_city', '')
        travel_date = data.get('travel_date', '')
        import json

        prompt = f"""Calculate jet lag and recovery plan for travelling from {from_city} to {to_city} on {travel_date}.

Return ONLY valid JSON:
{{
  "from_timezone": "timezone name + UTC offset",
  "to_timezone": "timezone name + UTC offset",
  "time_difference": "X hours ahead/behind",
  "jet_lag_severity": "Mild/Moderate/Severe",
  "recovery_days": 2,
  "direction": "Eastward/Westward",
  "eastward_harder": true,
  "symptoms": ["fatigue", "insomnia", "etc"],
  "before_flight": [
    {{"action": "specific action", "timing": "when to do it", "why": "reason"}}
  ],
  "during_flight": [
    {{"action": "specific action", "timing": "when", "why": "reason"}}
  ],
  "after_arrival": [
    {{"action": "specific action", "timing": "when", "why": "reason"}}
  ],
  "sleep_schedule": {{
    "night_before_flight": "recommended bedtime",
    "on_arrival": "recommended bedtime local time",
    "day_2": "recommended schedule"
  }},
  "avoid": ["avoid caffeine after X", "avoid alcohol", "etc"],
  "recovery_tip": "single best tip for fast recovery"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=1200
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]

        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/festivals', methods=['POST'])
def festivals():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        travel_date = data.get('travel_date', '')
        import json

        prompt = f"""List festivals, public holidays, and special events in {destination} around {travel_date}.

Return ONLY valid JSON:
{{
  "destination": "{destination}",
  "travel_month": "month name",
  "public_holidays": [
    {{"name": "holiday name", "date": "date", "impact": "what closes/opens", "tourist_impact": "positive/negative/neutral"}}
  ],
  "festivals": [
    {{"name": "festival name", "dates": "date range", "description": "brief desc", "tourist_friendly": true, "special_tips": "tip", "icon": "emoji"}}
  ],
  "peak_season": true or false,
  "season_type": "Peak/Shoulder/Off-peak",
  "price_impact": "prices X% higher/lower than average",
  "crowd_level": "Very Crowded/Crowded/Moderate/Quiet",
  "booking_advice": "book X weeks in advance",
  "weather_this_month": "weather description",
  "special_experiences": ["unique thing happening this month 1", "unique thing 2"],
  "avoid_dates": ["specific dates to avoid if any"],
  "best_festival_tip": "top tip for experiencing local festivals"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, max_tokens=1200
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]

        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/budget-plan', methods=['POST'])
def budget_plan():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        days = data.get('days', 5)
        budget = data.get('budget', 1000)
        currency = data.get('currency', 'USD')
        people = data.get('people', 1)
        import json

        prompt = f"""Create a detailed budget plan for {people} people visiting {destination} for {days} days.
Total budget: {currency} {budget}

Return ONLY valid JSON:
{{
  "destination": "{destination}",
  "total_budget": "{currency} {budget}",
  "per_person": "{currency} {int(float(str(budget).replace(',','')))/int(people):.0f}",
  "per_day": "{currency} {int(float(str(budget).replace(',','')))/int(days):.0f}",
  "budget_tier": "Budget/Mid-range/Luxury",
  "breakdown": {{
    "flights": {{"amount": "{currency} X", "percentage": 35, "tips": "book X weeks ahead"}},
    "accommodation": {{"amount": "{currency} X", "percentage": 25, "tips": "area recommendation"}},
    "food": {{"amount": "{currency} X", "percentage": 15, "tips": "where to eat"}},
    "transport": {{"amount": "{currency} X", "percentage": 10, "tips": "best transport"}},
    "activities": {{"amount": "{currency} X", "percentage": 10, "tips": "free vs paid"}},
    "shopping": {{"amount": "{currency} X", "percentage": 5, "tips": "what to buy"}}
  }},
  "daily_budget": {{
    "budget_day": "{currency} X - street food, hostel, free sights",
    "comfort_day": "{currency} X - restaurant, hotel, some paid attractions",
    "splurge_day": "{currency} X - fine dining, nice hotel, premium experiences"
  }},
  "money_saving_tips": ["tip1", "tip2", "tip3", "tip4", "tip5"],
  "hidden_costs": ["visa fee", "airport tax", "travel insurance", "tip X"],
  "free_things": ["free thing 1", "free thing 2", "free thing 3"],
  "worth_splurging": ["experience worth spending extra on"],
  "budget_verdict": "Is this budget realistic? One sentence verdict."
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=1500
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]

        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/luggage-check', methods=['POST'])
def luggage_check():
    try:
        data = request.get_json()
        airline = data.get('airline', '')
        cabin = data.get('cabin_class', 'Economy')
        destination = data.get('destination', '')
        import json

        prompt = f"""Provide complete luggage and duty free allowance information for {airline} airline flying to {destination} in {cabin} class.

Return ONLY valid JSON:
{{
  "airline": "{airline}",
  "cabin_class": "{cabin}",
  "carry_on": {{"weight": "kg", "dimensions": "cm", "pieces": 1}},
  "checked_baggage": {{"weight": "kg", "dimensions": "cm", "pieces": 1, "extra_cost": "per kg"}},
  "prohibited_items": ["item1", "item2", "item3"],
  "liquid_rules": "100ml rule details",
  "duty_free_allowance": {{
    "alcohol": "limit",
    "cigarettes": "limit",
    "cash": "limit",
    "gifts": "value limit",
    "electronics": "rules"
  }},
  "customs_destination": {{
    "declaration_required": true,
    "items_to_declare": ["item1", "item2"],
    "prohibited_items": ["item1", "item2"],
    "penalties": "what happens if caught"
  }},
  "packing_tips": ["tip1", "tip2", "tip3"],
  "weight_saving_tips": ["tip1", "tip2"],
  "pro_tip": "best single tip for this airline/route"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=1000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]

        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/emergency-card', methods=['POST'])
def emergency_card():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        name = data.get('name', '')
        blood_group = data.get('blood_group', '')
        allergies = data.get('allergies', '')
        emergency_contact = data.get('emergency_contact', '')
        import json

        prompt = f"""Create a complete emergency contact card for a traveller visiting {destination}.

Traveller: {name}
Blood Group: {blood_group}
Allergies: {allergies}

Return ONLY valid JSON:
{{
  "destination": "{destination}",
  "emergency_numbers": {{
    "police": "number",
    "ambulance": "number",
    "fire": "number",
    "tourist_helpline": "number",
    "women_helpline": "number"
  }},
  "indian_embassy": {{
    "address": "full address",
    "phone": "number",
    "emergency_phone": "24hr number",
    "email": "email"
  }},
  "nearest_hospitals": [
    {{"name": "hospital name", "type": "Government/Private", "phone": "number", "specialty": "specialty"}}
  ],
  "medical_phrases": [
    {{"english": "I need help", "local": "translation", "pronunciation": "how to say"}},
    {{"english": "Call an ambulance", "local": "translation", "pronunciation": "how to say"}},
    {{"english": "I am allergic to {allergies if allergies else 'nothing'}", "local": "translation", "pronunciation": "how to say"}},
    {{"english": "My blood group is {blood_group}", "local": "translation", "pronunciation": "how to say"}}
  ],
  "what_to_do_if_robbed": ["step1", "step2", "step3"],
  "what_to_do_if_sick": ["step1", "step2", "step3"],
  "what_to_do_if_lost": ["step1", "step2", "step3"],
  "important_apps": ["emergency app 1", "emergency app 2"],
  "card_tip": "most important thing to remember"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=1500
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]

        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/tools')
def tools_page():
    return render_template('tools_extra.html')

@app.route('/during')
def during_page():
    return render_template('during_trip.html')

@app.route('/medical-translate', methods=['POST'])
def medical_translate():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        destination = data.get('destination', '')
        language = data.get('language', 'Japanese')
        import json

        prompt = f"""You are a medical translator and emergency travel assistant.

Patient symptoms: {symptoms}
Destination: {destination}
Translate to: {language}

Return ONLY valid JSON:
{{
  "severity": "Minor/Moderate/Severe/Emergency",
  "severity_color": "green/yellow/orange/red",
  "possible_conditions": ["condition1", "condition2"],
  "translated_symptoms": "symptoms in {language}",
  "pronunciation": "how to pronounce it",
  "say_to_doctor": "exact phrase to say to doctor in {language}",
  "say_to_doctor_english": "English version",
  "say_pronunciation": "pronunciation guide",
  "immediate_actions": ["action1", "action2", "action3"],
  "medicines_to_ask": [
    {{"english_name": "medicine", "local_name": "local equivalent", "purpose": "what it treats"}}
  ],
  "avoid_doing": ["avoid1", "avoid2"],
  "nearest_hospital_search": "exact Google search to find nearest hospital",
  "emergency_number": "emergency number in {destination}",
  "when_to_call_ambulance": "specific symptoms that need ambulance",
  "home_remedies": ["remedy1", "remedy2"],
  "medical_phrases": [
    {{"english": "I need a doctor", "local": "translation", "pronunciation": "pronunciation"}},
    {{"english": "I have pain here", "local": "translation", "pronunciation": "pronunciation"}},
    {{"english": "I am allergic to", "local": "translation", "pronunciation": "pronunciation"}},
    {{"english": "Call an ambulance", "local": "translation", "pronunciation": "pronunciation"}}
  ]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a medical translator. Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=2000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/price-check', methods=['POST'])
def price_check():
    try:
        data = request.get_json()
        item = data.get('item', '')
        price = data.get('price', '')
        currency = data.get('currency', '')
        destination = data.get('destination', '')
        import json

        prompt = f"""You are a local price expert for {destination}.

Item/Service: {item}
Price being charged: {currency} {price}
Location: {destination}

Is this price fair for a tourist? Return ONLY valid JSON:
{{
  "verdict": "Great Deal/Fair Price/Slightly Overpriced/Overpriced/Scam",
  "verdict_color": "green/blue/yellow/orange/red",
  "fair_price_range": "{currency} X - {currency} Y",
  "local_price": "what locals pay",
  "tourist_price": "normal tourist price",
  "you_are_paying": "{currency} {price}",
  "overpaying_by": "amount or percentage if overpriced",
  "verdict_explanation": "clear explanation why",
  "negotiation_tips": ["tip1", "tip2", "tip3"],
  "how_to_negotiate": "exact script for negotiating in local language",
  "negotiation_english": "English version",
  "walk_away_price": "price at which you should walk away",
  "better_alternatives": ["alternative1", "alternative2"],
  "scam_warning": true or false,
  "scam_details": "details if scam",
  "local_phrase_to_say": "what to say in local language",
  "comparison": {{
    "vs_uber": "comparison if transport",
    "vs_local_restaurant": "comparison if food",
    "vs_official_price": "comparison if attraction"
  }}
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a local price expert. Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=1500
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/scam-alerts', methods=['POST'])
def scam_alerts():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        import json

        prompt = f"""You are a travel scam expert with deep knowledge of {destination}.

List ALL common scams tourists face in {destination} with detailed how-to-avoid guides.

Return ONLY valid JSON:
{{
  "destination": "{destination}",
  "scam_risk_level": "Low/Medium/High/Very High",
  "most_targeted": "which tourists are most targeted",
  "scams": [
    {{
      "name": "scam name",
      "category": "Transport/Food/Shopping/Attraction/Accommodation/Money/Distraction",
      "severity": "Annoying/Costly/Dangerous",
      "how_it_works": "detailed explanation",
      "red_flags": ["warning sign 1", "warning sign 2"],
      "how_to_avoid": "detailed avoidance strategy",
      "what_to_say": "exact phrase to shut them down",
      "what_to_say_local": "same in local language",
      "if_it_happens": "what to do if you fall for it",
      "icon": "emoji"
    }}
  ],
  "general_rules": ["rule1", "rule2", "rule3", "rule4", "rule5"],
  "safe_alternatives": {{
    "transport": "safest transport option",
    "money_exchange": "safest place to exchange money",
    "shopping": "safest place to shop",
    "food": "safest way to find food"
  }},
  "report_scam_to": "how/where to report scams in {destination}",
  "emergency_if_robbed": ["step1", "step2", "step3"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a travel scam expert. Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=3000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/allergy-card', methods=['POST'])
def allergy_card():
    try:
        data = request.get_json()
        allergies = data.get('allergies', [])
        destination = data.get('destination', '')
        name = data.get('name', '')
        import json

        allergies_str = ', '.join(allergies) if allergies else 'none'

        prompt = f"""You are a food allergy and travel safety expert.

Traveller: {name}
Allergies: {allergies_str}
Destination: {destination}

Create a comprehensive allergy safety guide. Return ONLY valid JSON:
{{
  "destination": "{destination}",
  "allergies": {json.dumps(allergies)},
  "allergy_card_text": {{
    "english": "I am severely allergic to {allergies_str}. This is a medical condition. Please ensure my food contains none of these ingredients.",
    "local_language": "translation in local language of {destination}",
    "pronunciation": "pronunciation guide"
  }},
  "dangerous_dishes": [
    {{"dish_name": "dish", "why_dangerous": "contains X allergen", "local_name": "local name", "severity": "High/Medium"}}
  ],
  "safe_dishes": [
    {{"dish_name": "safe dish", "why_safe": "naturally free of allergens", "local_name": "local name"}}
  ],
  "hidden_allergens": [
    {{"allergen": "allergen", "hidden_in": "commonly hidden in these foods", "local_term": "how its listed locally"}}
  ],
  "phrases_to_say": [
    {{"situation": "at restaurant", "english": "phrase", "local": "translation", "pronunciation": "pronunciation"}},
    {{"situation": "at food stall", "english": "phrase", "local": "translation", "pronunciation": "pronunciation"}},
    {{"situation": "at market", "english": "phrase", "local": "translation", "pronunciation": "pronunciation"}},
    {{"situation": "emergency", "english": "I am having an allergic reaction", "local": "translation", "pronunciation": "pronunciation"}}
  ],
  "restaurant_tips": ["tip1", "tip2", "tip3"],
  "emergency_protocol": ["step1", "step2", "step3"],
  "medicines_to_carry": ["medicine1", "medicine2"],
  "epipen_info": "epipen availability in {destination}",
  "hospital_phrase": "take me to hospital in local language",
  "hospital_pronunciation": "pronunciation"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a food allergy travel expert. Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=2000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/flight-rights', methods=['POST'])
def flight_rights():
    try:
        data = request.get_json()
        airline = data.get('airline', '')
        issue = data.get('issue', '')
        route = data.get('route', '')
        delay_hours = data.get('delay_hours', 0)
        import json

        prompt = f"""You are an aviation rights expert.

Airline: {airline}
Route: {route}
Issue: {issue}
Delay: {delay_hours} hours

Tell the traveller exactly what their rights are and how to claim compensation. Return ONLY valid JSON:
{{
  "issue_type": "{issue}",
  "entitled_to_compensation": true or false,
  "compensation_amount": "exact amount in EUR/USD",
  "compensation_basis": "which regulation applies",
  "your_rights": ["right1", "right2", "right3", "right4"],
  "immediate_actions": [
    {{"step": 1, "action": "exact action", "why": "reason", "timing": "when to do it"}}
  ],
  "documents_to_collect": ["document1", "document2", "document3"],
  "what_airline_must_provide": ["meals", "hotel", "transport", "etc based on delay"],
  "how_to_claim": [
    {{"step": 1, "action": "claim step", "where": "where to go/call", "timing": "deadline"}}
  ],
  "claim_deadline": "how many days/months to claim",
  "escalation_options": ["option1", "option2"],
  "exact_phrases_to_say": [
    {{"say_this": "exact phrase to say to airline staff", "situation": "when to say it"}}
  ],
  "airline_compensation_email": "compensation email if known",
  "regulator": "aviation regulator for this route",
  "regulator_contact": "regulator contact details",
  "claim_template": "short email template to claim compensation",
  "success_rate": "typical success rate for this type of claim",
  "pro_tips": ["tip1", "tip2", "tip3"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an aviation rights expert. Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=2000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/safe-route', methods=['POST'])
def safe_route():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        from_location = data.get('from_location', '')
        to_location = data.get('to_location', '')
        time_of_day = data.get('time_of_day', 'evening')
        traveller_type = data.get('traveller_type', 'solo female')
        import json

        prompt = f"""You are a travel safety expert for {destination}.

From: {from_location}
To: {to_location}
Time: {time_of_day}
Traveller: {traveller_type}

Provide a complete safety route guide. Return ONLY valid JSON:
{{
  "destination": "{destination}",
  "route_safety": "Safe/Moderate/Use Caution/Avoid at Night",
  "safety_score": 75,
  "recommended_transport": "safest transport option",
  "transport_options": [
    {{
      "mode": "transport mode",
      "safety_rating": "Safe/Moderate/Risky",
      "cost": "approximate cost",
      "duration": "time",
      "tips": "safety tips for this mode",
      "recommended_apps": ["app1", "app2"]
    }}
  ],
  "areas_to_avoid": ["area1", "area2"],
  "safe_checkpoints": ["landmark1", "landmark2"],
  "time_specific_warnings": ["warning for this time of day"],
  "if_followed": ["step1", "step2", "step3"],
  "if_harassed": ["step1", "step2", "step3"],
  "trusted_contacts": {{
    "police": "number",
    "women_helpline": "number",
    "tourist_helpline": "number"
  }},
  "share_location_with": "who to share live location with",
  "safety_apps": ["app1", "app2"],
  "local_phrase_help": "help me I am lost in local language",
  "local_phrase_emergency": "emergency phrase in local language",
  "pro_tips": ["tip1", "tip2", "tip3", "tip4", "tip5"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a travel safety expert. Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=2000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/immigration-help', methods=['POST'])
def immigration_help():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        passport = data.get('passport', 'India')
        purpose = data.get('purpose', 'Tourism')
        import json

        prompt = f"""You are an immigration expert.

Traveller from: {passport}
Destination: {destination}
Purpose: {purpose}

Provide complete immigration guidance. Return ONLY valid JSON:
{{
  "destination": "{destination}",
  "common_questions": [
    {{
      "question_english": "exact immigration question",
      "question_local": "in local language",
      "best_answer": "exact answer to give",
      "why_they_ask": "reason for this question",
      "avoid_saying": "what not to say"
    }}
  ],
  "documents_to_keep_ready": ["document1", "document2", "document3"],
  "declaration_items": ["item to declare1", "item to declare2"],
  "do_not_bring": ["prohibited item1", "prohibited item2"],
  "form_filling_tips": ["tip1", "tip2", "tip3"],
  "dress_code_for_immigration": "what to wear",
  "common_mistakes": ["mistake1", "mistake2", "mistake3"],
  "if_stopped_for_questioning": ["step1", "step2", "step3"],
  "if_denied_entry": ["step1", "step2", "step3"],
  "expected_wait_time": "approximate wait time",
  "fast_track_options": "any fast track options available",
  "immigration_phrases": [
    {{"english": "phrase", "local": "translation", "when": "when to use"}}
  ],
  "pro_tips": ["tip1", "tip2", "tip3"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an immigration expert. Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=2000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/place-photo', methods=['POST'])
def get_place_photo():
    try:
        data = request.get_json()
        place_name = data.get('place_name', '')
        UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

        if not UNSPLASH_KEY:
            return jsonify({'success': False, 'error': 'No key'})

        res = requests.get(
            "https://api.unsplash.com/search/photos",
            params={
                'query': f"{place_name} travel",
                'per_page': 5,
                'orientation': 'landscape',
                'client_id': UNSPLASH_KEY
            }
        )
        results = res.json().get('results', [])

        if not results:
            res = requests.get(
                "https://api.unsplash.com/search/photos",
                params={
                    'query': place_name,
                    'per_page': 3,
                    'client_id': UNSPLASH_KEY
                }
            )
            results = res.json().get('results', [])

        photos = [r['urls']['regular'] for r in results if r.get('urls', {}).get('regular')]

        if not photos:
            return jsonify({'success': False, 'error': 'No photos'})

        print(f"Found {len(photos)} Unsplash photos for '{place_name}'")
        return jsonify({'success': True, 'photos': photos})

    except Exception as e:
        print(f"Photo error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/detect-theme', methods=['POST'])
def detect_theme():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        import json

        prompt = f"""Analyze the destination "{destination}" and determine its visual theme for a travel app.

Return ONLY valid JSON:
{{
  "destination_type": "Beach/Mountain/City/Historical/Desert/Tropical/Arctic/Cultural/Island/Forest",
  "theme": {{
    "primary_color": "#hexcolor",
    "secondary_color": "#hexcolor",
    "accent_color": "#hexcolor",
    "text_on_primary": "#ffffff or #000000",
    "gradient_start": "#hexcolor",
    "gradient_end": "#hexcolor",
    "mood": "Tropical Paradise/Urban Sophistication/Mountain Adventure/Ancient Mystique/Desert Dreams/Arctic Wonder/Island Bliss/Forest Escape",
    "font_style": "serif or sans",
    "card_style": "dark or light or frosted",
    "emoji": "single emoji representing destination"
  }},
  "color_rationale": "why these colors fit this destination",
  "season": "best season to visit",
  "vibe_words": ["word1", "word2", "word3"]
}}

Examples:
- Bali: tropical greens and sunset oranges
- Paris: elegant navy and gold
- Tokyo: neon purple and cyan
- Sahara: warm sand and burnt orange
- Iceland: icy blue and aurora green
- New York: concrete grey and yellow taxi
- Santorini: white and deep blue"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]

        return jsonify({'success': True, 'theme': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/after')
def after_page():
    return render_template('after_trip.html')


@app.route('/trip-journal', methods=['POST'])
def trip_journal():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        days = data.get('days', 5)
        highlights = data.get('highlights', '')
        vibe = data.get('vibe', '')
        travel_with = data.get('travel_with', 'solo')
        import json

        prompt = f"""You are a professional travel writer. Write a beautiful, vivid, personal travel journal for this trip.

Destination: {destination}
Duration: {days} days
Travel style: {vibe}
Travelled with: {travel_with}
Personal highlights/notes: {highlights}

Write as if you ARE the traveller. First person. Emotional. Vivid. Real.

Return ONLY valid JSON:
{{
  "title": "creative trip title",
  "tagline": "one line that captures the trip",
  "opening": "beautiful opening paragraph 3-4 sentences",
  "chapters": [
    {{
      "day": 1,
      "title": "chapter title",
      "story": "2-3 paragraph vivid story of this day",
      "highlight": "best moment of the day",
      "emotion": "emotion of the day",
      "emoji": "emoji"
    }}
  ],
  "closing": "emotional closing paragraph",
  "best_memory": "single best memory",
  "lesson_learned": "what this trip taught you",
  "quote": "travel quote that fits this trip",
  "would_return": true,
  "rating": 9,
  "tags": ["tag1", "tag2", "tag3"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional travel writer. Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        parsed = json.loads(result)
        return jsonify({'success': True, 'data': parsed})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/expense-summary', methods=['POST'])
def expense_summary():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        expenses = data.get('expenses', [])
        budget = data.get('budget', 0)
        currency = data.get('currency', 'INR')
        people = data.get('people', 1)
        import json

        total = sum(float(e.get('amount', 0)) for e in expenses)
        by_category = {}
        for e in expenses:
            cat = e.get('category', 'Other')
            by_category[cat] = by_category.get(cat, 0) + float(e.get('amount', 0))

        prompt = f"""Analyse this trip expense data and give insights.

Destination: {destination}
Budget: {currency} {budget}
Total Spent: {currency} {total:.0f}
People: {people}
Expenses by category: {json.dumps(by_category)}

Return ONLY valid JSON:
{{
  "total_spent": {total:.0f},
  "budget": {budget},
  "saved_or_overspent": {float(budget) - total:.0f},
  "status": "Under Budget/On Budget/Over Budget",
  "per_person": {total/max(people,1):.0f},
  "per_day_avg": 0,
  "biggest_expense_category": "category",
  "best_value_category": "category",
  "insights": ["insight1", "insight2", "insight3"],
  "money_tips_next_trip": ["tip1", "tip2", "tip3"],
  "comparison": "how this compares to average tourist spend in {destination}",
  "verdict": "one line verdict on how well they managed money"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=800
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        summary = json.loads(result)
        summary['by_category'] = by_category
        summary['expenses'] = expenses
        return jsonify({'success': True, 'data': summary})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Journal error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/split-bill', methods=['POST'])
def split_bill():
    try:
        data = request.get_json()
        people = data.get('people', [])
        expenses = data.get('expenses', [])
        currency = data.get('currency', 'INR')
        import json

        # Calculate who owes who
        balances = {p: 0 for p in people}
        total = 0
        for exp in expenses:
            amount = float(exp.get('amount', 0))
            paid_by = exp.get('paid_by', people[0] if people else 'Unknown')
            split_between = exp.get('split_between', people)
            total += amount
            if not split_between: split_between = people
            share = amount / len(split_between)
            if paid_by in balances:
                balances[paid_by] += amount
            for p in split_between:
                if p in balances:
                    balances[p] -= share

        # Simplify debts
        settlements = []
        pos = {k: v for k, v in balances.items() if v > 0.01}
        neg = {k: v for k, v in balances.items() if v < -0.01}

        pos_list = sorted(pos.items(), key=lambda x: -x[1])
        neg_list = sorted(neg.items(), key=lambda x: x[1])

        i, j = 0, 0
        while i < len(pos_list) and j < len(neg_list):
            creditor, credit = pos_list[i]
            debtor, debt = neg_list[j]
            amount = min(credit, -debt)
            if amount > 0.01:
                settlements.append({
                    'from': debtor,
                    'to': creditor,
                    'amount': round(amount, 2),
                    'currency': currency
                })
            pos_list[i] = (creditor, credit - amount)
            neg_list[j] = (debtor, debt + amount)
            if pos_list[i][1] < 0.01: i += 1
            if neg_list[j][1] > -0.01: j += 1

        return jsonify({
            'success': True,
            'data': {
                'total': round(total, 2),
                'per_person': round(total / max(len(people), 1), 2),
                'balances': {k: round(v, 2) for k, v in balances.items()},
                'settlements': settlements,
                'currency': currency,
                'all_settled': len(settlements) == 0
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/trip-stats', methods=['POST'])
def trip_stats():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        days = data.get('days', 5)
        activities = data.get('activities', [])
        expenses = data.get('expenses', [])
        travel_with = data.get('travel_with', 'solo')
        vibes = data.get('vibes', [])
        import json

        total_spent = sum(float(e.get('amount', 0)) for e in expenses)

        prompt = f"""Generate fun, viral travel stats for this trip.

Destination: {destination}
Duration: {days} days
Travelled with: {travel_with}
Activities: {activities}
Total spent: {total_spent}
Travel vibes: {vibes}

Return ONLY valid JSON:
{{
  "traveller_type": "The Foodie Explorer / The Adventure Seeker / The Culture Vulture / etc",
  "traveller_description": "2 sentence fun description of their travel personality",
  "fun_stats": [
    {{"label": "Steps Walked (est)", "value": "42,000", "icon": "👟"}},
    {{"label": "Photos Taken (est)", "value": "380", "icon": "📸"}},
    {{"label": "Local Dishes Tried", "value": "12", "icon": "🍜"}},
    {{"label": "Distance Covered", "value": "2,400 km", "icon": "✈️"}},
    {{"label": "Languages Heard", "value": "3", "icon": "🗣️"}},
    {{"label": "New Friends Made", "value": "7", "icon": "🤝"}}
  ],
  "achievements": [
    {{"title": "achievement name", "description": "what they did", "icon": "emoji", "rarity": "Common/Rare/Epic/Legendary"}}
  ],
  "travel_score": 87,
  "next_destination_prediction": "destination they'll love next",
  "personality_match": "Famous traveller or explorer they are like",
  "carbon_footprint": "estimated CO2 in kg",
  "memorable_moments": ["moment1", "moment2", "moment3"],
  "instagram_caption": "perfect instagram caption for this trip with hashtags"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, max_tokens=2000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/review-generator', methods=['POST'])
def review_generator():
    try:
        data = request.get_json()
        place = data.get('place', '')
        rating = data.get('rating', 5)
        experience = data.get('experience', '')
        platform = data.get('platform', 'Google')
        import json

        prompt = f"""Write a genuine, helpful {platform} review for this place.

Place: {place}
Rating: {rating}/5
Personal experience: {experience}
Platform: {platform}

Return ONLY valid JSON:
{{
  "review_title": "catchy review title",
  "review_body": "2-3 paragraph genuine review, first person, specific details",
  "pros": ["pro1", "pro2", "pro3"],
  "cons": ["con1", "con2"],
  "best_for": "who this place is best for",
  "tip": "one insider tip for future visitors",
  "short_version": "one line review for social media",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Write genuine helpful reviews. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6, max_tokens=1000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/next-trip', methods=['POST'])
def next_trip():
    try:
        data = request.get_json()
        past_destination = data.get('past_destination', '')
        loved = data.get('loved', '')
        budget = data.get('budget', '')
        travel_month = data.get('travel_month', '')
        passport = data.get('passport', 'India')
        import json

        prompt = f"""Suggest perfect next trip destinations based on what this traveller loved.

Past trip: {past_destination}
What they loved: {loved}
Budget: {budget}
Travel month: {travel_month}
Passport: {passport}

Return ONLY valid JSON:
{{
  "recommendations": [
    {{
      "destination": "city, country",
      "why_perfect": "specific reason based on what they loved",
      "similarity_score": 92,
      "best_time": "months",
      "budget_level": "Budget/Mid-range/Luxury",
      "estimated_cost": "amount range",
      "unique_experience": "one thing they absolutely must do",
      "vibe": "vibe description",
      "emoji": "flag emoji",
      "visa_for_india": "visa required or free"
    }}
  ],
  "travel_pattern": "what type of traveller they are based on past trip",
  "bucket_list_suggestion": "one dream destination they should work towards",
  "best_month_to_travel": "optimal travel month based on their preferences"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4, max_tokens=1500
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/currency-leftover', methods=['POST'])
def currency_leftover():
    try:
        data = request.get_json()
        currency = data.get('currency', '')
        amount = data.get('amount', 0)
        home_currency = data.get('home_currency', 'INR')
        import json

        prompt = f"""Give advice on what to do with leftover foreign currency.

Leftover: {currency} {amount}
Home currency: {home_currency}

Return ONLY valid JSON:
{{
  "options": [
    {{
      "option": "option name",
      "description": "detailed explanation",
      "best_for": "who this works best for",
      "estimated_value": "approximate home currency value",
      "rating": 4,
      "pros": ["pro1", "pro2"],
      "cons": ["con1"]
    }}
  ],
  "best_option": "recommended option",
  "tips": ["tip1", "tip2", "tip3"],
  "avoid": "what NOT to do with leftover currency",
  "keep_some": "how much to keep for next trip if applicable"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, max_tokens=1000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            result = result.split('```')[1]
            if result.startswith('json'): result = result[4:]
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1: result = result[start:end]
        return jsonify({'success': True, 'data': json.loads(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/')
def index():
    return render_template('before_trip.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    app.run(debug=True, host='0.0.0.0', port=port)