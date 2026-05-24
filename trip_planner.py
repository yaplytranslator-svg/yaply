from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
from groq import Groq
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
WEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
EXCHANGE_KEY = os.getenv("EXCHANGE_API_KEY")
GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_API_KEY")


# ══════════════════════════════════════════════════════════════
# HELPER — clean JSON from Groq response
# ══════════════════════════════════════════════════════════════
def clean_json(text):
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
    end = text.rfind('}') + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)


# ══════════════════════════════════════════════════════════════
# ── AI TRIP PLANNER (existing) ──
# ══════════════════════════════════════════════════════════════
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
2. budget_breakdown values MUST be plain integers that SUM EXACTLY to {budget}. No currency symbols, no commas — just numbers like 25000 not "INR 25,000"
3. Include "flights" as the FIRST key in budget_breakdown (return flights from {origin})
4. Every single activity, restaurant, hotel price must be in {currency}
5. Tailor activities 100% to the {vibe} travel style
6. Include HIDDEN GEMS — not just tourist spots
7. Include LOCAL TRANSPORT options specific to {destination}
8. Include internet/SIM card costs in {currency}
9. Include vaccination requirements for travellers from {origin}
10. Include cultural dos and donts
11. Include must have apps for {destination}
12. budget_breakdown should cover: flights, accommodation, food, local_transport, activities, shopping, miscellaneous — proportioned realistically

Make the itinerary 100% unique based on the travel style. Different vibes = completely different activities, restaurants and experiences.

Return ONLY a valid JSON object with this exact structure:
{{
  "destination": "{destination}",
  "days": {days},
  "budget_breakdown": {{
    "flights": integer_number_only,
    "accommodation": integer_number_only,
    "food": integer_number_only,
    "local_transport": integer_number_only,
    "activities": integer_number_only,
    "shopping": integer_number_only,
    "miscellaneous": integer_number_only
  }},
  "budget_tips": ["money saving tip 1", "tip 2", "tip 3"],
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

        plan = clean_json(response.choices[0].message.content)
        return jsonify({'success': True, 'plan': plan})

    except Exception as e:
        print(f"Plan error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ══════════════════════════════════════════════════════════════
# ── NEW: MULTI-CITY TRIP PLANNER ──
# POST /multi-city-plan
# Body: { origin, cities: [{name, days}], total_budget, currency,
#         vibe, people, start_date }
# ══════════════════════════════════════════════════════════════
@app.route('/multi-city-plan', methods=['POST'])
def multi_city_plan():
    try:
        data = request.get_json()
        origin        = data.get('origin', 'India')
        cities        = data.get('cities', [])          # [{name, days}]
        total_budget  = data.get('total_budget', 100000)
        currency      = data.get('currency', 'INR')
        vibe          = data.get('vibe', 'adventure')
        people        = data.get('people', 1)
        start_date    = data.get('start_date', '')

        if not cities or len(cities) < 2:
            return jsonify({'success': False,
                            'error': 'Please add at least 2 cities for a multi-city trip.'})

        if len(cities) > 6:
            return jsonify({'success': False,
                            'error': 'Maximum 6 cities supported per trip.'})

        total_days   = sum(int(c.get('days', 3)) for c in cities)
        city_list    = ', '.join(c['name'] for c in cities)
        cities_json  = json.dumps(cities)

        prompt = f"""You are the world's best multi-city trip planner.

Plan an epic multi-city trip:
- Origin: {origin}
- Cities in order: {city_list}
- Days per city: {cities_json}
- Total days: {total_days}
- Total budget: {currency} {total_budget} for {people} people
- Travel style: {vibe}
- Start date: {start_date if start_date else 'flexible'}

CRITICAL RULES:
1. ALL prices in {currency} only
2. Plan each city as its own section with day-by-day itinerary
3. Include TRANSIT between every city (flight/train/bus options with real costs)
4. Budget must sum exactly to {total_budget} — split intelligently across cities
5. Each city gets activities matching the {vibe} style
6. Include SIM card advice for each country change
7. Include visa requirements for each country if different from origin
8. Include the best hidden gems in each city
9. Include packing notes specific to the multi-city route (weather changes, etc)
10. Transit days count as real days — plan activities near airports/stations

Return ONLY valid JSON:
{{
  "trip_title": "catchy name for this multi-city trip",
  "origin": "{origin}",
  "total_days": {total_days},
  "total_budget": {total_budget},
  "currency": "{currency}",
  "cities_count": {len(cities)},
  "route_overview": "one paragraph describing the full journey flow",
  "smart_suggestions": [
    "smart tip 1 about this specific route",
    "smart tip 2 about best order to visit",
    "smart tip 3 about budget allocation",
    "smart tip 4 about visa/transit"
  ],
  "budget_split": {{
    "flights_and_transit": integer,
    "accommodation": integer,
    "food": integer,
    "activities": integer,
    "local_transport": integer,
    "shopping": integer,
    "miscellaneous": integer
  }},
  "cities": [
    {{
      "city_number": 1,
      "city": "city name",
      "country": "country",
      "days": 3,
      "arrival_from": "where they come from",
      "city_budget": integer,
      "city_vibe": "what makes this city special for {vibe} travellers",
      "best_area_to_stay": "neighbourhood recommendation",
      "weather_note": "weather during travel period",
      "language": "local language",
      "local_currency": "currency name and code",
      "currency_tip": "best way to get local currency",
      "itinerary": [
        {{
          "day": 1,
          "day_label": "Day 1 — City Name",
          "theme": "theme for this day",
          "morning": {{"activity": "name", "location": "place", "cost": "{currency} amount", "tip": "insider tip"}},
          "afternoon": {{"activity": "name", "location": "place", "cost": "{currency} amount", "tip": "insider tip"}},
          "evening": {{"activity": "name", "location": "place", "cost": "{currency} amount", "tip": "insider tip"}},
          "lunch": {{"restaurant": "name", "cuisine": "type", "cost": "{currency} amount", "must_order": "dish name"}},
          "dinner": {{"restaurant": "name", "cuisine": "type", "cost": "{currency} amount", "must_order": "dish name"}},
          "accommodation": {{"name": "hotel name", "area": "area", "cost": "{currency} per night", "booking_tip": "tip"}}
        }}
      ],
      "hidden_gems": [
        {{"name": "gem name", "why": "why special", "cost": "{currency} amount", "best_time": "time to visit"}}
      ],
      "must_eat": ["dish1", "dish2", "dish3"],
      "must_do": ["activity1", "activity2", "activity3"],
      "avoid": ["thing to avoid 1", "thing to avoid 2"],
      "local_tips": ["tip1", "tip2", "tip3"]
    }}
  ],
  "transit_plans": [
    {{
      "from": "city1",
      "to": "city2",
      "transit_day": "which day number",
      "options": [
        {{
          "mode": "Flight/Train/Bus/Ferry",
          "operator": "airline or operator name",
          "duration": "travel time",
          "cost": "{currency} amount per person",
          "total_cost": "{currency} amount for {people} people",
          "frequency": "how often",
          "booking_tip": "where and when to book",
          "comfort": "Comfortable/Basic/Luxury",
          "recommended": true or false,
          "reason": "why recommended"
        }}
      ],
      "recommended_option": "mode name",
      "transit_tip": "specific tip for this transit"
    }}
  ],
  "visa_summary": [
    {{
      "country": "country name",
      "visa_for_{origin.lower().replace(' ', '_')}": "visa on arrival/e-visa/required/free",
      "cost": "fee",
      "tip": "visa tip"
    }}
  ],
  "sim_strategy": {{
    "recommendation": "best SIM strategy for this specific multi-country route",
    "options": [
      {{
        "type": "Global eSIM/Local SIM per country/International roaming",
        "providers": ["provider1", "provider2"],
        "cost": "{currency} approximate total",
        "coverage": "which cities covered",
        "recommended": true or false,
        "why": "reason"
      }}
    ],
    "best_apps_for_connectivity": ["app1", "app2"]
  }},
  "packing_for_route": {{
    "weather_variation": "temperature range across all cities",
    "key_items": ["item specific to this route 1", "item2", "item3"],
    "clothing_strategy": "capsule wardrobe advice for this specific route",
    "luggage_tip": "advice on luggage type for this trip"
  }},
  "health_and_safety": {{
    "vaccinations": ["vaccine1", "vaccine2"],
    "health_tips": ["tip1", "tip2"],
    "insurance_advice": "specific advice for multi-country travel"
  }},
  "money_saving_tips": ["tip1", "tip2", "tip3", "tip4"],
  "trip_highlights": ["highlight1", "highlight2", "highlight3"],
  "alternative_route": "suggest a smarter order if the current order is not optimal"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are the world's best multi-city travel planner. Return ONLY valid JSON. No markdown. No backticks. No extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=8000
        )

        plan = clean_json(response.choices[0].message.content)
        return jsonify({'success': True, 'plan': plan})

    except Exception as e:
        print(f"Multi-city plan error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ══════════════════════════════════════════════════════════════
# ── NEW: LOCAL SIM GUIDE ──
# POST /sim-guide
# Body: { destination, origin, days, data_needs, phone_type,
#         budget_conscious, multiple_countries }
# ══════════════════════════════════════════════════════════════
@app.route('/sim-guide', methods=['POST'])
def sim_guide():
    try:
        data = request.get_json()
        destination      = data.get('destination', '')
        origin           = data.get('origin', 'India')
        days             = data.get('days', 7)
        data_needs       = data.get('data_needs', 'moderate')   # light/moderate/heavy
        phone_type       = data.get('phone_type', 'unlocked')   # unlocked/locked/esim
        budget_conscious = data.get('budget_conscious', True)
        countries        = data.get('countries', [destination])  # for multi-country

        countries_str = ', '.join(countries) if isinstance(countries, list) else destination

        prompt = f"""You are an expert in international mobile connectivity and SIM cards.

A traveller from {origin} is visiting {countries_str} for {days} days.
Phone type: {phone_type}
Data usage: {data_needs} (light=social media only, moderate=maps+streaming, heavy=video calls+remote work)
Budget conscious: {budget_conscious}

Give them a COMPLETE, ACCURATE, ACTIONABLE SIM card guide.
Be specific about real carrier names, real prices, real GB amounts.
Include eSIM options where available.

Return ONLY valid JSON:
{{
  "destination": "{destination}",
  "countries_covered": {json.dumps(countries)},
  "days": {days},
  "data_recommendation": "how much data they actually need based on {data_needs} usage for {days} days",
  "top_recommendation": {{
    "name": "specific SIM/plan name",
    "provider": "carrier name",
    "type": "Physical SIM/eSIM/Dual SIM",
    "cost": "exact price in local currency and INR equivalent",
    "data": "exact GB amount",
    "validity": "days valid",
    "calls": "calls included or not",
    "sms": "SMS included",
    "why_best": "specific reason this is best for their situation",
    "where_to_buy": "exact location — airport terminal, convenience store, carrier store",
    "activation": "how to activate step by step",
    "coverage": "4G/5G coverage quality",
    "hotspot": "hotspot/tethering allowed",
    "esim_compatible": true or false,
    "esim_setup": "eSIM setup instructions if available"
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
      "provider": "Airalo/Holafly/Nomad/etc",
      "plan_name": "specific plan name",
      "cost_usd": "USD price",
      "cost_inr": "INR equivalent",
      "data": "GB",
      "validity": "days",
      "buy_link_hint": "app store or website",
      "setup_time": "how long to set up",
      "recommended": true or false,
      "note": "important note"
    }}
  ],
  "airport_buying_guide": {{
    "available_at_airport": true or false,
    "airport_terminal": "which terminal/exit to find SIM vendors",
    "airport_vs_city": "is airport price much higher or same",
    "price_difference": "approximate markup at airport",
    "recommendation": "buy at airport or wait for city",
    "timing": "when to buy — before leaving India or on arrival"
  }},
  "roaming_option": {{
    "worth_it": true or false,
    "airtel_international": "Airtel pack details and cost",
    "jio_international": "Jio pack details and cost",
    "vi_international": "Vi pack details and cost",
    "verdict": "roaming vs local SIM — which wins for this trip"
  }},
  "connectivity_tips": [
    "specific tip 1",
    "specific tip 2",
    "specific tip 3",
    "specific tip 4",
    "specific tip 5"
  ],
  "data_saving_tips": [
    "tip to save data 1",
    "tip to save data 2",
    "tip to save data 3"
  ],
  "offline_essentials": [
    {{"app": "Google Maps", "action": "download {destination} map offline before you leave", "data_saved": "saves ~500MB of data"}},
    {{"app": "app name", "action": "specific offline action", "data_saved": "approximate saving"}}
  ],
  "emergency_connectivity": {{
    "if_sim_fails": "what to do if SIM stops working",
    "free_wifi_spots": "where to find reliable free WiFi in {destination}",
    "emergency_call": "can you call emergency numbers without SIM"
  }},
  "multi_country_tip": "specific advice if travelling to multiple countries",
  "phone_unlock_check": "how to check if phone is unlocked and what to do if locked",
  "budget_summary": {{
    "cheapest_option": "option name and price",
    "best_value": "option name and price",
    "premium_option": "option name and price",
    "recommended_for_this_trip": "option name"
  }}
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert in international SIM cards and mobile connectivity. Always give specific real carrier names and real prices. Return ONLY valid JSON. No markdown."},
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


# ══════════════════════════════════════════════════════════════
# ── NEW: OFFLINE ITINERARY DOWNLOAD ──
# POST /download-itinerary
# Body: { plan, format } — plan = full plan JSON, format = html/pdf
# Returns: downloadable HTML file (works offline)
# ══════════════════════════════════════════════════════════════
@app.route('/download-itinerary', methods=['POST'])
def download_itinerary():
    try:
        data       = request.get_json()
        plan       = data.get('plan', {})
        fmt        = data.get('format', 'html')   # 'html' only for now
        multi_city = data.get('multi_city', False)

        if multi_city:
            html = _build_multi_city_html(plan)
        else:
            html = _build_single_city_html(plan)

        response = make_response(html)
        dest = plan.get('destination') or plan.get('trip_title', 'Trip')
        filename = f"yaply_{dest.replace(' ', '_').lower()}_itinerary.html"
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({'success': False, 'error': str(e)})


def _build_single_city_html(plan):
    """Generate a beautiful self-contained offline HTML itinerary."""
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

    # Build day cards
    day_cards = ''
    for day in itinerary:
        morning   = day.get('morning', {})
        afternoon = day.get('afternoon', {})
        evening   = day.get('evening', {})
        lunch     = day.get('lunch', {})
        dinner    = day.get('dinner', {})
        stay      = day.get('accommodation', {})

        day_cards += f"""
        <div class="day-card">
          <div class="day-header">
            <span class="day-num">Day {day.get('day', '')}</span>
            <span class="day-title">{day.get('title', '')}</span>
          </div>
          <div class="slots">
            {_slot('🌅', 'Morning', morning)}
            {_meal('☀️', 'Lunch', lunch)}
            {_slot('🌞', 'Afternoon', afternoon)}
            {_slot('🌆', 'Evening', evening)}
            {_meal('🌙', 'Dinner', dinner)}
            {_stay(stay)}
          </div>
        </div>"""

    # Packing list
    packing_html = ''.join(f'<span class="tag">{i}</span>' for i in packing)

    # Phrases
    phrase_html = ''
    for p in phrases:
        phrase_html += f"""
        <div class="phrase-row">
          <span class="phrase-en">{p.get('phrase','')}</span>
          <span class="phrase-local">{p.get('translation','')}</span>
          <span class="phrase-pron">({p.get('pronunciation','')})</span>
        </div>"""

    # Hidden gems
    gems_html = ''
    for g in gems:
        gems_html += f"""
        <div class="gem-card">
          <div class="gem-name">💎 {g.get('name','')}</div>
          <div class="gem-desc">{g.get('description','')}</div>
          <div class="gem-meta">📍 {g.get('location','')} · ⏰ {g.get('best_time','')} · 💰 {g.get('cost','')}</div>
        </div>"""

    # Cultural dos/don'ts
    dos   = ''.join(f'<li>✅ {d}</li>' for d in cultural.get('dos', []))
    donts = ''.join(f'<li>❌ {d}</li>' for d in cultural.get('donts', []))

    # Budget tips
    btips_html = ''.join(f'<li>💡 {t}</li>' for t in budget_tips)

    # Tips
    tips_html = ''.join(f'<li>🎯 {t}</li>' for t in tips)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{destination} — Yaply Itinerary</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background:#F5F0E8; color:#1A1612; line-height:1.6; }}
  .header {{ background: linear-gradient(135deg,#0080FF,#0066CC);
             color:white; padding:32px 24px; text-align:center; }}
  .header h1 {{ font-size:32px; font-weight:700; letter-spacing:-1px; }}
  .header .sub {{ opacity:0.8; font-size:14px; margin-top:6px; }}
  .badge {{ display:inline-block; background:rgba(255,255,255,0.2);
            border-radius:20px; padding:4px 12px; font-size:12px;
            margin:8px 4px 0; }}
  .yaply-brand {{ font-size:11px; opacity:0.6; margin-top:12px; letter-spacing:2px; }}
  .container {{ max-width:800px; margin:0 auto; padding:24px 16px; }}
  .section {{ background:white; border-radius:16px; padding:20px;
              margin-bottom:16px; box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
  .section-title {{ font-size:16px; font-weight:700; color:#0080FF;
                    margin-bottom:16px; display:flex; align-items:center; gap:8px; }}
  .day-card {{ background:white; border-radius:16px; padding:20px;
               margin-bottom:12px; border-left:4px solid #0080FF;
               box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
  .day-header {{ display:flex; align-items:center; gap:12px; margin-bottom:16px; }}
  .day-num {{ background:#0080FF; color:white; border-radius:50%;
              width:36px; height:36px; display:flex; align-items:center;
              justify-content:center; font-weight:700; font-size:14px;
              flex-shrink:0; }}
  .day-title {{ font-size:16px; font-weight:600; color:#1A1612; }}
  .slots {{ display:flex; flex-direction:column; gap:10px; }}
  .slot {{ background:#F5F0E8; border-radius:10px; padding:12px; }}
  .slot-header {{ font-size:11px; font-weight:600; color:#6B7280;
                  text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }}
  .slot-activity {{ font-weight:600; font-size:14px; color:#1A1612; }}
  .slot-location {{ font-size:12px; color:#0080FF; margin-top:2px; }}
  .slot-tip {{ font-size:11px; color:#6B7280; margin-top:4px;
               font-style:italic; border-left:2px solid #0080FF;
               padding-left:8px; }}
  .slot-cost {{ font-size:12px; font-weight:600; color:#059669; margin-top:4px; }}
  .meal-slot {{ background:#FFF7ED; border-radius:10px; padding:10px 12px; }}
  .stay-slot {{ background:#EFF6FF; border-radius:10px; padding:10px 12px; }}
  .phrase-row {{ display:flex; gap:12px; align-items:center;
                 padding:10px 0; border-bottom:1px solid #F0EBE0; flex-wrap:wrap; }}
  .phrase-en {{ font-weight:600; min-width:120px; font-size:13px; }}
  .phrase-local {{ color:#0080FF; font-size:14px; font-weight:600; }}
  .phrase-pron {{ color:#6B7280; font-size:12px; font-style:italic; }}
  .gem-card {{ background:#F5F0E8; border-radius:12px; padding:14px;
               margin-bottom:10px; }}
  .gem-name {{ font-weight:700; font-size:14px; margin-bottom:4px; }}
  .gem-desc {{ font-size:12px; color:#3D3730; }}
  .gem-meta {{ font-size:11px; color:#6B7280; margin-top:6px; }}
  .tag {{ display:inline-block; background:#EFF6FF; color:#0080FF;
          border-radius:20px; padding:4px 12px; font-size:12px;
          margin:3px; font-weight:500; }}
  .info-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  .info-item {{ background:#F5F0E8; border-radius:10px; padding:12px; }}
  .info-label {{ font-size:10px; color:#6B7280; text-transform:uppercase;
                 letter-spacing:1px; margin-bottom:4px; }}
  .info-value {{ font-size:14px; font-weight:600; color:#1A1612; }}
  ul {{ padding-left:0; list-style:none; }}
  ul li {{ padding:6px 0; border-bottom:1px solid #F0EBE0;
           font-size:13px; color:#3D3730; }}
  .culture-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .emergency-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
  .emergency-item {{ background:#FEF2F2; border-radius:10px; padding:12px;
                     text-align:center; }}
  .emergency-num {{ font-size:20px; font-weight:700; color:#DC2626; }}
  .emergency-label {{ font-size:11px; color:#6B7280; margin-top:2px; }}
  .footer {{ text-align:center; padding:24px; color:#6B7280; font-size:12px; }}
  .offline-note {{ background:#EFF6FF; border:1px solid #BFDBFE;
                   border-radius:12px; padding:12px 16px; margin-bottom:16px;
                   font-size:12px; color:#1D4ED8; text-align:center; }}
  @media print {{
    .header {{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
    .day-card {{ page-break-inside:avoid; }}
  }}
</style>
</head>
<body>
<div class="header">
  <div class="yaply-brand">YAPLY · AI TRAVEL OS</div>
  <h1>✈️ {destination}</h1>
  <div class="sub">{days}-Day Complete Itinerary</div>
  <div>
    <span class="badge">📅 {days} Days</span>
    <span class="badge">💰 {currency}</span>
    <span class="badge">🌍 Offline Ready</span>
  </div>
</div>

<div class="container">
  <div class="offline-note">
    📱 This file works completely offline — save it to your phone before you travel!
    Open in any browser, no internet needed.
  </div>

  <!-- QUICK INFO -->
  <div class="section">
    <div class="section-title">ℹ️ Trip Essentials</div>
    <div class="info-grid">
      <div class="info-item"><div class="info-label">Best Time to Visit</div>
        <div class="info-value">{plan.get('best_time_to_visit','')}</div></div>
      <div class="info-item"><div class="info-label">Local Language</div>
        <div class="info-value">{plan.get('language','')}</div></div>
      <div class="info-item"><div class="info-label">Timezone</div>
        <div class="info-value">{plan.get('timezone','')}</div></div>
      <div class="info-item"><div class="info-label">Local Currency</div>
        <div class="info-value">{plan.get('currency','')}</div></div>
    </div>
  </div>

  <!-- FLIGHT INFO -->
  {f'''<div class="section">
    <div class="section-title">✈️ Flight Info</div>
    <div class="info-grid">
      <div class="info-item"><div class="info-label">Estimated Cost</div>
        <div class="info-value">{flight.get("estimated_cost","")}</div></div>
      <div class="info-item"><div class="info-label">Duration</div>
        <div class="info-value">{flight.get("flight_duration","")}</div></div>
      <div class="info-item"><div class="info-label">Best Airlines</div>
        <div class="info-value">{", ".join(flight.get("best_airlines",[]))}</div></div>
      <div class="info-item"><div class="info-label">Book In Advance</div>
        <div class="info-value">{flight.get("best_time_to_book","")}</div></div>
    </div>
  </div>''' if flight else ''}

  <!-- VISA INFO -->
  {f'''<div class="section">
    <div class="section-title">🛂 Visa Information</div>
    <div class="info-grid">
      <div class="info-item"><div class="info-label">Visa Required</div>
        <div class="info-value">{"Yes" if visa.get("required") else "No"}</div></div>
      <div class="info-item"><div class="info-label">Visa Type</div>
        <div class="info-value">{visa.get("type","")}</div></div>
      <div class="info-item"><div class="info-label">Cost</div>
        <div class="info-value">{visa.get("cost","")}</div></div>
      <div class="info-item"><div class="info-label">Processing Time</div>
        <div class="info-value">{visa.get("processing_time","")}</div></div>
    </div>
    <div style="margin-top:12px; font-size:13px; color:#3D3730;">
      Apply at: <strong>{visa.get("apply_at","")}</strong>
    </div>
  </div>''' if visa else ''}

  <!-- SIM CARD -->
  {f'''<div class="section">
    <div class="section-title">📱 SIM Card</div>
    <div class="info-grid">
      <div class="info-item"><div class="info-label">Best Option</div>
        <div class="info-value">{sim.get("best_option","")}</div></div>
      <div class="info-item"><div class="info-label">Cost</div>
        <div class="info-value">{sim.get("cost","")}</div></div>
      <div class="info-item"><div class="info-label">Data Included</div>
        <div class="info-value">{sim.get("data","")}</div></div>
      <div class="info-item"><div class="info-label">Where to Buy</div>
        <div class="info-value">{sim.get("where_to_buy","")}</div></div>
    </div>
  </div>''' if sim else ''}

  <!-- ITINERARY -->
  <div class="section-title" style="padding:0 4px; margin-bottom:12px;">
    📅 Day by Day Itinerary
  </div>
  {day_cards}

  <!-- HIDDEN GEMS -->
  {f'''<div class="section">
    <div class="section-title">💎 Hidden Gems</div>
    {gems_html}
  </div>''' if gems_html else ''}

  <!-- LOCAL PHRASES -->
  {f'''<div class="section">
    <div class="section-title">🗣️ Essential Phrases</div>
    {phrase_html}
  </div>''' if phrase_html else ''}

  <!-- CULTURAL GUIDE -->
  {f'''<div class="section">
    <div class="section-title">🌍 Cultural Guide</div>
    <div class="culture-grid">
      <div><strong style="color:#059669;">✅ Do</strong><ul>{dos}</ul></div>
      <div><strong style="color:#DC2626;">❌ Don\'t</strong><ul>{donts}</ul></div>
    </div>
    <div style="margin-top:12px; display:grid; grid-template-columns:1fr 1fr; gap:12px;">
      <div class="info-item"><div class="info-label">Dress Code</div>
        <div class="info-value" style="font-size:12px;">{cultural.get("dress_code","")}</div></div>
      <div class="info-item"><div class="info-label">Tipping</div>
        <div class="info-value" style="font-size:12px;">{cultural.get("tipping","")}</div></div>
    </div>
  </div>''' if cultural else ''}

  <!-- EMERGENCY NUMBERS -->
  {f'''<div class="section">
    <div class="section-title">🆘 Emergency Numbers</div>
    <div class="emergency-grid">
      <div class="emergency-item"><div class="emergency-num">{emergency.get("police","")}</div>
        <div class="emergency-label">Police</div></div>
      <div class="emergency-item"><div class="emergency-num">{emergency.get("ambulance","")}</div>
        <div class="emergency-label">Ambulance</div></div>
      <div class="emergency-item"><div class="emergency-num">{emergency.get("fire","")}</div>
        <div class="emergency-label">Fire</div></div>
      <div class="emergency-item"><div class="emergency-num">{emergency.get("tourist_helpline","")}</div>
        <div class="emergency-label">Tourist Help</div></div>
    </div>
  </div>''' if emergency else ''}

  <!-- PACKING LIST -->
  {f'''<div class="section">
    <div class="section-title">🧳 Packing List</div>
    <div>{packing_html}</div>
  </div>''' if packing_html else ''}

  <!-- BUDGET TIPS -->
  {f'''<div class="section">
    <div class="section-title">💰 Budget Tips</div>
    <ul>{btips_html}</ul>
  </div>''' if btips_html else ''}

  <!-- TIPS -->
  {f'''<div class="section">
    <div class="section-title">🎯 Pro Tips</div>
    <ul>{tips_html}</ul>
  </div>''' if tips_html else ''}

</div>

<div class="footer">
  Generated by <strong>Yaply</strong> — Your Complete Travel OS<br>
  <a href="https://yaply.live" style="color:#0080FF;">yaply.live</a>
  · Works completely offline · Save this file to your phone
</div>
</body>
</html>"""


def _slot(emoji, label, slot):
    if not slot or not slot.get('activity'):
        return ''
    tip_html = f'<div class="slot-tip">💡 {slot.get("tip","")}</div>' if slot.get('tip') else ''
    return f"""
    <div class="slot">
      <div class="slot-header">{emoji} {label}</div>
      <div class="slot-activity">{slot.get('activity','')}</div>
      <div class="slot-location">📍 {slot.get('location','')}</div>
      <div class="slot-cost">💰 {slot.get('cost','')} · ⏱ {slot.get('duration','')}</div>
      {tip_html}
    </div>"""


def _meal(emoji, label, meal):
    if not meal or not meal.get('restaurant'):
        return ''
    return f"""
    <div class="meal-slot">
      <div class="slot-header">{emoji} {label}</div>
      <div class="slot-activity">{meal.get('restaurant','')}</div>
      <div class="slot-location">🍽️ {meal.get('cuisine','')} · 💰 {meal.get('cost','')}</div>
    </div>"""


def _stay(stay):
    if not stay or not stay.get('name'):
        return ''
    return f"""
    <div class="stay-slot">
      <div class="slot-header">🏨 Stay</div>
      <div class="slot-activity">{stay.get('name','')}</div>
      <div class="slot-location">📍 {stay.get('area','')} · 💰 {stay.get('cost','')} / night</div>
    </div>"""


def _build_multi_city_html(plan):
    """Generate offline HTML for a multi-city trip."""
    title      = plan.get('trip_title', 'Multi-City Trip')
    cities     = plan.get('cities', [])
    transits   = plan.get('transit_plans', [])
    sim_strat  = plan.get('sim_strategy', {})
    packing    = plan.get('packing_for_route', {})
    suggestions= plan.get('smart_suggestions', [])
    budget     = plan.get('budget_split', {})
    currency   = plan.get('currency', 'INR')

    cities_html = ''
    for city in cities:
        itinerary = city.get('itinerary', [])
        day_cards = ''
        for day in itinerary:
            morning   = day.get('morning', {})
            afternoon = day.get('afternoon', {})
            evening   = day.get('evening', {})
            lunch     = day.get('lunch', {})
            dinner    = day.get('dinner', {})
            stay      = day.get('accommodation', {})
            day_cards += f"""
            <div class="day-card">
              <div class="day-header">
                <span class="day-num">{day.get('day','')}</span>
                <span class="day-title">{day.get('day_label', day.get('theme',''))}</span>
              </div>
              <div class="slots">
                {_slot('🌅','Morning',morning)}
                {_meal('☀️','Lunch',lunch)}
                {_slot('🌞','Afternoon',afternoon)}
                {_slot('🌆','Evening',evening)}
                {_meal('🌙','Dinner',dinner)}
                {_stay(stay)}
              </div>
            </div>"""

        gems_html = ''
        for g in city.get('hidden_gems', []):
            gems_html += f"""
            <div class="gem-card">
              <div class="gem-name">💎 {g.get('name','')}</div>
              <div class="gem-desc">{g.get('why','')}</div>
              <div class="gem-meta">💰 {g.get('cost','')} · ⏰ {g.get('best_time','')}</div>
            </div>"""

        must_eat = ''.join(f'<span class="tag">🍜 {m}</span>' for m in city.get('must_eat',[]))
        must_do  = ''.join(f'<span class="tag">🎯 {a}</span>' for a in city.get('must_do',[]))
        local_tips = ''.join(f'<li>💡 {t}</li>' for t in city.get('local_tips',[]))

        cities_html += f"""
        <div class="city-section">
          <div class="city-header" style="background:linear-gradient(135deg,#0080FF,#0066CC)">
            <div class="city-num">City {city.get('city_number','')}</div>
            <h2>{city.get('city','')} · {city.get('country','')}</h2>
            <div class="city-meta">
              <span class="badge">{city.get('days','')} Days</span>
              <span class="badge">💰 {currency} {city.get('city_budget','')}</span>
              <span class="badge">🏨 {city.get('best_area_to_stay','')}</span>
            </div>
            <div style="opacity:0.8; font-size:13px; margin-top:8px;">
              {city.get('city_vibe','')}
            </div>
          </div>

          <div class="container">
            <div class="section">
              <div class="section-title">ℹ️ City Essentials</div>
              <div class="info-grid">
                <div class="info-item"><div class="info-label">Language</div>
                  <div class="info-value">{city.get('language','')}</div></div>
                <div class="info-item"><div class="info-label">Currency</div>
                  <div class="info-value">{city.get('local_currency','')}</div></div>
                <div class="info-item"><div class="info-label">Best Stay Area</div>
                  <div class="info-value">{city.get('best_area_to_stay','')}</div></div>
                <div class="info-item"><div class="info-label">Weather</div>
                  <div class="info-value">{city.get('weather_note','')}</div></div>
              </div>
              <div style="margin-top:10px;" class="info-item">
                <div class="info-label">💱 Currency Tip</div>
                <div class="info-value" style="font-size:13px;">{city.get('currency_tip','')}</div>
              </div>
            </div>

            {f'<div class="section"><div class="section-title">🍜 Must Eat</div>{must_eat}</div>' if must_eat else ''}
            {f'<div class="section"><div class="section-title">🎯 Must Do</div>{must_do}</div>' if must_do else ''}

            <div class="section-title" style="padding:0 4px; margin-bottom:12px;">
              📅 Daily Itinerary
            </div>
            {day_cards}

            {f'<div class="section"><div class="section-title">💎 Hidden Gems</div>{gems_html}</div>' if gems_html else ''}

            {f'<div class="section"><div class="section-title">💡 Local Tips</div><ul>{local_tips}</ul></div>' if local_tips else ''}
          </div>
        </div>"""

    # Transit plans
    transit_html = ''
    for t in transits:
        options_html = ''
        for opt in t.get('options', []):
            rec = '⭐ RECOMMENDED' if opt.get('recommended') else ''
            options_html += f"""
            <div class="info-item" style="margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>{opt.get('mode','')} — {opt.get('operator','')}</strong>
                <span style="color:#0080FF; font-size:11px;">{rec}</span>
              </div>
              <div style="font-size:12px; color:#6B7280; margin-top:4px;">
                ⏱ {opt.get('duration','')} · 💰 {opt.get('total_cost','')} · {opt.get('comfort','')}
              </div>
              <div style="font-size:12px; color:#3D3730; margin-top:4px;">
                📌 {opt.get('booking_tip','')}
              </div>
              {'<div style="font-size:11px; color:#059669; margin-top:4px;">✓ ' + opt.get('reason','') + '</div>' if opt.get('reason') else ''}
            </div>"""
        transit_html += f"""
        <div class="section">
          <div class="section-title">🚆 {t.get('from','')} → {t.get('to','')}</div>
          <div style="font-size:12px; color:#6B7280; margin-bottom:12px;">Day {t.get('transit_day','')}</div>
          {options_html}
          <div style="background:#EFF6FF; border-radius:8px; padding:10px; margin-top:8px; font-size:12px; color:#1D4ED8;">
            💡 {t.get('transit_tip','')}
          </div>
        </div>"""

    # Smart suggestions
    suggestions_html = ''.join(f'<li>🧠 {s}</li>' for s in suggestions)

    # Budget split
    budget_html = ''
    for k, v in budget.items():
        budget_html += f'<div class="info-item"><div class="info-label">{k.replace("_"," ").title()}</div><div class="info-value">{currency} {v}</div></div>'

    # SIM strategy
    sim_options_html = ''
    for opt in sim_strat.get('options', []):
        rec = '⭐ RECOMMENDED' if opt.get('recommended') else ''
        sim_options_html += f"""
        <div class="info-item" style="margin-bottom:8px;">
          <div style="display:flex; justify-content:space-between;">
            <strong>{opt.get('type','')}</strong>
            <span style="color:#0080FF; font-size:11px;">{rec}</span>
          </div>
          <div style="font-size:12px; color:#6B7280;">{', '.join(opt.get('providers',[]))}</div>
          <div style="font-size:12px; color:#059669; margin-top:4px;">💰 {opt.get('cost','')} · 📶 {opt.get('coverage','')}</div>
          <div style="font-size:12px; color:#3D3730; margin-top:4px;">{opt.get('why','')}</div>
        </div>"""

    packing_items = ''.join(f'<span class="tag">{i}</span>' for i in packing.get('key_items', []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Yaply Multi-City Itinerary</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background:#F5F0E8; color:#1A1612; line-height:1.6; }}
  .main-header {{ background:linear-gradient(135deg,#0080FF,#003d99);
                  color:white; padding:32px 24px; text-align:center; }}
  .main-header h1 {{ font-size:28px; font-weight:700; letter-spacing:-1px; }}
  .city-header {{ color:white; padding:24px; text-align:center; }}
  .city-header h2 {{ font-size:24px; font-weight:700; }}
  .city-num {{ font-size:11px; opacity:0.7; letter-spacing:2px;
               text-transform:uppercase; margin-bottom:4px; }}
  .city-section {{ margin-bottom:8px; }}
  .badge {{ display:inline-block; background:rgba(255,255,255,0.2);
            border-radius:20px; padding:4px 12px; font-size:11px;
            margin:4px 3px 0; }}
  .container {{ max-width:800px; margin:0 auto; padding:20px 16px; }}
  .section {{ background:white; border-radius:16px; padding:20px;
              margin-bottom:12px; box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
  .section-title {{ font-size:16px; font-weight:700; color:#0080FF;
                    margin-bottom:16px; }}
  .day-card {{ background:white; border-radius:16px; padding:20px;
               margin-bottom:12px; border-left:4px solid #0080FF;
               box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
  .day-header {{ display:flex; align-items:center; gap:12px; margin-bottom:16px; }}
  .day-num {{ background:#0080FF; color:white; border-radius:50%;
              width:36px; height:36px; display:flex; align-items:center;
              justify-content:center; font-weight:700; font-size:13px;
              flex-shrink:0; }}
  .day-title {{ font-size:15px; font-weight:600; }}
  .slots {{ display:flex; flex-direction:column; gap:10px; }}
  .slot {{ background:#F5F0E8; border-radius:10px; padding:12px; }}
  .slot-header {{ font-size:10px; font-weight:600; color:#6B7280;
                  text-transform:uppercase; letter-spacing:1px; margin-bottom:3px; }}
  .slot-activity {{ font-weight:600; font-size:13px; }}
  .slot-location {{ font-size:12px; color:#0080FF; margin-top:2px; }}
  .slot-tip {{ font-size:11px; color:#6B7280; margin-top:4px;
               font-style:italic; border-left:2px solid #0080FF; padding-left:8px; }}
  .slot-cost {{ font-size:12px; font-weight:600; color:#059669; margin-top:4px; }}
  .meal-slot {{ background:#FFF7ED; border-radius:10px; padding:10px 12px; }}
  .stay-slot {{ background:#EFF6FF; border-radius:10px; padding:10px 12px; }}
  .gem-card {{ background:#F5F0E8; border-radius:12px; padding:12px;
               margin-bottom:8px; }}
  .gem-name {{ font-weight:700; font-size:13px; margin-bottom:3px; }}
  .gem-desc {{ font-size:12px; color:#3D3730; }}
  .gem-meta {{ font-size:11px; color:#6B7280; margin-top:4px; }}
  .tag {{ display:inline-block; background:#EFF6FF; color:#0080FF;
          border-radius:20px; padding:4px 12px; font-size:12px;
          margin:3px; font-weight:500; }}
  .info-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
  .info-item {{ background:#F5F0E8; border-radius:10px; padding:12px; }}
  .info-label {{ font-size:10px; color:#6B7280; text-transform:uppercase;
                 letter-spacing:1px; margin-bottom:3px; }}
  .info-value {{ font-size:13px; font-weight:600; color:#1A1612; }}
  ul {{ padding-left:0; list-style:none; }}
  ul li {{ padding:6px 0; border-bottom:1px solid #F0EBE0; font-size:13px; }}
  .offline-note {{ background:#EFF6FF; border:1px solid #BFDBFE;
                   border-radius:12px; padding:12px 16px; margin-bottom:16px;
                   font-size:12px; color:#1D4ED8; text-align:center; }}
  .footer {{ text-align:center; padding:24px; color:#6B7280; font-size:12px; }}
  .yaply-brand {{ font-size:11px; opacity:0.6; letter-spacing:2px; margin-bottom:4px; }}
</style>
</head>
<body>

<div class="main-header">
  <div class="yaply-brand">YAPLY · AI TRAVEL OS</div>
  <h1>🗺️ {title}</h1>
  <div style="opacity:0.8; font-size:13px; margin-top:6px;">
    {plan.get('total_days','')} Days · {len(cities)} Cities · {currency} {plan.get('total_budget','')}
  </div>
  <div>
    {''.join(f'<span class="badge">{c.get("city","")}</span>' for c in cities)}
  </div>
</div>

<div class="container">
  <div class="offline-note">
    📱 This file works completely offline — save it to your phone before you travel!
  </div>

  <!-- SMART SUGGESTIONS -->
  {f'''<div class="section">
    <div class="section-title">🧠 Smart Suggestions for This Route</div>
    <ul>{suggestions_html}</ul>
  </div>''' if suggestions_html else ''}

  <!-- BUDGET SPLIT -->
  {f'''<div class="section">
    <div class="section-title">💰 Budget Split Across Trip</div>
    <div class="info-grid">{budget_html}</div>
  </div>''' if budget_html else ''}

  <!-- TRANSIT PLANS -->
  {f'''<div class="section">
    <div class="section-title">🚆 Transit Between Cities</div>
    {transit_html}
  </div>''' if transit_html else ''}

  <!-- SIM STRATEGY -->
  {f'''<div class="section">
    <div class="section-title">📱 SIM & Connectivity Strategy</div>
    <div style="background:#EFF6FF; border-radius:8px; padding:12px; margin-bottom:12px; font-size:13px; color:#1D4ED8;">
      💡 {sim_strat.get("recommendation","")}
    </div>
    {sim_options_html}
  </div>''' if sim_strat else ''}

  <!-- PACKING FOR ROUTE -->
  {f'''<div class="section">
    <div class="section-title">🧳 Packing for This Route</div>
    <div class="info-item" style="margin-bottom:10px;">
      <div class="info-label">Weather Variation</div>
      <div class="info-value" style="font-size:13px;">{packing.get("weather_variation","")}</div>
    </div>
    <div style="margin-bottom:10px;">{packing_items}</div>
    <div class="info-item">
      <div class="info-label">Luggage Tip</div>
      <div class="info-value" style="font-size:13px;">{packing.get("luggage_tip","")}</div>
    </div>
  </div>''' if packing else ''}

</div>

<!-- CITY BY CITY -->
{cities_html}

<div class="footer">
  Generated by <strong>Yaply</strong> — Your Complete Travel OS<br>
  <a href="https://yaply.live" style="color:#0080FF;">yaply.live</a>
  · Works completely offline
</div>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════
# ALL EXISTING ROUTES BELOW — unchanged
# ══════════════════════════════════════════════════════════════

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
        daily = {}
        for item in weather_data['list']:
            date = item['dt_txt'].split(' ')[0]
            if date not in daily:
                daily[date] = {'date': date, 'temp_max': item['main']['temp_max'],
                               'temp_min': item['main']['temp_min'],
                               'description': item['weather'][0]['description'],
                               'icon': item['weather'][0]['icon'],
                               'humidity': item['main']['humidity'],
                               'wind': item['wind']['speed']}
            else:
                daily[date]['temp_max'] = max(daily[date]['temp_max'], item['main']['temp_max'])
                daily[date]['temp_min'] = min(daily[date]['temp_min'], item['main']['temp_min'])
        forecast = list(daily.values())[:7]
        return jsonify({'success': True, 'city': weather_data['city']['name'],
                        'country': weather_data['city']['country'], 'forecast': forecast})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


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
        return jsonify({'success': True, 'from': from_curr, 'to': to_curr,
                        'amount': amount, 'converted': round(result['conversion_result'], 2),
                        'rate': result['conversion_rate']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/visa', methods=['POST'])
def check_visa():
    try:
        data = request.get_json()
        passport = data.get('passport', 'India')
        destination = data.get('destination', '')
        prompt = f"""A traveller with a {passport} passport wants to visit {destination}.
Return ONLY a JSON object:
{{"visa_required": true or false, "visa_type": "type", "validity": "how long",
  "cost": "cost in USD", "processing_days": "days", "apply_online": true,
  "apply_url": "url", "documents": ["doc1","doc2"], "tips": ["tip1"],
  "visa_on_arrival": true, "visa_free_days": 0}}"""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return valid JSON only. No markdown."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=800)
        return jsonify({'success': True, 'visa': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


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
{{"essentials":["i1","i2"],"clothing":["i1"],"toiletries":["i1"],
  "electronics":["i1"],"documents":["i1"],"health":["i1"],
  "destination_specific":["i1"]}}"""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return valid JSON only."},
                      {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1000)
        return jsonify({'success': True, 'packing': clean_json(response.choices[0].message.content)})
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
        prompt = f"""Expert travel logistics planner.
From "{origin}" to "{destination}". Mode: {travel_mode}. Prices in {currency}.
Return ONLY valid JSON with: origin, destination, nearest_airports,
destination_airports, recommended_route, flight_options,
alternative_routes, important_notes, documents_needed."""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Expert travel logistics. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=3000)
        return jsonify({'success': True, 'journey': clean_json(response.choices[0].message.content)})
    except Exception as e:
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
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": "Identify this place. Return ONLY valid JSON with: place_name, city, region, country, continent, confidence, place_type, description, tags, best_time, climate, budget_level, avg_daily_cost, language, currency, nearest_airport, airport_code, why_famous, instagram_spots, nearby, similar_places, travel_tips, best_food"}
            ]}],
            temperature=0.1, max_tokens=2000)
        return jsonify({'success': True, 'result': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/identify-text', methods=['POST'])
def identify_from_text():
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        if not description:
            return jsonify({'success': False, 'error': 'No description provided'})
        prompt = f'Identify this place: "{description}". Return ONLY valid JSON with full place details.'
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "World geography expert. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=2000)
        return jsonify({'success': True, 'result': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/passport-check', methods=['POST'])
def passport_check():
    try:
        data = request.get_json()
        from datetime import datetime
        expiry = datetime.strptime(data.get('expiry_date',''), '%Y-%m-%d')
        travel = datetime.strptime(data.get('travel_date',''), '%Y-%m-%d')
        today  = datetime.now()
        days_until_expiry    = (expiry - today).days
        days_valid_after_travel = (expiry - travel).days
        prompt = f"""Passport validity check for {data.get('destination','')}.
Expiry: {data.get('expiry_date')} | Travel: {data.get('travel_date')}
Days valid after travel: {days_valid_after_travel}
Return ONLY JSON: is_valid, validity_status, days_remaining({days_until_expiry}),
days_after_travel({days_valid_after_travel}), destination_requirement, verdict,
action_needed, renewal_urgency, renewal_time, renewal_cost,
tatkal_available, tatkal_time, tatkal_cost, tips"""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=800)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/safety-check', methods=['POST'])
def safety_check():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        prompt = f"Comprehensive safety information for {destination}. Return ONLY valid JSON with all safety fields."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Travel safety expert. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1200)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/local-laws', methods=['POST'])
def local_laws():
    try:
        data = request.get_json()
        destination = data.get('destination', '')
        prompt = f"Important laws for tourists in {destination}. Return ONLY valid JSON with strict_laws, photography_rules, dress_code_rules, alcohol_rules, drug_laws, customs_limits, good_to_know, legal_tip."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Legal travel expert. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1500)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/jetlag', methods=['POST'])
def jetlag_calc():
    try:
        data = request.get_json()
        prompt = f"Jet lag calculator from {data.get('from_city','')} to {data.get('to_city','')} on {data.get('travel_date','')}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1200)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/festivals', methods=['POST'])
def festivals():
    try:
        data = request.get_json()
        prompt = f"Festivals and events in {data.get('destination','')} around {data.get('travel_date','')}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1200)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/budget-plan', methods=['POST'])
def budget_plan():
    try:
        data = request.get_json()
        prompt = f"Budget plan for {data.get('people',1)} people in {data.get('destination','')} for {data.get('days',5)} days, budget {data.get('currency','USD')} {data.get('budget',1000)}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1500)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/luggage-check', methods=['POST'])
def luggage_check():
    try:
        data = request.get_json()
        prompt = f"Luggage allowance for {data.get('airline','')} to {data.get('destination','')} in {data.get('cabin_class','Economy')} class. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1000)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/emergency-card', methods=['POST'])
def emergency_card():
    try:
        data = request.get_json()
        prompt = f"Emergency card for {data.get('name','')} visiting {data.get('destination','')}. Blood: {data.get('blood_group','')}. Allergies: {data.get('allergies','')}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1500)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
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
        prompt = f"Medical translator. Symptoms: {data.get('symptoms','')}. Destination: {data.get('destination','')}. Translate to: {data.get('language','Japanese')}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Medical translator. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=2000)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/price-check', methods=['POST'])
def price_check():
    try:
        data = request.get_json()
        prompt = f"Price check in {data.get('destination','')}. Item: {data.get('item','')}. Price charged: {data.get('currency','')} {data.get('price','')}. Fair? Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Local price expert. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=1500)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/scam-alerts', methods=['POST'])
def scam_alerts():
    try:
        data = request.get_json()
        prompt = f"All common tourist scams in {data.get('destination','')} with detailed avoid guides. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Scam expert. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=3000)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/allergy-card', methods=['POST'])
def allergy_card():
    try:
        data = request.get_json()
        allergies = data.get('allergies', [])
        prompt = f"Allergy safety guide for {data.get('name','')} with allergies {', '.join(allergies)} visiting {data.get('destination','')}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Allergy travel expert. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=2000)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/flight-rights', methods=['POST'])
def flight_rights():
    try:
        data = request.get_json()
        prompt = f"Flight rights for {data.get('airline','')} on route {data.get('route','')}. Issue: {data.get('issue','')}. Delay: {data.get('delay_hours',0)}h. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Aviation rights expert. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=2000)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/safe-route', methods=['POST'])
def safe_route():
    try:
        data = request.get_json()
        prompt = f"Safe route guide in {data.get('destination','')} from {data.get('from_location','')} to {data.get('to_location','')} at {data.get('time_of_day','evening')} for {data.get('traveller_type','solo female')}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Safety expert. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=2000)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/immigration-help', methods=['POST'])
def immigration_help():
    try:
        data = request.get_json()
        prompt = f"Immigration guide for {data.get('passport','India')} passport to {data.get('destination','')} for {data.get('purpose','Tourism')}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Immigration expert. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=2000)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
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
        res = requests.get("https://api.unsplash.com/search/photos",
            params={'query': f"{place_name} travel", 'per_page': 5,
                    'orientation': 'landscape', 'client_id': UNSPLASH_KEY})
        results = res.json().get('results', [])
        photos = [r['urls']['regular'] for r in results if r.get('urls', {}).get('regular')]
        if not photos:
            return jsonify({'success': False, 'error': 'No photos'})
        return jsonify({'success': True, 'photos': photos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/detect-theme', methods=['POST'])
def detect_theme():
    try:
        data = request.get_json()
        prompt = f"Visual theme for travel app destination: {data.get('destination','')}. Return ONLY valid JSON with destination_type, theme(primary_color, secondary_color, accent_color, gradient_start, gradient_end, mood, emoji), color_rationale, vibe_words."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=600)
        return jsonify({'success': True, 'theme': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/after')
def after_page():
    return render_template('after_trip.html')


@app.route('/trip-journal', methods=['POST'])
def trip_journal():
    try:
        data = request.get_json()
        destination  = data.get('destination', '')
        days         = data.get('days', 5)
        highlights   = data.get('highlights', '')
        vibe         = data.get('vibe', '')
        travel_with  = data.get('travel_with', 'solo')
        itinerary    = data.get('itinerary', [])
        people       = data.get('people', 1)
        origin       = data.get('origin', '')

        itin_lines = []
        for day in itinerary:
            itin_lines.append(f"Day {day.get('day','?')} — {day.get('title','')}")
            for slot in ['morning','afternoon','evening']:
                s = day.get(slot,{})
                if s and s.get('activity'):
                    itin_lines.append(f"  {slot.capitalize()}: {s.get('activity','')} at {s.get('location','')}")
            for meal in ['lunch','dinner']:
                m = day.get(meal,{})
                if m and m.get('restaurant'):
                    itin_lines.append(f"  {meal.capitalize()}: {m.get('restaurant','')} ({m.get('cuisine','')})")

        num_chapters = len(itinerary) if itinerary else int(days)
        prompt = f"""Professional travel writer. Write a vivid personal journal.
Destination: {destination} | {days} days | {travel_with} | {vibe}
{f'From: {origin}' if origin else ''}
Itinerary: {chr(10).join(itin_lines)}
Highlights: {highlights}
Return ONLY valid JSON with {num_chapters} chapters: title, tagline, opening,
chapters(day,title,story,highlight,emotion,emoji), closing, best_memory,
lesson_learned, quote, would_return, rating, tags"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Professional travel writer. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.72, max_tokens=4000)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/expense-summary', methods=['POST'])
def expense_summary():
    try:
        data       = request.get_json()
        destination = data.get('destination', '')
        expenses   = data.get('expenses', [])
        budget     = data.get('budget', 0)
        currency   = data.get('currency', 'INR')
        people     = data.get('people', 1)
        total      = sum(float(e.get('amount', 0)) for e in expenses)
        by_category = {}
        for e in expenses:
            cat = e.get('category', 'Other')
            by_category[cat] = by_category.get(cat, 0) + float(e.get('amount', 0))
        prompt = f"Expense analysis for {destination}. Budget: {currency} {budget}. Spent: {currency} {total:.0f}. By category: {json.dumps(by_category)}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=800)
        summary = clean_json(response.choices[0].message.content)
        summary['by_category'] = by_category
        summary['expenses'] = expenses
        return jsonify({'success': True, 'data': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/split-bill', methods=['POST'])
def split_bill():
    try:
        data     = request.get_json()
        people   = data.get('people', [])
        expenses = data.get('expenses', [])
        currency = data.get('currency', 'INR')
        balances = {p: 0 for p in people}
        total    = 0
        for exp in expenses:
            amount = float(exp.get('amount', 0))
            paid_by = exp.get('paid_by', people[0] if people else 'Unknown')
            split_between = exp.get('split_between', people) or people
            total += amount
            share = amount / len(split_between)
            if paid_by in balances: balances[paid_by] += amount
            for p in split_between:
                if p in balances: balances[p] -= share
        settlements = []
        pos_list = sorted([(k,v) for k,v in balances.items() if v > 0.01], key=lambda x:-x[1])
        neg_list = sorted([(k,v) for k,v in balances.items() if v < -0.01], key=lambda x:x[1])
        i = j = 0
        while i < len(pos_list) and j < len(neg_list):
            creditor, credit = pos_list[i]
            debtor, debt = neg_list[j]
            amount = min(credit, -debt)
            if amount > 0.01:
                settlements.append({'from': debtor, 'to': creditor,
                                     'amount': round(amount,2), 'currency': currency})
            pos_list[i] = (creditor, credit - amount)
            neg_list[j] = (debtor, debt + amount)
            if pos_list[i][1] < 0.01: i += 1
            if neg_list[j][1] > -0.01: j += 1
        return jsonify({'success': True, 'data': {
            'total': round(total,2), 'per_person': round(total/max(len(people),1),2),
            'balances': {k: round(v,2) for k,v in balances.items()},
            'settlements': settlements, 'currency': currency,
            'all_settled': len(settlements) == 0}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/trip-stats', methods=['POST'])
def trip_stats():
    try:
        data      = request.get_json()
        destination = data.get('destination','')
        days      = data.get('days',5)
        expenses  = data.get('expenses',[])
        travel_with = data.get('travel_with','solo')
        itinerary = data.get('itinerary',[])
        budget    = data.get('budget','')
        origin    = data.get('origin','')
        currency  = data.get('currency','INR')

        actual_places = []
        actual_meals  = []
        actual_activities = []
        for day in itinerary:
            for slot in ['morning','afternoon','evening']:
                s = day.get(slot,{})
                if s and s.get('location'): actual_places.append(s['location'])
                if s and s.get('activity'): actual_activities.append(s['activity'])
            for meal in ['lunch','dinner']:
                m = day.get(meal,{})
                if m and m.get('restaurant'): actual_meals.append(m['restaurant'])

        unique_places = list(dict.fromkeys(actual_places))
        total_spent   = sum(float(e.get('amount',0)) for e in expenses)

        prompt = f"""Fun viral travel stats. Destination: {destination} | {days} days | {travel_with}
Places: {len(unique_places)} | Activities: {len(actual_activities)} | Meals: {len(actual_meals)}
Return ONLY valid JSON with traveller_type, traveller_description, fun_stats,
achievements, travel_score, next_destination_prediction, instagram_caption."""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=2000)
        parsed = clean_json(response.choices[0].message.content)
        parsed['real_counts'] = {'places': len(unique_places),
                                  'activities': len(actual_activities),
                                  'meals': len(actual_meals), 'days': int(days)}
        return jsonify({'success': True, 'data': parsed})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/review-generator', methods=['POST'])
def review_generator():
    try:
        data = request.get_json()
        prompt = f"Write a genuine {data.get('platform','Google')} review for {data.get('place','')}. Rating: {data.get('rating',5)}/5. Experience: {data.get('experience','')}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Genuine review writer. Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.6, max_tokens=1000)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/next-trip', methods=['POST'])
def next_trip():
    try:
        data = request.get_json()
        prompt = f"Next trip suggestions based on past trip to {data.get('past_destination','')}. Loved: {data.get('loved','')}. Budget: {data.get('budget','')}. Month: {data.get('travel_month','')}. Passport: {data.get('passport','India')}. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=1500)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/currency-leftover', methods=['POST'])
def currency_leftover():
    try:
        data = request.get_json()
        prompt = f"What to do with leftover {data.get('currency','')} {data.get('amount',0)} when returning to {data.get('home_currency','INR')} home. Return ONLY valid JSON."
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1000)
        return jsonify({'success': True, 'data': clean_json(response.choices[0].message.content)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/')
def index():
    return render_template('before_trip.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    app.run(debug=True, host='0.0.0.0', port=port)