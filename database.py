"""
Yaply — Production Database Layer
Supabase PostgreSQL — replaces SQLite database.py
"""
import os, json
from datetime import datetime, date, timedelta
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════
# CONNECTION POOL
# ══════════════════════════════════════════════════════════

DATABASE_URL = os.getenv('DATABASE_URL')  # Supabase connection string

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=20,
            dsn=DATABASE_URL,
            cursor_factory=psycopg2.extras.RealDictCursor
        )
    return _pool

def get_db():
    return get_pool().getconn()

def release_db(conn):
    get_pool().putconn(conn)

class db_conn:
    """Context manager for DB connections"""
    def __enter__(self):
        self.conn = get_db()
        self.conn.autocommit = False
        return self.conn
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.conn.rollback()
        else:
            self.conn.commit()
        release_db(self.conn)

def query_one(sql, params=None):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        row = cur.fetchone()
        return dict(row) if row else None

def query_all(sql, params=None):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        return [dict(r) for r in cur.fetchall()]

def execute(sql, params=None):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        try:
            return dict(cur.fetchone()) if cur.description else None
        except:
            return None

# ══════════════════════════════════════════════════════════
# PLAN LIMITS
# ══════════════════════════════════════════════════════════

FREE_LIMITS = {
    # Daily
    'plans_day':          1,
    'translations_day':   10,
    'voice_day':          5,
    'camera_scan_day':    3,
    'identify_photo_day': 3,
    'identify_text_day':  5,
    'price_check_day':    5,
    'packing_day':        2,
    'budget_day':         1,
    'trip_review_day':    1,

    # Monthly
    'plans_month':        3,
    'multicity_month':    0,
    'door_to_door_month': 0,
    'ai_journal_month':   0,
    'ai_story_month':     0,
    'next_trips_month':   0,
    'export_month':       0,
    'trip_card_month':    0,

    # Hard caps
    'diary_entries_trip': 10,
    'saved_places':       5,
    'trip_history':       3,
    'diary_trips':        3,
    'photos_per_entry':   2,
}

PRO_LIMITS = {
    # Daily
    'plans_day':          5,
    'translations_day':   80,
    'voice_day':          40,
    'camera_scan_day':    20,
    'identify_photo_day': 15,
    'identify_text_day':  20,
    'price_check_day':    30,
    'packing_day':        5,
    'budget_day':         5,
    'trip_review_day':    5,

    # Monthly
    'plans_month':        15,
    'multicity_month':    4,
    'door_to_door_month': 8,
    'ai_journal_month':   5,
    'ai_story_month':     2,   # per day above, 5 per month total
    'next_trips_month':   10,
    'export_month':       10,
    'trip_card_month':    10,

    # Hard caps
    'diary_entries_trip': 100,
    'saved_places':       100,
    'trip_history':       50,
    'diary_trips':        30,
    'photos_per_entry':   8,
}

# -- usage track -- #

def get_user_usage(user_id):
    """Returns current usage counts for all tracked features."""
    user = query_one(
        """SELECT
           plan_type, is_pro, pro_expires_at, pro_plan,
           onboarding_done,
           -- daily
           translations_today, voice_today, identify_today, tools_today,
           usage_reset_date,
           -- monthly
           plans_used_month, multicity_used_month, ai_story_used,
           journal_used_month, monthly_reset_date
           FROM users WHERE id=%s AND deleted_at IS NULL""",
        (user_id,)
    )
    if not user:
        return None
 
    from datetime import datetime, date
    today = date.today()
    plan  = user.get('plan_type', 'free')
 
    # Auto-expire pro
    expires = user.get('pro_expires_at')
    if plan == 'pro' and expires:
        if hasattr(expires, 'date'):
            expired = expires.date() < today
        else:
            expired = str(expires)[:10] < str(today)
        if expired:
            execute(
                "UPDATE users SET plan_type='free', is_pro=FALSE WHERE id=%s",
                (user_id,)
            )
            plan = 'free'
 
    limits = PRO_LIMITS if plan == 'pro' else FREE_LIMITS
 
    # Days remaining for pro
    days_remaining = 0
    if plan == 'pro' and expires:
        try:
            if hasattr(expires, 'date'):
                diff = (expires.date() - today).days
            else:
                from datetime import datetime
                exp_date = datetime.fromisoformat(str(expires)).date()
                diff = (exp_date - today).days
            days_remaining = max(0, diff)
        except:
            days_remaining = 0
 
    return {
        'plan':           plan,
        'is_pro':         plan == 'pro',
        'pro_expires_at': str(expires) if expires else None,
        'days_remaining': days_remaining,
        'onboarding_done': bool(user.get('onboarding_done')),
        'limits':         limits,
        'usage': {
            'translations_today':  user.get('translations_today', 0) or 0,
            'voice_today':         user.get('voice_today', 0) or 0,
            'identify_today':      user.get('identify_today', 0) or 0,
            'tools_today':         user.get('tools_today', 0) or 0,
            'plans_used_month':    user.get('plans_used_month', 0) or 0,
            'multicity_used_month':user.get('multicity_used_month', 0) or 0,
            'ai_story_used':       user.get('ai_story_used', 0) or 0,
            'journal_used_month':  user.get('journal_used_month', 0) or 0,
        }
    }

# ══════════════════════════════════════════════════════════
# USERS
# ══════════════════════════════════════════════════════════

def create_user(email, name, password_hash=None, google_id=None, avatar=None):
    try:
        row = execute(
            """INSERT INTO users
               (email, name, password_hash, google_id, avatar, email_verified)
               VALUES (%s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (email.lower().strip(), name.strip(), password_hash,
             google_id, avatar or '', bool(google_id))
        )
        return row['id'] if row else None
    except psycopg2.IntegrityError:
        return None

def get_user_by_email(email):
    return query_one(
        'SELECT * FROM users WHERE email=%s AND deleted_at IS NULL',
        (email.lower().strip(),)
    )

def get_user_by_id(user_id):
    return query_one(
        'SELECT * FROM users WHERE id=%s AND deleted_at IS NULL',
        (user_id,)
    )

def get_user_by_google(google_id):
    return query_one(
        'SELECT * FROM users WHERE google_id=%s AND deleted_at IS NULL',
        (google_id,)
    )

def update_user(user_id, **fields):
    allowed = [
        'name', 'avatar', 'passport', 'home_city', 'currency',
        'last_login_at', 'last_active_at', 'is_verified', 'is_pro',
        'plan_type', 'pro_expires_at', 'travel_style', 'budget_style',
        'onboarding_done', 'email_verified', 'phone', 'password_hash'
    ]
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    sets = ', '.join(f'{k}=%s' for k in updates)
    execute(
        f'UPDATE users SET {sets} WHERE id=%s',
        (*updates.values(), user_id)
    )

def get_user_stats(user_id):
    row = query_one(
        """SELECT
           (SELECT COUNT(*) FROM trips WHERE user_id=%s AND deleted_at IS NULL) AS trips,
           (SELECT COALESCE(SUM(days),0) FROM trips WHERE user_id=%s AND deleted_at IS NULL) AS days,
           (SELECT COUNT(*) FROM saved_places WHERE user_id=%s AND deleted_at IS NULL) AS places
        """,
        (user_id, user_id, user_id)
    )
    return row or {'trips': 0, 'days': 0, 'places': 0}

def safe_user(user):
    if not user:
        return None
    u = dict(user)
    u.pop('password_hash', None)
    u.pop('email_verify_token', None)
    # Convert datetime objects to ISO strings
    for k, v in u.items():
        if hasattr(v, 'isoformat'):
            u[k] = v.isoformat()
    return u

# ══════════════════════════════════════════════════════════
# SUBSCRIPTION
# ══════════════════════════════════════════════════════════

def get_user_plan(user_id):
    """Returns 'free' or 'pro', auto-expires if needed"""
    user = query_one(
        'SELECT plan_type, pro_expires_at FROM users WHERE id=%s',
        (user_id,)
    )
    if not user:
        return 'free'
    plan = user.get('plan_type', 'free')
    expires = user.get('pro_expires_at')
    if plan == 'pro' and expires and expires < datetime.now(expires.tzinfo):
        execute(
            "UPDATE users SET plan_type='free', is_pro=FALSE WHERE id=%s",
            (user_id,)
        )
        return 'free'
    return plan

def activate_pro(user_id, plan='monthly'):
    days = 7 if plan == 'weekly' else 30
    expires = datetime.utcnow() + timedelta(days=days)
    execute(
        """UPDATE users SET
           plan_type='pro', is_pro=TRUE,
           pro_expires_at=%s, pro_plan=%s
           WHERE id=%s""",
        (expires, plan, user_id)
    )
    return expires

def check_limit(user_id, feature):
    """Returns (allowed, used, limit)"""
    plan = get_user_plan(user_id)
    limits = PRO_LIMITS if plan == 'pro' else FREE_LIMITS

    col_map = {
        'plan':        ('plans_used_month',     limits['plans_month']),
        'multicity':   ('multicity_used_month', limits['multicity_month']),
        'translation': ('translations_today',   limits['translations_day']),
        'voice':       ('voice_today',          limits['voice_day']),
        'identify':    ('identify_today',       limits['identify_day']),
        'tool':        ('tools_today',          limits['tools_day']),
        'ai_story':    ('ai_story_used',        limits['ai_story']),
        'journal':     ('journal_used_month',   limits['journal_month']),
    }

    if feature not in col_map:
        return True, 0, 999

    col, limit = col_map[feature]
    if limit == 0:
        return False, 0, 0

    user = query_one(f'SELECT {col} FROM users WHERE id=%s', (user_id,))
    used = (user or {}).get(col, 0) or 0
    return used < limit, used, limit

def increment_usage(user_id, feature):
    col_map = {
        'plan':        'plans_used_month',
        'multicity':   'multicity_used_month',
        'translation': 'translations_today',
        'voice':       'voice_today',
        'identify':    'identify_today',
        'tool':        'tools_today',
        'ai_story':    'ai_story_used',
        'journal':     'journal_used_month',
    }
    col = col_map.get(feature)
    if col:
        execute(f'UPDATE users SET {col}={col}+1 WHERE id=%s', (user_id,))

def complete_onboarding(user_id, name, home_city='', passport='India',
                         currency='INR', travel_style='', budget_style=''):
    execute(
        """UPDATE users SET
           name=%s, home_city=%s, passport=%s, currency=%s,
           travel_style=%s, budget_style=%s, onboarding_done=TRUE
           WHERE id=%s""",
        (name.strip(), home_city.strip(), passport,
         currency, travel_style, budget_style, user_id)
    )

# ══════════════════════════════════════════════════════════
# SESSIONS (refresh tokens)
# ══════════════════════════════════════════════════════════

def create_session(user_id, token_hash, refresh_token, device_info='',
                   ip_address='', user_agent='', expires_at=None):
    if not expires_at:
        expires_at = datetime.utcnow() + timedelta(days=30)
    row = execute(
        """INSERT INTO user_sessions
           (user_id, token_hash, refresh_token, device_info, ip_address, user_agent, expires_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           RETURNING id""",
        (user_id, token_hash, refresh_token, device_info, ip_address, user_agent, expires_at)
    )
    return row['id'] if row else None

def invalidate_session(token_hash):
    execute(
        'UPDATE user_sessions SET is_active=FALSE WHERE token_hash=%s',
        (token_hash,)
    )

def invalidate_all_sessions(user_id):
    execute(
        'UPDATE user_sessions SET is_active=FALSE WHERE user_id=%s',
        (user_id,)
    )

def get_session_by_refresh(refresh_token):
    return query_one(
        """SELECT * FROM user_sessions
           WHERE refresh_token=%s AND is_active=TRUE AND expires_at > NOW()""",
        (refresh_token,)
    )

# ══════════════════════════════════════════════════════════
# PAYMENTS
# ══════════════════════════════════════════════════════════

def create_payment(user_id, razorpay_order_id, amount, plan,
                   promo_code=None, discount_amount=0):
    row = execute(
        """INSERT INTO payments
           (user_id, razorpay_order_id, amount, plan, promo_code,
            discount_amount, original_amount, status)
           VALUES (%s, %s, %s, %s, %s, %s, %s, 'created')
           RETURNING id""",
        (user_id, razorpay_order_id, amount, plan,
         promo_code, discount_amount, amount + discount_amount)
    )
    return row['id'] if row else None

def confirm_payment(razorpay_order_id, razorpay_payment_id,
                    razorpay_signature, pro_expires_at):
    execute(
        """UPDATE payments SET
           razorpay_payment_id=%s,
           razorpay_signature=%s,
           status='paid',
           paid_at=NOW(),
           pro_expires_at=%s
           WHERE razorpay_order_id=%s""",
        (razorpay_payment_id, razorpay_signature, pro_expires_at, razorpay_order_id)
    )

def get_user_payments(user_id, limit=10):
    return query_all(
        """SELECT * FROM payments WHERE user_id=%s
           ORDER BY created_at DESC LIMIT %s""",
        (user_id, limit)
    )

# ══════════════════════════════════════════════════════════
# TRIPS
# ══════════════════════════════════════════════════════════

def save_trip(user_id, destination, origin, days, people,
              budget, currency, vibes, passport,
              plan_data=None, start_date=None):
    vibes_str = vibes if isinstance(vibes, str) else '+'.join(vibes)
    end_date = None
    if start_date:
        try:
            sd = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = sd + timedelta(days=int(days)-1)
        except:
            pass
    row = execute(
        """INSERT INTO trips
           (user_id,destination,origin,days,people,budget,currency,
            vibes,passport,plan_data,start_date,end_date)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           RETURNING id""",
        (user_id, destination, origin, int(days), int(people),
         str(budget), currency, vibes_str, passport,
         json.dumps(plan_data) if plan_data else None,
         start_date, end_date)
    )
    return row['id'] if row else None

def get_trips(user_id):
    rows = query_all(
        """SELECT * FROM trips WHERE user_id=%s AND deleted_at IS NULL
           ORDER BY updated_at DESC""",
        (user_id,)
    )
    for t in rows:
        if isinstance(t.get('plan_data'), str):
            try: t['plan_data'] = json.loads(t['plan_data'])
            except: t['plan_data'] = None
        for k, v in t.items():
            if hasattr(v, 'isoformat'): t[k] = v.isoformat()
    return rows

def get_trip(trip_id, user_id):
    row = query_one(
        'SELECT * FROM trips WHERE id=%s AND user_id=%s AND deleted_at IS NULL',
        (trip_id, user_id)
    )
    if row:
        if isinstance(row.get('plan_data'), str):
            try: row['plan_data'] = json.loads(row['plan_data'])
            except: pass
        for k, v in row.items():
            if hasattr(v, 'isoformat'): row[k] = v.isoformat()
    return row

def update_trip(trip_id, user_id, **fields):
    allowed = ['destination','origin','days','people','budget','currency',
               'vibes','passport','status','plan_data','notes','is_favourite']
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates: return
    if 'plan_data' in updates and isinstance(updates['plan_data'], (dict, list)):
        updates['plan_data'] = json.dumps(updates['plan_data'])
    sets = ', '.join(f'{k}=%s' for k in updates)
    execute(
        f'UPDATE trips SET {sets} WHERE id=%s AND user_id=%s',
        (*updates.values(), trip_id, user_id)
    )

def delete_trip(trip_id, user_id):
    # Soft delete
    execute(
        'UPDATE trips SET deleted_at=NOW() WHERE id=%s AND user_id=%s',
        (trip_id, user_id)
    )

# ══════════════════════════════════════════════════════════
# SAVED PLACES
# ══════════════════════════════════════════════════════════

def save_place(user_id, name, city='', country='', continent='',
               description='', image_url='', emoji='📍',
               tags=None, trip_id=None, lat=None, lng=None):
    row = execute(
        """INSERT INTO saved_places
           (user_id,name,city,country,continent,description,
            image_url,emoji,tags,trip_id,lat,lng)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           RETURNING id""",
        (user_id, name, city, country, continent, description,
         image_url, emoji, json.dumps(tags or []), trip_id, lat, lng)
    )
    return row['id'] if row else None

def get_places(user_id):
    rows = query_all(
        """SELECT * FROM saved_places WHERE user_id=%s AND deleted_at IS NULL
           ORDER BY created_at DESC""",
        (user_id,)
    )
    for p in rows:
        if isinstance(p.get('tags'), str):
            try: p['tags'] = json.loads(p['tags'])
            except: p['tags'] = []
        for k, v in p.items():
            if hasattr(v, 'isoformat'): p[k] = v.isoformat()
    return rows

def delete_place(place_id, user_id):
    execute(
        'UPDATE saved_places SET deleted_at=NOW() WHERE id=%s AND user_id=%s',
        (place_id, user_id)
    )

# ══════════════════════════════════════════════════════════
# EXPENSES
# ══════════════════════════════════════════════════════════

def add_expense(trip_id, user_id, title, amount, category='Other',
                currency='INR', paid_by='', split_with=None, notes=''):
    row = execute(
        """INSERT INTO expenses
           (trip_id,user_id,title,amount,category,currency,paid_by,split_with,notes)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
           RETURNING id""",
        (trip_id, user_id, title, float(amount), category,
         currency, paid_by, json.dumps(split_with or []), notes)
    )
    return row['id'] if row else None

def get_expenses(trip_id, user_id):
    rows = query_all(
        """SELECT * FROM expenses WHERE trip_id=%s AND user_id=%s
           AND deleted_at IS NULL ORDER BY created_at DESC""",
        (trip_id, user_id)
    )
    for e in rows:
        if isinstance(e.get('split_with'), str):
            try: e['split_with'] = json.loads(e['split_with'])
            except: e['split_with'] = []
        for k, v in e.items():
            if hasattr(v, 'isoformat'): e[k] = v.isoformat()
    return rows

def delete_expense(expense_id, user_id):
    execute(
        'UPDATE expenses SET deleted_at=NOW() WHERE id=%s AND user_id=%s',
        (expense_id, user_id)
    )

# ══════════════════════════════════════════════════════════
# JOURNALS
# ══════════════════════════════════════════════════════════

def save_journal(trip_id, user_id, content):
    content_str = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
    execute(
        """INSERT INTO journals (trip_id, user_id, content)
           VALUES (%s, %s, %s)
           ON CONFLICT (trip_id) DO UPDATE
           SET content=EXCLUDED.content, updated_at=NOW()""",
        (trip_id, user_id, content_str)
    )

def get_journal(trip_id, user_id):
    row = query_one(
        'SELECT * FROM journals WHERE trip_id=%s AND user_id=%s',
        (trip_id, user_id)
    )
    if row:
        if isinstance(row.get('content'), str):
            try: row['content'] = json.loads(row['content'])
            except: pass
        for k, v in row.items():
            if hasattr(v, 'isoformat'): row[k] = v.isoformat()
    return row

# ══════════════════════════════════════════════════════════
# DIARY
# ══════════════════════════════════════════════════════════

def get_diary_trips(user_id):
    rows = query_all(
        """SELECT * FROM diary_trips WHERE user_id=%s AND deleted_at IS NULL
           ORDER BY created_at DESC""",
        (user_id,)
    )
    for r in rows:
        for k, v in r.items():
            if hasattr(v, 'isoformat'): r[k] = v.isoformat()
    return rows

def get_diary_trip(trip_id, user_id):
    row = query_one(
        'SELECT * FROM diary_trips WHERE id=%s AND user_id=%s AND deleted_at IS NULL',
        (trip_id, user_id)
    )
    if row:
        for k, v in row.items():
            if hasattr(v, 'isoformat'): row[k] = v.isoformat()
    return row

def create_diary_trip(user_id, destination, currency='INR', start_date=None):
    row = execute(
        """INSERT INTO diary_trips (user_id, destination, currency, start_date)
           VALUES (%s, %s, %s, %s) RETURNING id""",
        (user_id, destination, currency, start_date)
    )
    return row['id'] if row else None

def get_diary_entries(user_id, trip_id=None, entry_type=None,
                       search=None, limit=100):
    sql = """SELECT * FROM diary_entries
             WHERE user_id=%s AND deleted_at IS NULL"""
    params = [user_id]
    if trip_id:
        sql += ' AND trip_id=%s'; params.append(trip_id)
    if entry_type:
        sql += ' AND type=%s'; params.append(entry_type)
    if search:
        sql += ' AND to_tsvector(\'english\', text) @@ plainto_tsquery(%s)'
        params.append(search)
    sql += ' ORDER BY created_at DESC LIMIT %s'
    params.append(limit)
    rows = query_all(sql, params)
    for e in rows:
        for field in ['tags', 'photos']:
            if isinstance(e.get(field), str):
                try: e[field] = json.loads(e[field])
                except: e[field] = []
        for k, v in e.items():
            if hasattr(v, 'isoformat'): e[k] = v.isoformat()
    return rows

def create_diary_entry(user_id, trip_id, entry_type, text, mood='',
                        location='', tags=None, photos=None,
                        amount=0, currency='INR', day_number=1):
    row = execute(
        """INSERT INTO diary_entries
           (user_id,trip_id,type,text,mood,location,tags,photos,amount,currency,day_number)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           RETURNING id""",
        (user_id, trip_id, entry_type, text, mood, location,
         json.dumps(tags or []), json.dumps(photos or []),
         float(amount or 0), currency, int(day_number or 1))
    )
    return row['id'] if row else None

def delete_diary_entry(entry_id, user_id):
    execute(
        'UPDATE diary_entries SET deleted_at=NOW() WHERE id=%s AND user_id=%s',
        (entry_id, user_id)
    )

def toggle_diary_favorite(entry_id, user_id):
    execute(
        """UPDATE diary_entries
           SET is_favorite = NOT is_favorite
           WHERE id=%s AND user_id=%s""",
        (entry_id, user_id)
    )
    row = query_one(
        'SELECT is_favorite FROM diary_entries WHERE id=%s',
        (entry_id,)
    )
    return bool(row['is_favorite']) if row else False

def get_diary_stats(user_id, trip_id=None):
    params = [user_id]
    trip_filter = ''
    if trip_id:
        trip_filter = ' AND trip_id=%s'
        params.append(trip_id)
    total = query_one(
        f'SELECT COUNT(*) AS c FROM diary_entries WHERE user_id=%s{trip_filter} AND deleted_at IS NULL',
        params
    )
    spent = query_one(
        f'SELECT COALESCE(SUM(amount),0) AS s FROM diary_entries WHERE user_id=%s AND type=\'expense\'{trip_filter} AND deleted_at IS NULL',
        params
    )
    days = query_one(
        f'SELECT COUNT(DISTINCT DATE(created_at)) AS d FROM diary_entries WHERE user_id=%s{trip_filter} AND deleted_at IS NULL',
        params
    )
    return {
        'total_entries': (total or {}).get('c', 0),
        'total_spent':   float((spent or {}).get('s', 0)),
        'days_logged':   (days or {}).get('d', 0),
    }

# ══════════════════════════════════════════════════════════
# PROMO CODES
# ══════════════════════════════════════════════════════════

def get_promo_code(code):
    return query_one(
        """SELECT * FROM promo_codes
           WHERE code=%s AND is_active=TRUE AND expires_at > NOW()""",
        (code.upper().strip(),)
    )

def redeem_promo(user_id, code, discount_amount=0, pro_months=0):
    try:
        execute(
            """INSERT INTO promo_redemptions
               (user_id, code, discount_amount, pro_months)
               VALUES (%s, %s, %s, %s)""",
            (user_id, code, discount_amount, pro_months)
        )
        execute(
            'UPDATE promo_codes SET uses=uses+1 WHERE code=%s',
            (code,)
        )
        return True
    except psycopg2.IntegrityError:
        return False  # Already redeemed

def check_promo_redeemed(user_id, code):
    row = query_one(
        'SELECT id FROM promo_redemptions WHERE user_id=%s AND code=%s',
        (user_id, code)
    )
    return bool(row)

# ══════════════════════════════════════════════════════════
# NOTIFICATIONS
# ══════════════════════════════════════════════════════════

def create_notification(user_id, notif_type, title, body, data=None):
    execute(
        """INSERT INTO notifications (user_id, type, title, body, data)
           VALUES (%s, %s, %s, %s, %s)""",
        (user_id, notif_type, title, body, json.dumps(data or {}))
    )

def get_notifications(user_id, unread_only=False, limit=20):
    sql = 'SELECT * FROM notifications WHERE user_id=%s'
    params = [user_id]
    if unread_only:
        sql += ' AND is_read=FALSE'
    sql += ' ORDER BY created_at DESC LIMIT %s'
    params.append(limit)
    return query_all(sql, params)

def mark_notifications_read(user_id):
    execute(
        'UPDATE notifications SET is_read=TRUE, read_at=NOW() WHERE user_id=%s AND is_read=FALSE',
        (user_id,)
    )

# ══════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════

def log_action(user_id, action, ip='', feature='', duration_ms=None,
               success=True, error=None, metadata=None):
    try:
        execute(
            """INSERT INTO usage_logs
               (user_id, action, feature, ip_address, duration_ms, success, error, metadata)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (user_id, action, feature, ip, duration_ms,
             success, error, json.dumps(metadata or {}))
        )
    except:
        pass

# ══════════════════════════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════════════════════════

def submit_feedback(user_id, message, rating=None, feedback_type='general',
                    subject='', feature='', email=''):
    execute(
        """INSERT INTO feedback
           (user_id, message, rating, type, subject, feature, email)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (user_id, message, rating, feedback_type, subject, feature, email)
    )

# ══════════════════════════════════════════════════════════
# ADMIN QUERIES
# ══════════════════════════════════════════════════════════

def admin_get_stats():
    return {
        'total_users':     (query_one('SELECT COUNT(*) AS c FROM users WHERE deleted_at IS NULL') or {}).get('c', 0),
        'total_trips':     (query_one('SELECT COUNT(*) AS c FROM trips WHERE deleted_at IS NULL') or {}).get('c', 0),
        'pro_users':       (query_one("SELECT COUNT(*) AS c FROM users WHERE plan_type='pro' AND deleted_at IS NULL") or {}).get('c', 0),
        'total_revenue':   (query_one("SELECT COALESCE(SUM(amount),0)/100.0 AS r FROM payments WHERE status='paid'") or {}).get('r', 0),
        'revenue_today':   (query_one("SELECT COALESCE(SUM(amount),0)/100.0 AS r FROM payments WHERE status='paid' AND DATE(paid_at)=CURRENT_DATE") or {}).get('r', 0),
        'new_users_today': (query_one('SELECT COUNT(*) AS c FROM users WHERE DATE(created_at)=CURRENT_DATE AND deleted_at IS NULL') or {}).get('c', 0),
        'trips_today':     (query_one('SELECT COUNT(*) AS c FROM trips WHERE DATE(created_at)=CURRENT_DATE AND deleted_at IS NULL') or {}).get('c', 0),
        'actions_today':   (query_one('SELECT COUNT(*) AS c FROM usage_logs WHERE DATE(created_at)=CURRENT_DATE') or {}).get('c', 0),
        'total_payments':  (query_one("SELECT COUNT(*) AS c FROM payments WHERE status='paid'") or {}).get('c', 0),
    }

def admin_get_users(limit=200, offset=0, search=None):
    sql = """SELECT u.*,
             COUNT(t.id) AS trip_count,
             COALESCE(SUM(p.amount)/100.0, 0) AS total_paid
             FROM users u
             LEFT JOIN trips t ON t.user_id=u.id AND t.deleted_at IS NULL
             LEFT JOIN payments p ON p.user_id=u.id AND p.status='paid'
             WHERE u.deleted_at IS NULL"""
    params = []
    if search:
        sql += ' AND (u.email ILIKE %s OR u.name ILIKE %s)'
        params += [f'%{search}%', f'%{search}%']
    sql += ' GROUP BY u.id ORDER BY u.created_at DESC LIMIT %s OFFSET %s'
    params += [limit, offset]
    rows = query_all(sql, params)
    return [safe_user(r) for r in rows]

def admin_get_revenue_chart(days=30):
    return query_all(
        """SELECT DATE(paid_at) AS date,
           COUNT(*) AS transactions,
           SUM(amount)/100.0 AS revenue
           FROM payments
           WHERE status='paid' AND paid_at > NOW() - INTERVAL '%s days'
           GROUP BY DATE(paid_at)
           ORDER BY date DESC""",
        (days,)
    )

def link_google_to_user(user_id, google_id, avatar=''):
    execute(
        'UPDATE users SET google_id=%s, avatar=%s, email_verified=TRUE WHERE id=%s',
        (google_id, avatar, user_id)
    )

def update_user_password(user_id, password_hash):
    execute(
        'UPDATE users SET password_hash=%s WHERE id=%s',
        (password_hash, user_id)
    )

def delete_user(user_id):
    execute(
        'UPDATE users SET deleted_at=NOW() WHERE id=%s',
        (user_id,)
    )

# ══════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════

def init_db():
    """Test connection on startup"""
    try:
        row = query_one('SELECT NOW() AS time')
        print(f"[DB] Supabase connected ✅ — {row['time']}")
    except Exception as e:
        print(f"[DB] Connection failed ❌ — {e}")
        raise

if __name__ == '__main__':
    init_db()
    print("[DB] All systems go")