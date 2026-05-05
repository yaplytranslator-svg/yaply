"""
database.py — Yaply Complete Database Layer
Tables: users, trips, saved_places, sessions, usage_logs
"""
import sqlite3, os, json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'yaply.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    # ── USERS ──
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            password    TEXT,
            google_id   TEXT UNIQUE,
            avatar      TEXT,
            passport    TEXT DEFAULT 'India',
            home_city   TEXT DEFAULT '',
            currency    TEXT DEFAULT 'INR',
            is_verified INTEGER DEFAULT 0,
            is_pro      INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT (datetime('now')),
            last_login  TEXT DEFAULT (datetime('now'))
        )
    ''')

    # ── TRIPS ──
    c.execute('''
        CREATE TABLE IF NOT EXISTS trips (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            destination  TEXT NOT NULL,
            origin       TEXT NOT NULL,
            days         INTEGER NOT NULL,
            people       INTEGER DEFAULT 1,
            budget       TEXT DEFAULT '50000',
            currency     TEXT DEFAULT 'INR',
            vibes        TEXT DEFAULT 'Adventure',
            passport     TEXT DEFAULT 'India',
            status       TEXT DEFAULT 'planning',
            plan_data    TEXT,
            notes        TEXT DEFAULT '',
            is_favourite INTEGER DEFAULT 0,
            created_at   TEXT DEFAULT (datetime('now')),
            updated_at   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # ── SAVED PLACES ──
    c.execute('''
        CREATE TABLE IF NOT EXISTS saved_places (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            name        TEXT NOT NULL,
            city        TEXT DEFAULT '',
            country     TEXT DEFAULT '',
            continent   TEXT DEFAULT '',
            description TEXT DEFAULT '',
            image_url   TEXT DEFAULT '',
            emoji       TEXT DEFAULT '📍',
            tags        TEXT DEFAULT '[]',
            trip_id     INTEGER,
            created_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (trip_id) REFERENCES trips(id) ON DELETE SET NULL
        )
    ''')

    # ── EXPENSES (per trip) ──
    c.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id     INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            title       TEXT NOT NULL,
            amount      REAL NOT NULL,
            category    TEXT DEFAULT 'Other',
            currency    TEXT DEFAULT 'INR',
            paid_by     TEXT DEFAULT '',
            split_with  TEXT DEFAULT '[]',
            date        TEXT DEFAULT (date('now')),
            created_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (trip_id) REFERENCES trips(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # ── JOURNALS (per trip) ──
    c.execute('''
        CREATE TABLE IF NOT EXISTS journals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id     INTEGER NOT NULL UNIQUE,
            user_id     INTEGER NOT NULL,
            content     TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (trip_id) REFERENCES trips(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # ── USAGE LOGS ──
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            action      TEXT NOT NULL,
            ip          TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now'))
        )
    ''')

    conn.commit()
    conn.close()
    print(f"[DB] Initialized at {DB_PATH}")

# ── USER HELPERS ──

def create_user(email, name, password_hash=None, google_id=None, avatar=None):
    conn = get_db()
    try:
        c = conn.execute(
            'INSERT INTO users (email, name, password, google_id, avatar, is_verified) VALUES (?,?,?,?,?,?)',
            (email.lower().strip(), name.strip(), password_hash, google_id, avatar, 1 if google_id else 0)
        )
        conn.commit()
        return c.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_by_email(email):
    conn = get_db()
    row = conn.execute('SELECT * FROM users WHERE email=?', (email.lower().strip(),)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_google(google_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM users WHERE google_id=?', (google_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def update_user(user_id, **fields):
    allowed = ['name','avatar','passport','home_city','currency','last_login','is_verified','is_pro']
    updates = {k:v for k,v in fields.items() if k in allowed}
    if not updates: return
    conn = get_db()
    sets = ', '.join(f'{k}=?' for k in updates)
    conn.execute(f'UPDATE users SET {sets} WHERE id=?', (*updates.values(), user_id))
    conn.commit()
    conn.close()

def get_user_stats(user_id):
    conn = get_db()
    trips = conn.execute('SELECT COUNT(*) as c FROM trips WHERE user_id=?', (user_id,)).fetchone()['c']
    days  = conn.execute('SELECT SUM(days) as s FROM trips WHERE user_id=?', (user_id,)).fetchone()['s'] or 0
    places= conn.execute('SELECT COUNT(*) as c FROM saved_places WHERE user_id=?', (user_id,)).fetchone()['c']
    conn.close()
    return {'trips': trips, 'days': days, 'places': places}

# ── TRIP HELPERS ──

def save_trip(user_id, destination, origin, days, people, budget, currency, vibes, passport, plan_data=None):
    conn = get_db()
    c = conn.execute(
        'INSERT INTO trips (user_id,destination,origin,days,people,budget,currency,vibes,passport,plan_data) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (user_id, destination, origin, int(days), int(people), str(budget), currency,
         vibes if isinstance(vibes,str) else '+'.join(vibes),
         passport, json.dumps(plan_data) if plan_data else None)
    )
    conn.commit()
    trip_id = c.lastrowid
    conn.close()
    return trip_id

def get_trips(user_id):
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM trips WHERE user_id=? ORDER BY updated_at DESC',
        (user_id,)
    ).fetchall()
    conn.close()
    trips = []
    for r in rows:
        t = dict(r)
        t['plan_data'] = json.loads(t['plan_data']) if t['plan_data'] else None
        trips.append(t)
    return trips

def get_trip(trip_id, user_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM trips WHERE id=? AND user_id=?', (trip_id, user_id)).fetchone()
    conn.close()
    if not row: return None
    t = dict(row)
    t['plan_data'] = json.loads(t['plan_data']) if t['plan_data'] else None
    return t

def update_trip(trip_id, user_id, **fields):
    allowed = ['destination','origin','days','people','budget','currency','vibes',
               'passport','status','plan_data','notes','is_favourite']
    updates = {k:v for k,v in fields.items() if k in allowed}
    if not updates: return
    if 'plan_data' in updates and isinstance(updates['plan_data'], (dict,list)):
        updates['plan_data'] = json.dumps(updates['plan_data'])
    updates['updated_at'] = datetime.now().isoformat()
    conn = get_db()
    sets = ', '.join(f'{k}=?' for k in updates)
    conn.execute(f'UPDATE trips SET {sets} WHERE id=? AND user_id=?', (*updates.values(), trip_id, user_id))
    conn.commit()
    conn.close()

def delete_trip(trip_id, user_id):
    conn = get_db()
    conn.execute('DELETE FROM trips WHERE id=? AND user_id=?', (trip_id, user_id))
    conn.commit()
    conn.close()

def set_active_trip(user_id, trip_id):
    """Mark one trip as active, rest as planning"""
    conn = get_db()
    conn.execute('UPDATE trips SET status=? WHERE user_id=? AND status=?', ('planning','active'))
    conn.execute('UPDATE trips SET status=? WHERE id=? AND user_id=?', ('active', trip_id, user_id))
    conn.commit()
    conn.close()

# ── SAVED PLACES ──

def save_place(user_id, name, city='', country='', continent='', description='', image_url='', emoji='📍', tags=None, trip_id=None):
    conn = get_db()
    c = conn.execute(
        'INSERT INTO saved_places (user_id,name,city,country,continent,description,image_url,emoji,tags,trip_id) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (user_id, name, city, country, continent, description, image_url, emoji,
         json.dumps(tags or []), trip_id)
    )
    conn.commit()
    place_id = c.lastrowid
    conn.close()
    return place_id

def get_places(user_id):
    conn = get_db()
    rows = conn.execute('SELECT * FROM saved_places WHERE user_id=? ORDER BY created_at DESC', (user_id,)).fetchall()
    conn.close()
    places = []
    for r in rows:
        p = dict(r)
        try: p['tags'] = json.loads(p['tags'])
        except: p['tags'] = []
        places.append(p)
    return places

def delete_place(place_id, user_id):
    conn = get_db()
    conn.execute('DELETE FROM saved_places WHERE id=? AND user_id=?', (place_id, user_id))
    conn.commit()
    conn.close()

# ── EXPENSES ──

def add_expense(trip_id, user_id, title, amount, category='Other', currency='INR', paid_by='', split_with=None):
    conn = get_db()
    c = conn.execute(
        'INSERT INTO expenses (trip_id,user_id,title,amount,category,currency,paid_by,split_with) VALUES (?,?,?,?,?,?,?,?)',
        (trip_id, user_id, title, float(amount), category, currency, paid_by, json.dumps(split_with or []))
    )
    conn.commit()
    exp_id = c.lastrowid
    conn.close()
    return exp_id

def get_expenses(trip_id, user_id):
    conn = get_db()
    rows = conn.execute('SELECT * FROM expenses WHERE trip_id=? AND user_id=? ORDER BY created_at DESC', (trip_id, user_id)).fetchall()
    conn.close()
    expenses = []
    for r in rows:
        e = dict(r)
        try: e['split_with'] = json.loads(e['split_with'])
        except: e['split_with'] = []
        expenses.append(e)
    return expenses

def delete_expense(expense_id, user_id):
    conn = get_db()
    conn.execute('DELETE FROM expenses WHERE id=? AND user_id=?', (expense_id, user_id))
    conn.commit()
    conn.close()

# ── JOURNAL ──

def save_journal(trip_id, user_id, content):
    conn = get_db()
    content_str = json.dumps(content) if isinstance(content, (dict,list)) else str(content)
    conn.execute('''
        INSERT INTO journals (trip_id, user_id, content) VALUES (?,?,?)
        ON CONFLICT(trip_id) DO UPDATE SET content=excluded.content, updated_at=datetime('now')
    ''', (trip_id, user_id, content_str))
    conn.commit()
    conn.close()

def get_journal(trip_id, user_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM journals WHERE trip_id=? AND user_id=?', (trip_id, user_id)).fetchone()
    conn.close()
    if not row: return None
    j = dict(row)
    try: j['content'] = json.loads(j['content'])
    except: pass
    return j

# ── USAGE LOGGING ──

def log_action(user_id, action, ip=''):
    try:
        conn = get_db()
        conn.execute('INSERT INTO usage_logs (user_id,action,ip) VALUES (?,?,?)', (user_id, action, ip))
        conn.commit()
        conn.close()
    except: pass

if __name__ == '__main__':
    init_db()
    print("[DB] All tables created successfully")