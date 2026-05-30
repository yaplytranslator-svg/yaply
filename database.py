"""
database.py — Yaply Database Layer v2 FIXED
"""
import sqlite3, os, json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yaply.db')

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def init_db():
    conn = get_db(); c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        password TEXT,
        google_id TEXT UNIQUE,
        avatar TEXT DEFAULT '',
        passport TEXT DEFAULT 'India',
        home_city TEXT DEFAULT '',
        currency TEXT DEFAULT 'INR',
        is_verified INTEGER DEFAULT 0,
        is_pro INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        last_login TEXT DEFAULT (datetime('now'))
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS trips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        destination TEXT NOT NULL,
        origin TEXT NOT NULL DEFAULT '',
        days INTEGER NOT NULL DEFAULT 5,
        people INTEGER DEFAULT 1,
        budget TEXT DEFAULT '50000',
        currency TEXT DEFAULT 'INR',
        vibes TEXT DEFAULT 'Adventure',
        passport TEXT DEFAULT 'India',
        status TEXT DEFAULT 'planning',
        plan_data TEXT,
        notes TEXT DEFAULT '',
        is_favourite INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS saved_places (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        city TEXT DEFAULT '',
        country TEXT DEFAULT '',
        continent TEXT DEFAULT '',
        description TEXT DEFAULT '',
        image_url TEXT DEFAULT '',
        emoji TEXT DEFAULT '📍',
        tags TEXT DEFAULT '[]',
        trip_id INTEGER,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (trip_id) REFERENCES trips(id) ON DELETE SET NULL
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trip_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        amount REAL NOT NULL,
        category TEXT DEFAULT 'Other',
        currency TEXT DEFAULT 'INR',
        paid_by TEXT DEFAULT '',
        split_with TEXT DEFAULT '[]',
        date TEXT DEFAULT (date('now')),
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (trip_id) REFERENCES trips(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS journals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trip_id INTEGER NOT NULL UNIQUE,
        user_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (trip_id) REFERENCES trips(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS usage_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT NOT NULL,
        ip TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    )''')

    # Performance indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_trips_user ON trips(user_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_places_user ON saved_places(user_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_expenses_trip ON expenses(trip_id, user_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_logs_user ON usage_logs(user_id)')

    conn.commit(); conn.close()
    print(f"[DB] Initialized at {DB_PATH}")

# ── USERS ──
def create_user(email, name, password_hash=None, google_id=None, avatar=None):
    conn = get_db()
    try:
        c = conn.execute(
            'INSERT INTO users (email,name,password,google_id,avatar,is_verified) VALUES (?,?,?,?,?,?)',
            (email.lower().strip(), name.strip(), password_hash, google_id, avatar or '', 1 if google_id else 0)
        )
        conn.commit(); return c.lastrowid
    except sqlite3.IntegrityError: return None
    finally: conn.close()

def get_user_by_email(email):
    conn = get_db()
    try:
        row = conn.execute('SELECT * FROM users WHERE email=?', (email.lower().strip(),)).fetchone()
        return dict(row) if row else None
    finally: conn.close()

def get_user_by_id(user_id):
    conn = get_db()
    try:
        row = conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
        return dict(row) if row else None
    finally: conn.close()

def get_user_by_google(google_id):
    conn = get_db()
    try:
        row = conn.execute('SELECT * FROM users WHERE google_id=?', (google_id,)).fetchone()
        return dict(row) if row else None
    finally: conn.close()

def update_user(user_id, **fields):
    allowed = ['name','avatar','passport','home_city','currency','last_login','is_verified','is_pro']
    updates = {k:v for k,v in fields.items() if k in allowed}
    if not updates: return
    conn = get_db()
    try:
        sets = ', '.join(f'{k}=?' for k in updates)
        conn.execute(f'UPDATE users SET {sets} WHERE id=?', (*updates.values(), user_id))
        conn.commit()
    finally: conn.close()

def link_google_to_user(user_id, google_id, avatar=''):
    conn = get_db()
    try:
        conn.execute('UPDATE users SET google_id=?, avatar=? WHERE id=?', (google_id, avatar, user_id))
        conn.commit()
    finally: conn.close()

def update_user_password(user_id, hashed):
    conn = get_db()
    try:
        conn.execute('UPDATE users SET password=? WHERE id=?', (hashed, user_id))
        conn.commit()
    finally: conn.close()

def delete_user(user_id):
    conn = get_db()
    try:
        conn.execute('DELETE FROM users WHERE id=?', (user_id,))
        conn.commit()
    finally: conn.close()

def get_user_stats(user_id):
    conn = get_db()
    try:
        trips  = conn.execute('SELECT COUNT(*) as c FROM trips WHERE user_id=?', (user_id,)).fetchone()['c']
        days   = conn.execute('SELECT SUM(days) as s FROM trips WHERE user_id=?', (user_id,)).fetchone()['s'] or 0
        places = conn.execute('SELECT COUNT(*) as c FROM saved_places WHERE user_id=?', (user_id,)).fetchone()['c']
        return {'trips': trips, 'days': int(days), 'places': places}
    finally: conn.close()

# ── TRIPS ──
def save_trip(user_id, destination, origin, days, people,
              budget, currency, vibes, passport,
              plan_data=None, start_date=None):
    conn = get_db()
    try:
        vibes_str = vibes if isinstance(vibes, str) else '+'.join(vibes)
        c = conn.execute(
            '''INSERT INTO trips
               (user_id,destination,origin,days,people,
                budget,currency,vibes,passport,plan_data,start_date)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            (user_id, destination, origin, int(days), int(people),
             str(budget), currency, vibes_str, passport,
             json.dumps(plan_data) if plan_data else None,
             start_date)
        )
        conn.commit()
        return c.lastrowid
    finally:
        conn.close()

def get_trips(user_id):
    conn = get_db()
    try:
        rows = conn.execute('SELECT * FROM trips WHERE user_id=? ORDER BY updated_at DESC', (user_id,)).fetchall()
        trips = []
        for r in rows:
            t = dict(r)
            try:    t['plan_data'] = json.loads(t['plan_data']) if t['plan_data'] else None
            except: t['plan_data'] = None
            trips.append(t)
        return trips
    finally: conn.close()

def get_trip(trip_id, user_id):
    conn = get_db()
    try:
        row = conn.execute('SELECT * FROM trips WHERE id=? AND user_id=?', (trip_id, user_id)).fetchone()
        if not row: return None
        t = dict(row)
        try:    t['plan_data'] = json.loads(t['plan_data']) if t['plan_data'] else None
        except: t['plan_data'] = None
        return t
    finally: conn.close()

def update_trip(trip_id, user_id, **fields):
    allowed = ['destination','origin','days','people','budget','currency','vibes',
               'passport','status','plan_data','notes','is_favourite']
    updates = {k:v for k,v in fields.items() if k in allowed}
    if not updates: return
    if 'plan_data' in updates and isinstance(updates['plan_data'], (dict, list)):
        updates['plan_data'] = json.dumps(updates['plan_data'])
    updates['updated_at'] = datetime.now().isoformat()
    conn = get_db()
    try:
        sets = ', '.join(f'{k}=?' for k in updates)
        conn.execute(f'UPDATE trips SET {sets} WHERE id=? AND user_id=?', (*updates.values(), trip_id, user_id))
        conn.commit()
    finally: conn.close()

def delete_trip(trip_id, user_id):
    conn = get_db()
    try:
        conn.execute('DELETE FROM trips WHERE id=? AND user_id=?', (trip_id, user_id))
        conn.commit()
    finally: conn.close()

# ── SAVED PLACES ──
def save_place(user_id, name, city='', country='', continent='', description='', image_url='', emoji='📍', tags=None, trip_id=None):
    conn = get_db()
    try:
        c = conn.execute(
            'INSERT INTO saved_places (user_id,name,city,country,continent,description,image_url,emoji,tags,trip_id) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (user_id, name, city, country, continent, description, image_url, emoji, json.dumps(tags or []), trip_id)
        )
        conn.commit(); return c.lastrowid
    finally: conn.close()

def get_places(user_id):
    conn = get_db()
    try:
        rows = conn.execute('SELECT * FROM saved_places WHERE user_id=? ORDER BY created_at DESC', (user_id,)).fetchall()
        places = []
        for r in rows:
            p = dict(r)
            try:    p['tags'] = json.loads(p['tags'])
            except: p['tags'] = []
            places.append(p)
        return places
    finally: conn.close()

def delete_place(place_id, user_id):
    conn = get_db()
    try:
        conn.execute('DELETE FROM saved_places WHERE id=? AND user_id=?', (place_id, user_id))
        conn.commit()
    finally: conn.close()

# ── EXPENSES ──
def add_expense(trip_id, user_id, title, amount, category='Other', currency='INR', paid_by='', split_with=None):
    conn = get_db()
    try:
        c = conn.execute(
            'INSERT INTO expenses (trip_id,user_id,title,amount,category,currency,paid_by,split_with) VALUES (?,?,?,?,?,?,?,?)',
            (trip_id, user_id, title, float(amount), category, currency, paid_by, json.dumps(split_with or []))
        )
        conn.commit(); return c.lastrowid
    finally: conn.close()

def get_expenses(trip_id, user_id):
    conn = get_db()
    try:
        rows = conn.execute('SELECT * FROM expenses WHERE trip_id=? AND user_id=? ORDER BY created_at DESC', (trip_id, user_id)).fetchall()
        expenses = []
        for r in rows:
            e = dict(r)
            try:    e['split_with'] = json.loads(e['split_with'])
            except: e['split_with'] = []
            expenses.append(e)
        return expenses
    finally: conn.close()

def delete_expense(expense_id, user_id):
    conn = get_db()
    try:
        conn.execute('DELETE FROM expenses WHERE id=? AND user_id=?', (expense_id, user_id))
        conn.commit()
    finally: conn.close()

# ── JOURNAL ──
def save_journal(trip_id, user_id, content):
    conn = get_db()
    try:
        content_str = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
        conn.execute('''INSERT INTO journals (trip_id,user_id,content) VALUES (?,?,?)
            ON CONFLICT(trip_id) DO UPDATE SET content=excluded.content, updated_at=datetime('now')
        ''', (trip_id, user_id, content_str))
        conn.commit()
    finally: conn.close()

def get_journal(trip_id, user_id):
    conn = get_db()
    try:
        row = conn.execute('SELECT * FROM journals WHERE trip_id=? AND user_id=?', (trip_id, user_id)).fetchone()
        if not row: return None
        j = dict(row)
        try:    j['content'] = json.loads(j['content'])
        except: pass
        return j
    finally: conn.close()

def migrate_db():
    conn = get_db()
    try:
        # Add start_date to trips if not exists
        conn.execute('ALTER TABLE trips ADD COLUMN start_date TEXT')
        conn.commit()
        print('[DB] start_date column added to trips')
    except Exception as e:
        pass  # Column already exists
    finally:
        conn.close()

# ── LOGGING ──
def log_action(user_id, action, ip=''):
    try:
        conn = get_db()
        conn.execute('INSERT INTO usage_logs (user_id,action,ip) VALUES (?,?,?)', (user_id, action, ip))
        conn.commit(); conn.close()
    except: pass

if __name__ == '__main__':
    init_db()
    print("[DB] All tables created successfully")