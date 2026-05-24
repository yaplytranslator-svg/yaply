# ═══════════════════════════════════════════════════════════════
# YAPLY TRAVEL DIARY — Complete Backend
# File: diary.py
# ═══════════════════════════════════════════════════════════════
#
# HOW TO USE:
# 1. Save this file as diary.py in your project root
# 2. In app.py add at the top:
#       from diary import diary_bp, init_diary_db
# 3. After app = Flask(__name__) add:
#       app.register_blueprint(diary_bp)
# 4. In your DB init call:
#       init_diary_db()
# ═══════════════════════════════════════════════════════════════

import os
import json
import base64
import hashlib
import sqlite3
from datetime import datetime
from functools import wraps
from flask import Blueprint, request, jsonify, g

diary_bp = Blueprint('diary', __name__)

DB_PATH      = os.environ.get('DB_PATH', 'yaply.db')
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'static/diary_photos')

# ════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════

def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    return db


def init_diary_db():
    """Create all diary tables. Call once on startup."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    db = get_db()
    db.executescript('''
        -- Diary entries
        CREATE TABLE IF NOT EXISTS diary_entries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            trip_id     INTEGER,
            entry_type  TEXT    NOT NULL DEFAULT 'note',
            text        TEXT    NOT NULL DEFAULT '',
            mood        TEXT    DEFAULT '😊',
            location    TEXT    DEFAULT '',
            lat         REAL,
            lng         REAL,
            amount      REAL,
            currency    TEXT    DEFAULT 'INR',
            category    TEXT,
            tags        TEXT    DEFAULT '[]',
            photos      TEXT    DEFAULT '[]',
            day_number  INTEGER DEFAULT 1,
            is_favorite INTEGER DEFAULT 0,
            is_deleted  INTEGER DEFAULT 0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Diary photos (stored separately for efficiency)
        CREATE TABLE IF NOT EXISTS diary_photos (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id    INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            filename    TEXT    NOT NULL,
            file_path   TEXT    NOT NULL,
            file_size   INTEGER,
            width       INTEGER,
            height      INTEGER,
            caption     TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(entry_id) REFERENCES diary_entries(id)
        );

        -- Diary trips (a diary can span a trip)
        CREATE TABLE IF NOT EXISTS diary_trips (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            destination TEXT    NOT NULL,
            start_date  TEXT,
            end_date    TEXT,
            cover_photo TEXT,
            total_spent REAL    DEFAULT 0,
            currency    TEXT    DEFAULT 'INR',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active   INTEGER DEFAULT 1
        );

        -- Expense summary cache (updated on every expense entry)
        CREATE TABLE IF NOT EXISTS diary_expenses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            trip_id     INTEGER,
            entry_id    INTEGER NOT NULL,
            amount      REAL    NOT NULL,
            currency    TEXT    DEFAULT 'INR',
            category    TEXT    NOT NULL,
            description TEXT,
            date        TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(entry_id) REFERENCES diary_entries(id)
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_de_user   ON diary_entries(user_id);
        CREATE INDEX IF NOT EXISTS idx_de_trip   ON diary_entries(trip_id);
        CREATE INDEX IF NOT EXISTS idx_de_type   ON diary_entries(entry_type);
        CREATE INDEX IF NOT EXISTS idx_de_date   ON diary_entries(created_at);
        CREATE INDEX IF NOT EXISTS idx_dp_entry  ON diary_photos(entry_id);
        CREATE INDEX IF NOT EXISTS idx_dexp_user ON diary_expenses(user_id);
        CREATE INDEX IF NOT EXISTS idx_dexp_trip ON diary_expenses(trip_id);
    ''')
    db.commit()
    db.close()
    print('[Diary] Database initialized ✓')


# ════════════════════════════════════════════════════════════
# AUTH
# ════════════════════════════════════════════════════════════

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth  = request.headers.get('Authorization', '')
        if auth.startswith('Bearer '):
            token = auth[7:]
        if not token:
            token = request.headers.get('X-Auth-Token')
        if not token:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401
        try:
            import jwt as pyjwt
            secret  = os.environ.get('JWT_SECRET', 'yaply-secret-key')
            payload = pyjwt.decode(token, secret, algorithms=['HS256'])
            g.user_id = payload.get('user_id') or payload.get('id')
            if not g.user_id:
                return jsonify({'success': False, 'error': 'Invalid token'}), 401
        except Exception:
            return jsonify({'success': False, 'error': 'Invalid or expired token'}), 401
        return f(*args, **kwargs)
    return decorated


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    # Parse JSON fields
    for field in ('tags', 'photos'):
        if field in d and d[field]:
            try:
                d[field] = json.loads(d[field])
            except Exception:
                d[field] = []
    return d


def rows_to_list(rows):
    return [row_to_dict(r) for r in rows]


def save_photo_file(base64_data, user_id, entry_id):
    """Save a base64 photo to disk and return the file path."""
    try:
        # Strip data URI prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]

        img_bytes = base64.b64decode(base64_data)
        file_hash = hashlib.md5(img_bytes).hexdigest()[:12]
        filename  = 'diary_' + str(user_id) + '_' + str(entry_id) + '_' + file_hash + '.jpg'

        user_folder = os.path.join(UPLOAD_FOLDER, str(user_id))
        os.makedirs(user_folder, exist_ok=True)

        file_path = os.path.join(user_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(img_bytes)

        # Return web-accessible path
        web_path = '/static/diary_photos/' + str(user_id) + '/' + filename
        return web_path, filename, len(img_bytes)
    except Exception as e:
        print('[Diary] Photo save error:', e)
        return None, None, 0


def update_expense_totals(db, user_id, trip_id):
    """Recalculate and cache expense totals for a trip."""
    if not trip_id:
        return
    total = db.execute('''
        SELECT COALESCE(SUM(amount), 0) as total
        FROM diary_expenses
        WHERE user_id = ? AND trip_id = ?
    ''', (user_id, trip_id)).fetchone()['total']

    db.execute(
        'UPDATE diary_trips SET total_spent = ? WHERE id = ? AND user_id = ?',
        (total, trip_id, user_id)
    )


# ════════════════════════════════════════════════════════════
# TRIP ROUTES
# ════════════════════════════════════════════════════════════

@diary_bp.route('/api/diary/trips', methods=['POST'])
@require_auth
def create_trip():
    """Create a new diary trip."""
    data        = request.get_json() or {}
    destination = (data.get('destination') or '').strip()
    start_date  = data.get('start_date', '')
    end_date    = data.get('end_date', '')
    currency    = data.get('currency', 'INR')

    if not destination:
        return jsonify({'success': False, 'error': 'Destination required'}), 400

    db = get_db()
    try:
        db.execute('''
            INSERT INTO diary_trips
                (user_id, destination, start_date, end_date, currency)
            VALUES (?, ?, ?, ?, ?)
        ''', (g.user_id, destination, start_date, end_date, currency))

        trip_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
        db.commit()

        trip = row_to_dict(db.execute(
            'SELECT * FROM diary_trips WHERE id = ?', (trip_id,)
        ).fetchone())

        return jsonify({'success': True, 'trip': trip, 'trip_id': trip_id}), 201

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@diary_bp.route('/api/diary/trips', methods=['GET'])
@require_auth
def get_trips():
    """Get all diary trips for the current user."""
    db = get_db()
    try:
        trips = db.execute('''
            SELECT dt.*,
                   COUNT(de.id) as entry_count,
                   COUNT(CASE WHEN de.entry_type='expense' THEN 1 END) as expense_count
            FROM diary_trips dt
            LEFT JOIN diary_entries de ON de.trip_id = dt.id AND de.is_deleted = 0
            WHERE dt.user_id = ? AND dt.is_active = 1
            GROUP BY dt.id
            ORDER BY dt.created_at DESC
        ''', (g.user_id,)).fetchall()

        return jsonify({'success': True, 'trips': rows_to_list(trips)})
    finally:
        db.close()


@diary_bp.route('/api/diary/trips/<int:trip_id>', methods=['GET'])
@require_auth
def get_trip(trip_id):
    """Get a single trip with full stats."""
    db = get_db()
    try:
        trip = db.execute(
            'SELECT * FROM diary_trips WHERE id = ? AND user_id = ?',
            (trip_id, g.user_id)
        ).fetchone()

        if not trip:
            return jsonify({'success': False, 'error': 'Trip not found'}), 404

        trip = row_to_dict(trip)

        # Entry counts by type
        counts = db.execute('''
            SELECT entry_type, COUNT(*) as count
            FROM diary_entries
            WHERE trip_id = ? AND user_id = ? AND is_deleted = 0
            GROUP BY entry_type
        ''', (trip_id, g.user_id)).fetchall()

        trip['type_counts'] = {r['entry_type']: r['count'] for r in counts}

        # Expense breakdown by category
        expenses = db.execute('''
            SELECT category, SUM(amount) as total, COUNT(*) as count
            FROM diary_expenses
            WHERE trip_id = ? AND user_id = ?
            GROUP BY category
            ORDER BY total DESC
        ''', (trip_id, g.user_id)).fetchall()

        trip['expense_breakdown'] = rows_to_list(expenses)

        # Days covered
        days = db.execute('''
            SELECT DISTINCT day_number FROM diary_entries
            WHERE trip_id = ? AND user_id = ? AND is_deleted = 0
            ORDER BY day_number
        ''', (trip_id, g.user_id)).fetchall()

        trip['days_covered'] = [r['day_number'] for r in days]

        return jsonify({'success': True, 'trip': trip})
    finally:
        db.close()


@diary_bp.route('/api/diary/trips/<int:trip_id>', methods=['DELETE'])
@require_auth
def delete_trip(trip_id):
    """Soft delete a trip and all its entries."""
    db = get_db()
    try:
        db.execute(
            'UPDATE diary_trips SET is_active = 0 WHERE id = ? AND user_id = ?',
            (trip_id, g.user_id)
        )
        db.execute(
            'UPDATE diary_entries SET is_deleted = 1 WHERE trip_id = ? AND user_id = ?',
            (trip_id, g.user_id)
        )
        db.commit()
        return jsonify({'success': True, 'message': 'Trip deleted'})
    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# ENTRY ROUTES
# ════════════════════════════════════════════════════════════

@diary_bp.route('/api/diary/entries', methods=['POST'])
@require_auth
def create_entry():
    """Create a new diary entry with optional photos."""
    data       = request.get_json() or {}
    trip_id    = data.get('trip_id')
    entry_type = data.get('type', data.get('entry_type', 'note'))
    text       = (data.get('text') or '').strip()
    mood       = data.get('mood', '😊')
    location   = data.get('location', '')
    lat        = data.get('lat')
    lng        = data.get('lng')
    amount     = data.get('amount')
    currency   = data.get('currency', 'INR')
    category   = data.get('category', '')
    tags       = data.get('tags', [])
    photos     = data.get('photos', [])   # base64 strings or URLs
    day_number = data.get('day_number', 1)

    if not text and entry_type != 'expense':
        return jsonify({'success': False, 'error': 'Entry text required'}), 400
    if entry_type == 'expense' and not amount:
        return jsonify({'success': False, 'error': 'Amount required for expense'}), 400

    # Auto-detect expense category
    if entry_type == 'expense' and not category:
        t = text.lower()
        if any(w in t for w in ['food', 'eat', 'lunch', 'dinner', 'breakfast', 'ramen', 'sushi', 'coffee', 'drink']):
            category = 'Food'
        elif any(w in t for w in ['taxi', 'uber', 'train', 'bus', 'metro', 'flight', 'transport']):
            category = 'Transport'
        elif any(w in t for w in ['hotel', 'hostel', 'stay', 'accommodation', 'airbnb']):
            category = 'Stay'
        elif any(w in t for w in ['shop', 'buy', 'souvenir', 'market', 'mall', 'purchase']):
            category = 'Shopping'
        else:
            category = 'Entertainment'

    db = get_db()
    try:
        db.execute('''
            INSERT INTO diary_entries
                (user_id, trip_id, entry_type, text, mood, location, lat, lng,
                 amount, currency, category, tags, photos, day_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            g.user_id, trip_id, entry_type, text, mood, location, lat, lng,
            float(amount) if amount else None, currency, category,
            json.dumps(tags), json.dumps([]), day_number
        ))

        entry_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]

        # Handle photos
        saved_photo_paths = []
        for idx, photo in enumerate(photos[:6]):  # max 6 photos per entry
            if photo.startswith('http') or photo.startswith('/static'):
                # Already a URL
                saved_photo_paths.append(photo)
            elif len(photo) > 100:
                # Base64 — save to disk
                web_path, filename, size = save_photo_file(photo, g.user_id, entry_id)
                if web_path:
                    db.execute('''
                        INSERT INTO diary_photos
                            (entry_id, user_id, filename, file_path, file_size)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (entry_id, g.user_id, filename, web_path, size))
                    saved_photo_paths.append(web_path)

        # Update photos JSON in entry
        db.execute(
            'UPDATE diary_entries SET photos = ? WHERE id = ?',
            (json.dumps(saved_photo_paths), entry_id)
        )

        # If expense — also save to expenses table for fast aggregation
        if entry_type == 'expense' and amount:
            db.execute('''
                INSERT INTO diary_expenses
                    (user_id, trip_id, entry_id, amount, currency, category, description, date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                g.user_id, trip_id, entry_id,
                float(amount), currency, category, text,
                datetime.now().strftime('%Y-%m-%d')
            ))
            update_expense_totals(db, g.user_id, trip_id)

        db.commit()

        entry = row_to_dict(db.execute(
            'SELECT * FROM diary_entries WHERE id = ?', (entry_id,)
        ).fetchone())

        return jsonify({'success': True, 'entry': entry, 'entry_id': entry_id}), 201

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@diary_bp.route('/api/diary/entries', methods=['GET'])
@require_auth
def get_entries():
    """Get diary entries with filtering, search, pagination."""
    trip_id    = request.args.get('trip_id', type=int)
    entry_type = request.args.get('type')
    search     = request.args.get('search', '').strip()
    day        = request.args.get('day', type=int)
    limit      = min(int(request.args.get('limit', 50)), 100)
    offset     = int(request.args.get('offset', 0))
    favorite   = request.args.get('favorite') == '1'

    db = get_db()
    try:
        query  = '''
            SELECT * FROM diary_entries
            WHERE user_id = ? AND is_deleted = 0
        '''
        params = [g.user_id]

        if trip_id:
            query  += ' AND trip_id = ?'
            params.append(trip_id)

        if entry_type:
            query  += ' AND entry_type = ?'
            params.append(entry_type)

        if day:
            query  += ' AND day_number = ?'
            params.append(day)

        if favorite:
            query  += ' AND is_favorite = 1'

        if search:
            query  += ' AND (text LIKE ? OR location LIKE ? OR category LIKE ?)'
            like    = '%' + search + '%'
            params += [like, like, like]

        # Count total
        count_query  = query.replace('SELECT *', 'SELECT COUNT(*) as total')
        total        = db.execute(count_query, params).fetchone()['total']

        query  += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params += [limit, offset]

        entries = rows_to_list(db.execute(query, params).fetchall())

        return jsonify({
            'success': True,
            'entries': entries,
            'total':   total,
            'limit':   limit,
            'offset':  offset,
            'has_more': offset + limit < total
        })

    finally:
        db.close()


@diary_bp.route('/api/diary/entries/<int:entry_id>', methods=['GET'])
@require_auth
def get_entry(entry_id):
    """Get a single entry."""
    db = get_db()
    try:
        entry = db.execute(
            'SELECT * FROM diary_entries WHERE id = ? AND user_id = ? AND is_deleted = 0',
            (entry_id, g.user_id)
        ).fetchone()

        if not entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404

        result = row_to_dict(entry)

        # Include photos
        photos = db.execute(
            'SELECT * FROM diary_photos WHERE entry_id = ?',
            (entry_id,)
        ).fetchall()
        result['photo_details'] = rows_to_list(photos)

        return jsonify({'success': True, 'entry': result})
    finally:
        db.close()


@diary_bp.route('/api/diary/entries/<int:entry_id>', methods=['PUT'])
@require_auth
def update_entry(entry_id):
    """Update an existing entry."""
    db = get_db()
    try:
        entry = db.execute(
            'SELECT * FROM diary_entries WHERE id = ? AND user_id = ? AND is_deleted = 0',
            (entry_id, g.user_id)
        ).fetchone()

        if not entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404

        data     = request.get_json() or {}
        text     = data.get('text',     entry['text'])
        mood     = data.get('mood',     entry['mood'])
        location = data.get('location', entry['location'])
        tags     = data.get('tags',     json.loads(entry['tags'] or '[]'))
        favorite = data.get('is_favorite', entry['is_favorite'])

        db.execute('''
            UPDATE diary_entries
            SET text=?, mood=?, location=?, tags=?, is_favorite=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND user_id=?
        ''', (text, mood, location, json.dumps(tags), favorite, entry_id, g.user_id))

        db.commit()

        updated = row_to_dict(db.execute(
            'SELECT * FROM diary_entries WHERE id = ?', (entry_id,)
        ).fetchone())

        return jsonify({'success': True, 'entry': updated})

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@diary_bp.route('/api/diary/entries/<int:entry_id>', methods=['DELETE'])
@require_auth
def delete_entry(entry_id):
    """Soft delete an entry."""
    db = get_db()
    try:
        entry = db.execute(
            'SELECT * FROM diary_entries WHERE id = ? AND user_id = ?',
            (entry_id, g.user_id)
        ).fetchone()

        if not entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404

        db.execute(
            'UPDATE diary_entries SET is_deleted = 1 WHERE id = ? AND user_id = ?',
            (entry_id, g.user_id)
        )

        # Update expense totals if it was an expense
        if entry['entry_type'] == 'expense' and entry['trip_id']:
            db.execute(
                'DELETE FROM diary_expenses WHERE entry_id = ?', (entry_id,)
            )
            update_expense_totals(db, g.user_id, entry['trip_id'])

        db.commit()
        return jsonify({'success': True, 'message': 'Entry deleted'})

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@diary_bp.route('/api/diary/entries/<int:entry_id>/favorite', methods=['POST'])
@require_auth
def toggle_favorite(entry_id):
    """Toggle favorite on an entry."""
    db = get_db()
    try:
        entry = db.execute(
            'SELECT is_favorite FROM diary_entries WHERE id = ? AND user_id = ?',
            (entry_id, g.user_id)
        ).fetchone()

        if not entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404

        new_val = 0 if entry['is_favorite'] else 1
        db.execute(
            'UPDATE diary_entries SET is_favorite = ? WHERE id = ?',
            (new_val, entry_id)
        )
        db.commit()
        return jsonify({'success': True, 'is_favorite': bool(new_val)})
    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# EXPENSE ROUTES
# ════════════════════════════════════════════════════════════

@diary_bp.route('/api/diary/expenses', methods=['GET'])
@require_auth
def get_expenses():
    """Get full expense summary for a trip."""
    trip_id = request.args.get('trip_id', type=int)

    db = get_db()
    try:
        # Total spent
        base_q = 'SELECT * FROM diary_expenses WHERE user_id = ?'
        base_p = [g.user_id]
        if trip_id:
            base_q += ' AND trip_id = ?'
            base_p.append(trip_id)

        expenses = rows_to_list(db.execute(
            base_q + ' ORDER BY created_at DESC', base_p
        ).fetchall())

        total = sum(e.get('amount', 0) or 0 for e in expenses)

        # By category
        cat_q = '''
            SELECT category,
                   SUM(amount) as total,
                   COUNT(*) as count,
                   MIN(amount) as min_amount,
                   MAX(amount) as max_amount,
                   AVG(amount) as avg_amount
            FROM diary_expenses
            WHERE user_id = ?
        '''
        cat_p = [g.user_id]
        if trip_id:
            cat_q += ' AND trip_id = ?'
            cat_p.append(trip_id)
        cat_q += ' GROUP BY category ORDER BY total DESC'

        by_category = rows_to_list(db.execute(cat_q, cat_p).fetchall())

        # By day
        day_q = '''
            SELECT de.day_number,
                   SUM(exp.amount) as total,
                   COUNT(*) as count
            FROM diary_expenses exp
            JOIN diary_entries de ON de.id = exp.entry_id
            WHERE exp.user_id = ?
        '''
        day_p = [g.user_id]
        if trip_id:
            day_q += ' AND exp.trip_id = ?'
            day_p.append(trip_id)
        day_q += ' GROUP BY de.day_number ORDER BY de.day_number'

        by_day = rows_to_list(db.execute(day_q, day_p).fetchall())

        # Trip budget info
        trip_info = None
        if trip_id:
            trip_info = row_to_dict(db.execute(
                'SELECT * FROM diary_trips WHERE id = ? AND user_id = ?',
                (trip_id, g.user_id)
            ).fetchone())

        return jsonify({
            'success':     True,
            'expenses':    expenses,
            'total':       round(total, 2),
            'by_category': by_category,
            'by_day':      by_day,
            'trip':        trip_info,
            'currency':    (trip_info or {}).get('currency', 'INR')
        })

    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# STATS & INSIGHTS
# ════════════════════════════════════════════════════════════

@diary_bp.route('/api/diary/stats', methods=['GET'])
@require_auth
def get_stats():
    """Get diary stats and travel insights for a trip or all time."""
    trip_id = request.args.get('trip_id', type=int)

    db = get_db()
    try:
        base_where = 'user_id = ? AND is_deleted = 0'
        base_p     = [g.user_id]
        if trip_id:
            base_where += ' AND trip_id = ?'
            base_p.append(trip_id)

        # Entry counts by type
        type_counts = db.execute(
            'SELECT entry_type, COUNT(*) as count FROM diary_entries WHERE ' +
            base_where + ' GROUP BY entry_type',
            base_p
        ).fetchall()

        counts = {r['entry_type']: r['count'] for r in type_counts}
        total  = sum(counts.values())

        # Mood distribution
        moods = db.execute(
            'SELECT mood, COUNT(*) as count FROM diary_entries WHERE ' +
            base_where + ' GROUP BY mood ORDER BY count DESC',
            base_p
        ).fetchall()

        # Most visited locations
        locations = db.execute(
            'SELECT location, COUNT(*) as count FROM diary_entries WHERE ' +
            base_where + " AND location != '' GROUP BY location ORDER BY count DESC LIMIT 5",
            base_p
        ).fetchall()

        # Total photos
        photo_count = 0
        entries_raw = db.execute(
            'SELECT photos FROM diary_entries WHERE ' + base_where, base_p
        ).fetchall()
        for e in entries_raw:
            try:
                photo_count += len(json.loads(e['photos'] or '[]'))
            except Exception:
                pass

        # Total expenses
        total_spent = 0
        exp_row = db.execute(
            'SELECT SUM(amount) as total FROM diary_expenses WHERE user_id = ?' +
            (' AND trip_id = ?' if trip_id else ''),
            [g.user_id] + ([trip_id] if trip_id else [])
        ).fetchone()
        if exp_row:
            total_spent = exp_row['total'] or 0

        # Days travelled
        days = db.execute(
            'SELECT COUNT(DISTINCT day_number) as days FROM diary_entries WHERE ' + base_where,
            base_p
        ).fetchone()['days']

        # Most common mood
        top_mood = moods[0]['mood'] if moods else '😊'

        # Streak calculation
        dates = db.execute(
            'SELECT DISTINCT DATE(created_at) as d FROM diary_entries WHERE ' +
            base_where + ' ORDER BY d DESC',
            base_p
        ).fetchall()

        streak = 0
        if dates:
            today    = datetime.now().date()
            prev     = today
            for row in dates:
                try:
                    d = datetime.strptime(row['d'], '%Y-%m-%d').date()
                    if (prev - d).days <= 1:
                        streak += 1
                        prev    = d
                    else:
                        break
                except Exception:
                    break

        return jsonify({
            'success':     True,
            'total_entries': total,
            'by_type':     counts,
            'total_photos': photo_count,
            'total_spent': round(total_spent, 2),
            'days_logged': days,
            'top_mood':    top_mood,
            'mood_distribution': rows_to_list(moods),
            'top_locations':     rows_to_list(locations),
            'writing_streak':    streak,
            'traveller_type':    _get_traveller_type(counts)
        })

    finally:
        db.close()


def _get_traveller_type(counts):
    """Derive traveller personality from entry types."""
    if not counts:
        return 'The Explorer'
    top = max(counts, key=counts.get)
    types_map = {
        'food':    'The Foodie 🍜',
        'place':   'The Explorer 📍',
        'expense': 'The Budget Master 💰',
        'memory':  'The Memory Keeper ✨',
        'note':    'The Storyteller 📝'
    }
    return types_map.get(top, 'The Traveller ✈️')


# ════════════════════════════════════════════════════════════
# SEARCH
# ════════════════════════════════════════════════════════════

@diary_bp.route('/api/diary/search', methods=['GET'])
@require_auth
def search_entries():
    """Full-text search across all diary entries."""
    q       = request.args.get('q', '').strip()
    trip_id = request.args.get('trip_id', type=int)

    if not q or len(q) < 2:
        return jsonify({'success': False, 'error': 'Search query too short'}), 400

    db = get_db()
    try:
        like   = '%' + q + '%'
        query  = '''
            SELECT * FROM diary_entries
            WHERE user_id = ? AND is_deleted = 0
            AND (text LIKE ? OR location LIKE ? OR category LIKE ? OR tags LIKE ?)
        '''
        params = [g.user_id, like, like, like, like]

        if trip_id:
            query  += ' AND trip_id = ?'
            params.append(trip_id)

        query  += ' ORDER BY created_at DESC LIMIT 30'

        entries = rows_to_list(db.execute(query, params).fetchall())

        return jsonify({'success': True, 'entries': entries, 'count': len(entries)})
    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════

@diary_bp.route('/api/diary/export', methods=['GET'])
@require_auth
def export_diary():
    """Export full diary as structured JSON or plain text."""
    trip_id    = request.args.get('trip_id', type=int)
    fmt        = request.args.get('format', 'json')

    db = get_db()
    try:
        query  = 'SELECT * FROM diary_entries WHERE user_id = ? AND is_deleted = 0'
        params = [g.user_id]

        if trip_id:
            query  += ' AND trip_id = ?'
            params.append(trip_id)

        query  += ' ORDER BY created_at ASC'
        entries = rows_to_list(db.execute(query, params).fetchall())

        trip = None
        if trip_id:
            trip = row_to_dict(db.execute(
                'SELECT * FROM diary_trips WHERE id = ?', (trip_id,)
            ).fetchone())

        if fmt == 'text':
            lines = []
            if trip:
                lines.append('═' * 50)
                lines.append('YAPLY TRAVEL DIARY')
                lines.append(trip.get('destination', 'My Trip'))
                lines.append(trip.get('start_date', '') + ' → ' + trip.get('end_date', ''))
                lines.append('═' * 50)
                lines.append('')

            for e in entries:
                lines.append(
                    '[' + e.get('created_at', '')[:10] + '] ' +
                    e.get('entry_type', '').upper() +
                    ((' · ' + e.get('location', '')) if e.get('location') else '') +
                    ' ' + (e.get('mood') or '')
                )
                if e.get('text'):
                    lines.append(e['text'])
                if e.get('amount'):
                    lines.append('Amount: ' + (e.get('currency') or '') + ' ' + str(e['amount']))
                lines.append('')

            from flask import Response
            return Response(
                '\n'.join(lines),
                mimetype='text/plain',
                headers={'Content-Disposition': 'attachment; filename="yaply_diary.txt"'}
            )
        else:
            from flask import Response
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'trip':        trip,
                'entries':     entries,
                'total':       len(entries)
            }
            return Response(
                json.dumps(export_data, indent=2, ensure_ascii=False),
                mimetype='application/json',
                headers={'Content-Disposition': 'attachment; filename="yaply_diary.json"'}
            )

    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# AI DIARY — Auto-generate a beautiful journal from entries
# ════════════════════════════════════════════════════════════

@diary_bp.route('/api/diary/ai-journal', methods=['POST'])
@require_auth
def generate_ai_journal():
    """Use Groq to turn raw diary entries into a beautiful travel story."""
    data    = request.get_json() or {}
    trip_id = data.get('trip_id')
    style   = data.get('style', 'storytelling')  # storytelling / poetic / factual

    db = get_db()
    try:
        entries = rows_to_list(db.execute('''
            SELECT entry_type, text, mood, location, amount, currency,
                   category, day_number, created_at
            FROM diary_entries
            WHERE user_id = ? AND is_deleted = 0
            ''' + (' AND trip_id = ?' if trip_id else '') + '''
            ORDER BY created_at ASC
            LIMIT 40
        ''', [g.user_id] + ([trip_id] if trip_id else [])).fetchall())

        trip = None
        if trip_id:
            trip = row_to_dict(db.execute(
                'SELECT * FROM diary_trips WHERE id = ? AND user_id = ?',
                (trip_id, g.user_id)
            ).fetchone())

        if not entries:
            return jsonify({'success': False, 'error': 'No entries to generate journal from'}), 400

        # Build context for Groq
        entry_lines = []
        for e in entries:
            line = 'Day ' + str(e.get('day_number', 1))
            line += ' [' + (e.get('entry_type') or 'note').upper() + ']'
            if e.get('location'):
                line += ' at ' + e['location']
            line += ' (' + (e.get('mood') or '') + '): ' + (e.get('text') or '')
            if e.get('amount'):
                line += ' [Spent: ' + str(e.get('currency', '₹')) + str(e['amount']) + ' on ' + (e.get('category') or '') + ']'
            entry_lines.append(line)

        dest = (trip or {}).get('destination', 'the destination')

        prompt = (
            'You are a talented travel writer. Based on these raw diary notes from a trip to ' +
            dest + ', write a beautiful ' + style + ' travel journal.\n\n'
            'Raw diary entries:\n' +
            '\n'.join(entry_lines) + '\n\n'
            'Write a vivid, emotional, first-person travel journal that:\n'
            '- Brings the trip to life with sensory details\n'
            '- Groups entries naturally by day\n'
            '- Includes real places, foods and moments from the notes\n'
            '- Has a poetic title and tagline\n'
            '- Ends with a reflection on what the trip meant\n\n'
            'Return ONLY valid JSON:\n'
            '{\n'
            '  "title": "evocative trip title",\n'
            '  "tagline": "one poetic line",\n'
            '  "chapters": [\n'
            '    {\n'
            '      "day": 1,\n'
            '      "chapter_title": "title",\n'
            '      "story": "2-3 vivid paragraphs",\n'
            '      "highlight": "best moment of the day",\n'
            '      "mood": "emoji"\n'
            '    }\n'
            '  ],\n'
            '  "closing": "emotional closing paragraph",\n'
            '  "best_memory": "single best moment",\n'
            '  "lesson": "what this trip taught you",\n'
            '  "quote": "a fitting travel quote"\n'
            '}'
        )

        # Import your existing groq_client
        from app import groq_client
        response = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {'role': 'system', 'content': 'You are a talented travel writer. Return ONLY valid JSON.'},
                {'role': 'user',   'content': prompt}
            ],
            temperature=0.75,
            max_tokens=3000
        )

        result = response.choices[0].message.content.strip()
        if '```' in result:
            parts = result.split('```')
            for part in parts:
                if '{' in part:
                    result = part
                    if result.startswith('json'):
                        result = result[4:]
                    break
        start = result.find('{')
        end   = result.rfind('}') + 1
        if start != -1 and end > start:
            result = result[start:end]

        journal = json.loads(result)

        return jsonify({'success': True, 'journal': journal})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()