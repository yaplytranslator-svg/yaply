# ═══════════════════════════════════════════════════════════════
# YAPLY GROUPS — Complete Backend
# File: groups.py
# Add to your project and import into app.py
# ═══════════════════════════════════════════════════════════════
#
# HOW TO USE:
# 1. Save this file as groups.py in your project root
# 2. In app.py add at the top:
#       from groups import groups_bp, init_groups_db
# 3. In app.py after app = Flask(__name__) add:
#       app.register_blueprint(groups_bp)
# 4. In app.py inside your DB init function call:
#       init_groups_db()
# 5. Add to requirements.txt:
#       flask-socketio
#       simple-websocket
# ═══════════════════════════════════════════════════════════════

import os
import json
import string
import random
import sqlite3
from datetime import datetime, timedelta
from functools import wraps
from flask import Blueprint, request, jsonify, g
from flask_socketio import SocketIO, emit, join_room, leave_room

# ── Blueprint ────────────────────────────────────────────────
groups_bp = Blueprint('groups', __name__)

# ── DB path (same as your existing yaply.db) ─────────────────
DB_PATH = os.environ.get('DB_PATH', 'yaply.db')

# ── Member colors & emojis ───────────────────────────────────
MEMBER_COLORS = [
    '#00D4AA', '#FF6B4A', '#FFB547', '#4A9EFF',
    '#A78BFA', '#34D399', '#F472B6', '#FB923C'
]

MEMBER_EMOJIS = [
    '🧑', '👩', '👨‍🦱', '👩‍🦰', '🧔', '👩‍🦳',
    '🧑‍🦱', '👨‍🦰', '🧑‍🦲', '👩‍🦲'
]

# ════════════════════════════════════════════════════════════
# DATABASE SETUP
# ════════════════════════════════════════════════════════════

def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA foreign_keys=ON')
    return db


def init_groups_db():
    """Create all Groups tables. Call once on app startup."""
    db = get_db()
    db.executescript('''
        -- Groups
        CREATE TABLE IF NOT EXISTS travel_groups (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            code        TEXT    NOT NULL UNIQUE,
            description TEXT,
            trip_dest   TEXT,
            trip_start  TEXT,
            trip_end    TEXT,
            created_by  INTEGER NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active   INTEGER DEFAULT 1,
            max_members INTEGER DEFAULT 20
        );

        -- Group members
        CREATE TABLE IF NOT EXISTS group_members (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id    INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            nickname    TEXT,
            emoji       TEXT    DEFAULT '🧑',
            color       TEXT    DEFAULT '#00D4AA',
            role        TEXT    DEFAULT 'member',
            joined_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active   INTEGER DEFAULT 1,
            location_sharing INTEGER DEFAULT 0,
            UNIQUE(group_id, user_id),
            FOREIGN KEY(group_id) REFERENCES travel_groups(id)
        );

        -- Messages
        CREATE TABLE IF NOT EXISTS group_messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id    INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            msg_type    TEXT    DEFAULT 'text',
            content     TEXT    NOT NULL,
            metadata    TEXT,
            sent_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_deleted  INTEGER DEFAULT 0,
            reply_to    INTEGER,
            FOREIGN KEY(group_id) REFERENCES travel_groups(id)
        );

        -- Live locations
        CREATE TABLE IF NOT EXISTS live_locations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL UNIQUE,
            group_id    INTEGER NOT NULL,
            lat         REAL    NOT NULL,
            lng         REAL    NOT NULL,
            accuracy    REAL,
            place_name  TEXT,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_sharing  INTEGER DEFAULT 1
        );

        -- Meetups
        CREATE TABLE IF NOT EXISTS meetups (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id    INTEGER NOT NULL,
            created_by  INTEGER NOT NULL,
            title       TEXT    NOT NULL,
            place_name  TEXT    NOT NULL,
            place_lat   REAL,
            place_lng   REAL,
            meet_time   TEXT    NOT NULL,
            note        TEXT,
            status      TEXT    DEFAULT 'pending',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(group_id) REFERENCES travel_groups(id)
        );

        -- Meetup RSVPs
        CREATE TABLE IF NOT EXISTS meetup_rsvps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            meetup_id   INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            rsvp        TEXT    DEFAULT 'pending',
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(meetup_id, user_id),
            FOREIGN KEY(meetup_id) REFERENCES meetups(id)
        );

        -- Notifications
        CREATE TABLE IF NOT EXISTS group_notifications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id    INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            type        TEXT    NOT NULL,
            title       TEXT    NOT NULL,
            body        TEXT,
            data        TEXT,
            is_read     INTEGER DEFAULT 0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- SOS alerts
        CREATE TABLE IF NOT EXISTS sos_alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id    INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            sos_type    TEXT    NOT NULL,
            message     TEXT,
            lat         REAL,
            lng         REAL,
            resolved    INTEGER DEFAULT 0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_gm_group  ON group_members(group_id);
        CREATE INDEX IF NOT EXISTS idx_gm_user   ON group_members(user_id);
        CREATE INDEX IF NOT EXISTS idx_msg_group ON group_messages(group_id);
        CREATE INDEX IF NOT EXISTS idx_msg_sent  ON group_messages(sent_at);
        CREATE INDEX IF NOT EXISTS idx_loc_group ON live_locations(group_id);
        CREATE INDEX IF NOT EXISTS idx_meet_grp  ON meetups(group_id);
        CREATE INDEX IF NOT EXISTS idx_notif_usr ON group_notifications(user_id);
    ''')
    db.commit()
    db.close()
    print('[Groups] Database initialized ✓')


# ════════════════════════════════════════════════════════════
# AUTH HELPER
# ════════════════════════════════════════════════════════════

def require_auth(f):
    """Use your existing JWT auth. Attach user_id to g."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
        if not token:
            token = request.headers.get('X-Auth-Token')

        if not token:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401

        try:
            import jwt as pyjwt
            secret = os.environ.get('JWT_SECRET', 'yaply-secret-key')
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

def generate_group_code():
    """Generate a unique 8-char group code like YAP-7K2M."""
    while True:
        part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        code = 'YAP-' + part
        db = get_db()
        exists = db.execute(
            'SELECT id FROM travel_groups WHERE code = ?', (code,)
        ).fetchone()
        db.close()
        if not exists:
            return code


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two GPS points in km."""
    import math
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lon/2)**2)
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)


def get_member_info(db, group_id, user_id):
    """Get member details including user profile."""
    return db.execute('''
        SELECT gm.*, u.name, u.email
        FROM group_members gm
        JOIN users u ON u.id = gm.user_id
        WHERE gm.group_id = ? AND gm.user_id = ? AND gm.is_active = 1
    ''', (group_id, user_id)).fetchone()


def is_member(db, group_id, user_id):
    return get_member_info(db, group_id, user_id) is not None


def assign_member_appearance(db, group_id):
    """Pick the next unused color/emoji for a new member."""
    used = db.execute(
        'SELECT emoji, color FROM group_members WHERE group_id = ? AND is_active = 1',
        (group_id,)
    ).fetchall()
    used_emojis  = {r['emoji'] for r in used}
    used_colors  = {r['color'] for r in used}

    emoji = next((e for e in MEMBER_EMOJIS if e not in used_emojis), MEMBER_EMOJIS[0])
    color = next((c for c in MEMBER_COLORS if c not in used_colors), MEMBER_COLORS[0])
    return emoji, color


def push_notification(db, group_id, user_id, notif_type, title, body, data=None):
    """Store a notification for a user."""
    db.execute('''
        INSERT INTO group_notifications
            (group_id, user_id, type, title, body, data)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (group_id, user_id, notif_type, title, body,
          json.dumps(data) if data else None))


def notify_all_members(db, group_id, exclude_user, notif_type, title, body, data=None):
    """Send notification to all active members except the sender."""
    members = db.execute('''
        SELECT user_id FROM group_members
        WHERE group_id = ? AND is_active = 1 AND user_id != ?
    ''', (group_id, exclude_user)).fetchall()

    for m in members:
        push_notification(db, group_id, m['user_id'], notif_type, title, body, data)


def row_to_dict(row):
    if row is None:
        return None
    return dict(row)


def rows_to_list(rows):
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════
# GROUP MANAGEMENT ROUTES
# ════════════════════════════════════════════════════════════

@groups_bp.route('/api/groups', methods=['POST'])
@require_auth
def create_group():
    """Create a new travel group."""
    data      = request.get_json() or {}
    name      = (data.get('name') or '').strip()
    trip_dest = data.get('trip_dest', '')
    trip_start= data.get('trip_start', '')
    trip_end  = data.get('trip_end', '')
    description = data.get('description', '')

    if not name:
        return jsonify({'success': False, 'error': 'Group name required'}), 400

    db   = get_db()
    code = generate_group_code()
    emoji, color = assign_member_appearance(db, 0)

    try:
        db.execute('''
            INSERT INTO travel_groups
                (name, code, description, trip_dest, trip_start, trip_end, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, code, description, trip_dest, trip_start, trip_end, g.user_id))

        group_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]

        # Auto-join creator as admin
        db.execute('''
            INSERT INTO group_members
                (group_id, user_id, role, emoji, color, location_sharing)
            VALUES (?, ?, 'admin', ?, ?, 1)
        ''', (group_id, g.user_id, emoji, color))

        db.commit()

        group = row_to_dict(db.execute(
            'SELECT * FROM travel_groups WHERE id = ?', (group_id,)
        ).fetchone())

        return jsonify({
            'success':  True,
            'group':    group,
            'group_id': group_id,
            'code':     code,
            'message':  'Group created! Share the code with your squad.'
        }), 201

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@groups_bp.route('/api/groups/join', methods=['POST'])
@require_auth
def join_group():
    """Join a group using the invite code."""
    data = request.get_json() or {}
    code = (data.get('code') or '').strip().upper()

    if not code:
        return jsonify({'success': False, 'error': 'Group code required'}), 400

    db = get_db()
    try:
        group = db.execute(
            'SELECT * FROM travel_groups WHERE code = ? AND is_active = 1',
            (code,)
        ).fetchone()

        if not group:
            return jsonify({'success': False, 'error': 'Invalid group code'}), 404

        group_id = group['id']

        # Already a member?
        existing = db.execute(
            'SELECT * FROM group_members WHERE group_id = ? AND user_id = ?',
            (group_id, g.user_id)
        ).fetchone()

        if existing and existing['is_active']:
            return jsonify({
                'success':  True,
                'group_id': group_id,
                'code':     code,
                'message':  'Already a member!',
                'already_member': True
            })

        # Check max members
        count = db.execute(
            'SELECT COUNT(*) as c FROM group_members WHERE group_id = ? AND is_active = 1',
            (group_id,)
        ).fetchone()['c']

        if count >= group['max_members']:
            return jsonify({'success': False, 'error': 'Group is full'}), 400

        emoji, color = assign_member_appearance(db, group_id)

        if existing:
            db.execute(
                'UPDATE group_members SET is_active=1, emoji=?, color=? WHERE group_id=? AND user_id=?',
                (emoji, color, group_id, g.user_id)
            )
        else:
            db.execute('''
                INSERT INTO group_members
                    (group_id, user_id, role, emoji, color)
                VALUES (?, ?, 'member', ?, ?)
            ''', (group_id, g.user_id, emoji, color))

        # Notify existing members
        user = db.execute('SELECT name FROM users WHERE id = ?', (g.user_id,)).fetchone()
        user_name = user['name'] if user else 'Someone'

        notify_all_members(
            db, group_id, g.user_id,
            'member_joined',
            user_name + ' joined the squad! 🎉',
            user_name + ' joined ' + group['name'],
            {'group_id': group_id}
        )

        db.commit()

        return jsonify({
            'success':  True,
            'group_id': group_id,
            'group':    row_to_dict(group),
            'emoji':    emoji,
            'color':    color,
            'message':  'Joined ' + group['name'] + '!'
        })

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@groups_bp.route('/api/groups', methods=['GET'])
@require_auth
def get_my_groups():
    """Get all groups the current user belongs to."""
    db = get_db()
    try:
        groups = db.execute('''
            SELECT tg.*, gm.emoji, gm.color, gm.role,
                   (SELECT COUNT(*) FROM group_members gm2
                    WHERE gm2.group_id = tg.id AND gm2.is_active = 1) as member_count,
                   (SELECT COUNT(*) FROM group_messages gm3
                    WHERE gm3.group_id = tg.id) as message_count,
                   (SELECT content FROM group_messages gm4
                    WHERE gm4.group_id = tg.id
                    ORDER BY sent_at DESC LIMIT 1) as last_message,
                   (SELECT COUNT(*) FROM group_notifications gn
                    WHERE gn.group_id = tg.id AND gn.user_id = ? AND gn.is_read = 0) as unread
            FROM travel_groups tg
            JOIN group_members gm ON gm.group_id = tg.id
            WHERE gm.user_id = ? AND gm.is_active = 1 AND tg.is_active = 1
            ORDER BY tg.created_at DESC
        ''', (g.user_id, g.user_id)).fetchall()

        return jsonify({'success': True, 'groups': rows_to_list(groups)})

    finally:
        db.close()


@groups_bp.route('/api/groups/<int:group_id>', methods=['GET'])
@require_auth
def get_group(group_id):
    """Get full group details including members and recent messages."""
    db = get_db()
    try:
        if not is_member(db, group_id, g.user_id):
            return jsonify({'success': False, 'error': 'Not a member'}), 403

        group = db.execute(
            'SELECT * FROM travel_groups WHERE id = ?', (group_id,)
        ).fetchone()

        if not group:
            return jsonify({'success': False, 'error': 'Group not found'}), 404

        # Members with live location
        members = db.execute('''
            SELECT gm.*, u.name, u.email,
                   ll.lat, ll.lng, ll.place_name, ll.updated_at as loc_updated,
                   ll.is_sharing as is_sharing_location
            FROM group_members gm
            JOIN users u ON u.id = gm.user_id
            LEFT JOIN live_locations ll ON ll.user_id = gm.user_id AND ll.group_id = ?
            WHERE gm.group_id = ? AND gm.is_active = 1
            ORDER BY gm.role DESC, gm.joined_at
        ''', (group_id, group_id)).fetchall()

        # Recent messages
        messages = db.execute('''
            SELECT gm.*, u.name as sender_name,
                   mem.emoji as sender_emoji, mem.color as sender_color
            FROM group_messages gm
            JOIN users u ON u.id = gm.user_id
            JOIN group_members mem ON mem.user_id = gm.user_id AND mem.group_id = gm.group_id
            WHERE gm.group_id = ? AND gm.is_deleted = 0
            ORDER BY gm.sent_at DESC
            LIMIT 50
        ''', (group_id,)).fetchall()

        # Upcoming meetups
        meetups = db.execute('''
            SELECT m.*,
                   (SELECT COUNT(*) FROM meetup_rsvps mr WHERE mr.meetup_id = m.id AND mr.rsvp = 'going') as going_count,
                   (SELECT rsvp FROM meetup_rsvps mr WHERE mr.meetup_id = m.id AND mr.user_id = ?) as my_rsvp
            FROM meetups m
            WHERE m.group_id = ? AND m.status != 'cancelled'
            ORDER BY m.meet_time ASC
        ''', (g.user_id, group_id)).fetchall()

        return jsonify({
            'success':  True,
            'group':    row_to_dict(group),
            'members':  rows_to_list(members),
            'messages': rows_to_list(reversed(list(messages))),
            'meetups':  rows_to_list(meetups)
        })

    finally:
        db.close()


@groups_bp.route('/api/groups/<int:group_id>/leave', methods=['POST'])
@require_auth
def leave_group(group_id):
    """Leave a group."""
    db = get_db()
    try:
        member = get_member_info(db, group_id, g.user_id)
        if not member:
            return jsonify({'success': False, 'error': 'Not a member'}), 403

        # If last admin, transfer or delete
        if member['role'] == 'admin':
            other_admin = db.execute('''
                SELECT user_id FROM group_members
                WHERE group_id = ? AND user_id != ? AND role = 'admin' AND is_active = 1
            ''', (group_id, g.user_id)).fetchone()

            if not other_admin:
                # Promote next member
                next_member = db.execute('''
                    SELECT user_id FROM group_members
                    WHERE group_id = ? AND user_id != ? AND is_active = 1
                    ORDER BY joined_at LIMIT 1
                ''', (group_id, g.user_id)).fetchone()

                if next_member:
                    db.execute(
                        'UPDATE group_members SET role = ? WHERE group_id = ? AND user_id = ?',
                        ('admin', group_id, next_member['user_id'])
                    )

        db.execute(
            'UPDATE group_members SET is_active = 0 WHERE group_id = ? AND user_id = ?',
            (group_id, g.user_id)
        )

        # Stop location sharing
        db.execute(
            'UPDATE live_locations SET is_sharing = 0 WHERE group_id = ? AND user_id = ?',
            (group_id, g.user_id)
        )

        db.commit()
        return jsonify({'success': True, 'message': 'Left the group'})

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# MESSAGING ROUTES
# ════════════════════════════════════════════════════════════

@groups_bp.route('/api/groups/<int:group_id>/messages', methods=['GET'])
@require_auth
def get_messages(group_id):
    """Get messages with pagination."""
    db = get_db()
    try:
        if not is_member(db, group_id, g.user_id):
            return jsonify({'success': False, 'error': 'Not a member'}), 403

        before_id = request.args.get('before_id', type=int)
        limit     = min(int(request.args.get('limit', 50)), 100)

        query  = '''
            SELECT gm.*, u.name as sender_name,
                   mem.emoji as sender_emoji, mem.color as sender_color
            FROM group_messages gm
            JOIN users u ON u.id = gm.user_id
            JOIN group_members mem ON mem.user_id = gm.user_id AND mem.group_id = gm.group_id
            WHERE gm.group_id = ? AND gm.is_deleted = 0
        '''
        params = [group_id]

        if before_id:
            query  += ' AND gm.id < ?'
            params.append(before_id)

        query  += ' ORDER BY gm.sent_at DESC LIMIT ?'
        params.append(limit)

        messages = db.execute(query, params).fetchall()
        msgs_list = rows_to_list(reversed(list(messages)))

        # Parse metadata
        for m in msgs_list:
            if m.get('metadata'):
                try:
                    m['metadata'] = json.loads(m['metadata'])
                except Exception:
                    pass

        return jsonify({'success': True, 'messages': msgs_list})

    finally:
        db.close()


@groups_bp.route('/api/groups/<int:group_id>/messages', methods=['POST'])
@require_auth
def send_message(group_id):
    """Send a message to the group."""
    db = get_db()
    try:
        if not is_member(db, group_id, g.user_id):
            return jsonify({'success': False, 'error': 'Not a member'}), 403

        data     = request.get_json() or {}
        msg_type = data.get('type', 'text')
        content  = (data.get('content') or data.get('text') or '').strip()
        metadata = data.get('metadata')
        reply_to = data.get('reply_to')

        if not content and msg_type == 'text':
            return jsonify({'success': False, 'error': 'Message content required'}), 400

        # For non-text types, content can be empty
        if not content:
            content = msg_type

        db.execute('''
            INSERT INTO group_messages
                (group_id, user_id, msg_type, content, metadata, reply_to)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (group_id, g.user_id, msg_type, content,
              json.dumps(metadata) if metadata else None, reply_to))

        msg_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]

        # Get full message
        msg = row_to_dict(db.execute('''
            SELECT gm.*, u.name as sender_name,
                   mem.emoji as sender_emoji, mem.color as sender_color
            FROM group_messages gm
            JOIN users u ON u.id = gm.user_id
            JOIN group_members mem ON mem.user_id = gm.user_id AND mem.group_id = gm.group_id
            WHERE gm.id = ?
        ''', (msg_id,)).fetchone())

        if msg.get('metadata'):
            try:
                msg['metadata'] = json.loads(msg['metadata'])
            except Exception:
                pass

        db.commit()

        return jsonify({'success': True, 'message': msg}), 201

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@groups_bp.route('/api/groups/<int:group_id>/messages/<int:msg_id>', methods=['DELETE'])
@require_auth
def delete_message(group_id, msg_id):
    """Delete own message (soft delete)."""
    db = get_db()
    try:
        msg = db.execute(
            'SELECT * FROM group_messages WHERE id = ? AND group_id = ?',
            (msg_id, group_id)
        ).fetchone()

        if not msg:
            return jsonify({'success': False, 'error': 'Message not found'}), 404

        # Only own messages or admin can delete
        member = get_member_info(db, group_id, g.user_id)
        if msg['user_id'] != g.user_id and (not member or member['role'] != 'admin'):
            return jsonify({'success': False, 'error': 'Cannot delete this message'}), 403

        db.execute(
            'UPDATE group_messages SET is_deleted = 1, content = ? WHERE id = ?',
            ('[Message deleted]', msg_id)
        )
        db.commit()
        return jsonify({'success': True})

    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# LIVE LOCATION ROUTES
# ════════════════════════════════════════════════════════════

@groups_bp.route('/api/groups/<int:group_id>/location', methods=['POST'])
@require_auth
def update_location(group_id):
    """Update user's live location for a group."""
    db = get_db()
    try:
        if not is_member(db, group_id, g.user_id):
            return jsonify({'success': False, 'error': 'Not a member'}), 403

        data       = request.get_json() or {}
        lat        = data.get('lat')
        lng        = data.get('lng')
        accuracy   = data.get('accuracy')
        place_name = data.get('place_name', '')
        is_sharing = data.get('is_sharing', True)

        if lat is None or lng is None:
            return jsonify({'success': False, 'error': 'lat and lng required'}), 400

        # Upsert location
        existing = db.execute(
            'SELECT id FROM live_locations WHERE user_id = ? AND group_id = ?',
            (g.user_id, group_id)
        ).fetchone()

        if existing:
            db.execute('''
                UPDATE live_locations
                SET lat=?, lng=?, accuracy=?, place_name=?,
                    is_sharing=?, updated_at=CURRENT_TIMESTAMP
                WHERE user_id=? AND group_id=?
            ''', (lat, lng, accuracy, place_name, 1 if is_sharing else 0,
                  g.user_id, group_id))
        else:
            db.execute('''
                INSERT INTO live_locations
                    (user_id, group_id, lat, lng, accuracy, place_name, is_sharing)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (g.user_id, group_id, lat, lng, accuracy, place_name,
                  1 if is_sharing else 0))

        # Update member sharing flag
        db.execute(
            'UPDATE group_members SET location_sharing=? WHERE group_id=? AND user_id=?',
            (1 if is_sharing else 0, group_id, g.user_id)
        )

        db.commit()
        return jsonify({'success': True, 'updated_at': datetime.now().isoformat()})

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@groups_bp.route('/api/groups/<int:group_id>/locations', methods=['GET'])
@require_auth
def get_locations(group_id):
    """Get all live locations for a group."""
    db = get_db()
    try:
        if not is_member(db, group_id, g.user_id):
            return jsonify({'success': False, 'error': 'Not a member'}), 403

        locations = db.execute('''
            SELECT ll.*, u.name, gm.emoji, gm.color, gm.nickname
            FROM live_locations ll
            JOIN users u ON u.id = ll.user_id
            JOIN group_members gm ON gm.user_id = ll.user_id AND gm.group_id = ll.group_id
            WHERE ll.group_id = ? AND ll.is_sharing = 1
        ''', (group_id,)).fetchall()

        locs = rows_to_list(locations)

        # Calculate distances from requesting user's location
        my_loc = db.execute(
            'SELECT lat, lng FROM live_locations WHERE user_id = ? AND group_id = ?',
            (g.user_id, group_id)
        ).fetchone()

        if my_loc:
            for loc in locs:
                if loc['user_id'] != g.user_id and loc.get('lat') and loc.get('lng'):
                    loc['distance_km'] = haversine_km(
                        my_loc['lat'], my_loc['lng'],
                        loc['lat'], loc['lng']
                    )

        # Mark stale locations (>10 minutes old)
        now = datetime.now()
        for loc in locs:
            if loc.get('updated_at'):
                try:
                    updated = datetime.fromisoformat(loc['updated_at'].replace('Z',''))
                    loc['is_stale'] = (now - updated).seconds > 600
                    loc['mins_ago'] = int((now - updated).seconds / 60)
                except Exception:
                    loc['is_stale'] = False
                    loc['mins_ago'] = 0

        return jsonify({'success': True, 'locations': locs})

    finally:
        db.close()


@groups_bp.route('/api/groups/<int:group_id>/location/stop', methods=['POST'])
@require_auth
def stop_location_sharing(group_id):
    """Stop sharing location."""
    db = get_db()
    try:
        db.execute(
            'UPDATE live_locations SET is_sharing=0 WHERE user_id=? AND group_id=?',
            (g.user_id, group_id)
        )
        db.execute(
            'UPDATE group_members SET location_sharing=0 WHERE user_id=? AND group_id=?',
            (g.user_id, group_id)
        )
        db.commit()
        return jsonify({'success': True, 'message': 'Location sharing stopped'})
    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# MEETUP ROUTES
# ════════════════════════════════════════════════════════════

@groups_bp.route('/api/groups/<int:group_id>/meetups', methods=['GET'])
@require_auth
def get_meetups(group_id):
    """Get all meetups for a group."""
    db = get_db()
    try:
        if not is_member(db, group_id, g.user_id):
            return jsonify({'success': False, 'error': 'Not a member'}), 403

        meetups = db.execute('''
            SELECT m.*,
                   u.name as creator_name,
                   (SELECT COUNT(*) FROM meetup_rsvps mr
                    WHERE mr.meetup_id = m.id AND mr.rsvp = 'going') as going_count,
                   (SELECT COUNT(*) FROM meetup_rsvps mr
                    WHERE mr.meetup_id = m.id AND mr.rsvp = 'not_going') as not_going_count,
                   (SELECT rsvp FROM meetup_rsvps mr
                    WHERE mr.meetup_id = m.id AND mr.user_id = ?) as my_rsvp
            FROM meetups m
            JOIN users u ON u.id = m.created_by
            WHERE m.group_id = ?
            ORDER BY m.meet_time ASC
        ''', (g.user_id, group_id)).fetchall()

        result = []
        for m in rows_to_list(meetups):
            # Get RSVP list
            rsvps = db.execute('''
                SELECT mr.rsvp, u.name, gm.emoji, gm.color
                FROM meetup_rsvps mr
                JOIN users u ON u.id = mr.user_id
                JOIN group_members gm ON gm.user_id = mr.user_id AND gm.group_id = ?
                WHERE mr.meetup_id = ?
            ''', (group_id, m['id'])).fetchall()

            m['rsvps'] = rows_to_list(rsvps)
            result.append(m)

        return jsonify({'success': True, 'meetups': result})

    finally:
        db.close()


@groups_bp.route('/api/groups/<int:group_id>/meetups', methods=['POST'])
@require_auth
def create_meetup(group_id):
    """Create a meetup and notify all members."""
    db = get_db()
    try:
        if not is_member(db, group_id, g.user_id):
            return jsonify({'success': False, 'error': 'Not a member'}), 403

        data       = request.get_json() or {}
        title      = (data.get('title') or '').strip()
        place_name = (data.get('place_name') or '').strip()
        meet_time  = (data.get('meet_time') or '').strip()
        place_lat  = data.get('place_lat')
        place_lng  = data.get('place_lng')
        note       = data.get('note', '')

        if not title or not place_name or not meet_time:
            return jsonify({'success': False, 'error': 'title, place_name and meet_time required'}), 400

        db.execute('''
            INSERT INTO meetups
                (group_id, created_by, title, place_name, place_lat, place_lng, meet_time, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (group_id, g.user_id, title, place_name, place_lat, place_lng, meet_time, note))

        meetup_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]

        # Auto-RSVP creator as going
        db.execute(
            'INSERT INTO meetup_rsvps (meetup_id, user_id, rsvp) VALUES (?, ?, ?)',
            (meetup_id, g.user_id, 'going')
        )

        # Get creator name
        user = db.execute('SELECT name FROM users WHERE id = ?', (g.user_id,)).fetchone()
        creator = user['name'] if user else 'Someone'

        # Notify all members
        notify_all_members(
            db, group_id, g.user_id,
            'meetup_created',
            '📍 New meetup: ' + title,
            creator + ' proposed a meetup at ' + place_name,
            {'meetup_id': meetup_id, 'group_id': group_id}
        )

        # Auto-send as group message
        db.execute('''
            INSERT INTO group_messages
                (group_id, user_id, msg_type, content, metadata)
            VALUES (?, ?, 'meetup', ?, ?)
        ''', (group_id, g.user_id, title,
              json.dumps({
                  'meetup_id': meetup_id,
                  'title':     title,
                  'place':     place_name,
                  'time':      meet_time,
                  'note':      note
              })))

        db.commit()

        meetup = row_to_dict(db.execute(
            'SELECT * FROM meetups WHERE id = ?', (meetup_id,)
        ).fetchone())

        return jsonify({
            'success': True,
            'meetup':  meetup,
            'message': 'Meetup created! Squad has been notified.'
        }), 201

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@groups_bp.route('/api/groups/<int:group_id>/meetups/<int:meetup_id>/rsvp', methods=['POST'])
@require_auth
def rsvp_meetup(group_id, meetup_id):
    """RSVP to a meetup: going / not_going / maybe."""
    db = get_db()
    try:
        if not is_member(db, group_id, g.user_id):
            return jsonify({'success': False, 'error': 'Not a member'}), 403

        data = request.get_json() or {}
        rsvp = data.get('rsvp', 'going')

        if rsvp not in ('going', 'not_going', 'maybe'):
            return jsonify({'success': False, 'error': 'rsvp must be going/not_going/maybe'}), 400

        existing = db.execute(
            'SELECT id FROM meetup_rsvps WHERE meetup_id=? AND user_id=?',
            (meetup_id, g.user_id)
        ).fetchone()

        if existing:
            db.execute(
                'UPDATE meetup_rsvps SET rsvp=?, updated_at=CURRENT_TIMESTAMP WHERE meetup_id=? AND user_id=?',
                (rsvp, meetup_id, g.user_id)
            )
        else:
            db.execute(
                'INSERT INTO meetup_rsvps (meetup_id, user_id, rsvp) VALUES (?,?,?)',
                (meetup_id, g.user_id, rsvp)
            )

        # Notify meetup creator
        meetup = db.execute('SELECT * FROM meetups WHERE id=?', (meetup_id,)).fetchone()
        user   = db.execute('SELECT name FROM users WHERE id=?', (g.user_id,)).fetchone()
        if meetup and user and meetup['created_by'] != g.user_id:
            label = {'going': 'is going ✓', 'not_going': "can't make it", 'maybe': 'might come'}
            push_notification(
                db, group_id, meetup['created_by'],
                'meetup_rsvp',
                user['name'] + ' ' + label.get(rsvp, rsvp),
                'For: ' + (meetup['title'] or ''),
                {'meetup_id': meetup_id}
            )

        # If going, auto-update to confirmed if > 50% of members RSVPd
        total = db.execute(
            'SELECT COUNT(*) as c FROM group_members WHERE group_id=? AND is_active=1',
            (group_id,)
        ).fetchone()['c']

        going = db.execute(
            'SELECT COUNT(*) as c FROM meetup_rsvps WHERE meetup_id=? AND rsvp=?',
            (meetup_id, 'going')
        ).fetchone()['c']

        if going >= max(2, total // 2):
            db.execute(
                'UPDATE meetups SET status=? WHERE id=?',
                ('confirmed', meetup_id)
            )

        db.commit()

        # Return updated counts
        counts = db.execute('''
            SELECT
                SUM(CASE WHEN rsvp='going'     THEN 1 ELSE 0 END) as going,
                SUM(CASE WHEN rsvp='not_going' THEN 1 ELSE 0 END) as not_going,
                SUM(CASE WHEN rsvp='maybe'     THEN 1 ELSE 0 END) as maybe
            FROM meetup_rsvps WHERE meetup_id=?
        ''', (meetup_id,)).fetchone()

        return jsonify({
            'success': True,
            'rsvp':    rsvp,
            'counts':  row_to_dict(counts)
        })

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# SOS ROUTES
# ════════════════════════════════════════════════════════════

@groups_bp.route('/api/groups/<int:group_id>/sos', methods=['POST'])
@require_auth
def send_sos(group_id):
    """Send an SOS alert to the entire group."""
    db = get_db()
    try:
        if not is_member(db, group_id, g.user_id):
            return jsonify({'success': False, 'error': 'Not a member'}), 403

        data     = request.get_json() or {}
        sos_type = data.get('sos_type', 'general')
        message  = data.get('message', '')
        lat      = data.get('lat')
        lng      = data.get('lng')

        # Save SOS alert
        db.execute('''
            INSERT INTO sos_alerts
                (group_id, user_id, sos_type, message, lat, lng)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (group_id, g.user_id, sos_type, message, lat, lng))

        sos_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]

        user = db.execute('SELECT name FROM users WHERE id=?', (g.user_id,)).fetchone()
        name = user['name'] if user else 'A member'

        sos_messages = {
            'lost':    name + ' is lost and needs help!',
            'medical': name + ' has a medical emergency!',
            'safety':  name + ' needs immediate safety help!',
            'general': name + ' sent an SOS alert!'
        }

        full_message = sos_messages.get(sos_type, sos_messages['general'])
        if message:
            full_message += ' — ' + message

        # Send as group message
        db.execute('''
            INSERT INTO group_messages
                (group_id, user_id, msg_type, content, metadata)
            VALUES (?, ?, 'sos', ?, ?)
        ''', (group_id, g.user_id, full_message,
              json.dumps({
                  'sos_id':   sos_id,
                  'sos_type': sos_type,
                  'lat':      lat,
                  'lng':      lng
              })))

        # URGENT notifications to everyone
        notify_all_members(
            db, group_id, g.user_id,
            'sos',
            '🆘 EMERGENCY — ' + name,
            full_message,
            {'sos_id': sos_id, 'group_id': group_id, 'lat': lat, 'lng': lng}
        )

        db.commit()
        return jsonify({
            'success': True,
            'sos_id':  sos_id,
            'message': 'SOS sent to all squad members!'
        }), 201

    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@groups_bp.route('/api/groups/<int:group_id>/sos/<int:sos_id>/resolve', methods=['POST'])
@require_auth
def resolve_sos(group_id, sos_id):
    """Mark SOS as resolved."""
    db = get_db()
    try:
        db.execute(
            'UPDATE sos_alerts SET resolved=1 WHERE id=? AND group_id=?',
            (sos_id, group_id)
        )
        db.commit()
        return jsonify({'success': True, 'message': 'SOS resolved — glad you are safe!'})
    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# NOTIFICATION ROUTES
# ════════════════════════════════════════════════════════════

@groups_bp.route('/api/groups/notifications', methods=['GET'])
@require_auth
def get_notifications():
    """Get all unread notifications for the current user."""
    db = get_db()
    try:
        notifs = db.execute('''
            SELECT gn.*, tg.name as group_name
            FROM group_notifications gn
            JOIN travel_groups tg ON tg.id = gn.group_id
            WHERE gn.user_id = ?
            ORDER BY gn.created_at DESC
            LIMIT 50
        ''', (g.user_id,)).fetchall()

        result = []
        for n in rows_to_list(notifs):
            if n.get('data'):
                try:
                    n['data'] = json.loads(n['data'])
                except Exception:
                    pass
            result.append(n)

        unread_count = db.execute(
            'SELECT COUNT(*) as c FROM group_notifications WHERE user_id=? AND is_read=0',
            (g.user_id,)
        ).fetchone()['c']

        return jsonify({
            'success':      True,
            'notifications': result,
            'unread_count':  unread_count
        })

    finally:
        db.close()


@groups_bp.route('/api/groups/notifications/read', methods=['POST'])
@require_auth
def mark_notifications_read():
    """Mark all or specific notifications as read."""
    db = get_db()
    try:
        data    = request.get_json() or {}
        notif_ids = data.get('ids')

        if notif_ids:
            placeholders = ','.join(['?'] * len(notif_ids))
            db.execute(
                'UPDATE group_notifications SET is_read=1 WHERE id IN (' + placeholders + ') AND user_id=?',
                notif_ids + [g.user_id]
            )
        else:
            db.execute(
                'UPDATE group_notifications SET is_read=1 WHERE user_id=?',
                (g.user_id,)
            )

        db.commit()
        return jsonify({'success': True})
    finally:
        db.close()


# ════════════════════════════════════════════════════════════
# WEBSOCKET — Real-time everything
# ════════════════════════════════════════════════════════════
#
# Add this to your app.py AFTER creating the Flask app:
#
#   from flask_socketio import SocketIO
#   socketio = SocketIO(app, cors_allowed_origins="*",
#                       async_mode='gevent')
#
# Then register events like this in app.py:
#
#   from groups import register_socketio_events
#   register_socketio_events(socketio)
#
# And change your run command:
#   socketio.run(app, host='0.0.0.0', port=port)
# ════════════════════════════════════════════════════════════

def register_socketio_events(socketio):
    """Register all Socket.IO events for real-time Groups."""

    def auth_socket(f):
        """Authenticate WebSocket connections."""
        @wraps(f)
        def decorated(*args, **kwargs):
            from flask_socketio import disconnect
            token = (request.args.get('token') or
                     request.headers.get('Authorization', '').replace('Bearer ', ''))
            if not token:
                disconnect()
                return
            try:
                import jwt as pyjwt
                secret = os.environ.get('JWT_SECRET', 'yaply-secret-key')
                payload = pyjwt.decode(token, secret, algorithms=['HS256'])
                g.user_id = payload.get('user_id') or payload.get('id')
            except Exception:
                disconnect()
                return
            return f(*args, **kwargs)
        return decorated

    @socketio.on('connect')
    def on_connect():
        print('[WS] Client connected:', request.sid)

    @socketio.on('disconnect')
    def on_disconnect():
        print('[WS] Client disconnected:', request.sid)

    @socketio.on('join_group')
    def on_join_group(data):
        """Client joins a group room for real-time updates."""
        group_id = data.get('group_id')
        token    = data.get('token')
        if not group_id or not token:
            return

        try:
            import jwt as pyjwt
            secret  = os.environ.get('JWT_SECRET', 'yaply-secret-key')
            payload = pyjwt.decode(token, secret, algorithms=['HS256'])
            user_id = payload.get('user_id') or payload.get('id')

            db = get_db()
            if is_member(db, group_id, user_id):
                room = 'group_' + str(group_id)
                join_room(room)
                # Notify others
                emit('member_online', {
                    'user_id': user_id,
                    'group_id': group_id
                }, room=room, include_self=False)
                db.close()
        except Exception as e:
            print('[WS] join_group error:', e)

    @socketio.on('leave_group')
    def on_leave_group(data):
        group_id = data.get('group_id')
        if group_id:
            leave_room('group_' + str(group_id))

    @socketio.on('send_message')
    def on_send_message(data):
        """Real-time message broadcast."""
        group_id  = data.get('group_id')
        token     = data.get('token')
        content   = (data.get('content') or '').strip()
        msg_type  = data.get('type', 'text')
        metadata  = data.get('metadata')

        if not group_id or not content or not token:
            return

        try:
            import jwt as pyjwt
            secret  = os.environ.get('JWT_SECRET', 'yaply-secret-key')
            payload = pyjwt.decode(token, secret, algorithms=['HS256'])
            user_id = payload.get('user_id') or payload.get('id')

            db = get_db()
            if not is_member(db, group_id, user_id):
                db.close()
                return

            db.execute('''
                INSERT INTO group_messages
                    (group_id, user_id, msg_type, content, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (group_id, user_id, msg_type, content,
                  json.dumps(metadata) if metadata else None))

            msg_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]

            msg = row_to_dict(db.execute('''
                SELECT gm.*, u.name as sender_name,
                       mem.emoji as sender_emoji, mem.color as sender_color
                FROM group_messages gm
                JOIN users u ON u.id = gm.user_id
                JOIN group_members mem ON mem.user_id = gm.user_id AND mem.group_id = gm.group_id
                WHERE gm.id = ?
            ''', (msg_id,)).fetchone())

            if msg.get('metadata'):
                try:
                    msg['metadata'] = json.loads(msg['metadata'])
                except Exception:
                    pass

            db.commit()
            db.close()

            # Broadcast to entire group room
            emit('new_message', msg, room='group_' + str(group_id))

        except Exception as e:
            print('[WS] send_message error:', e)

    @socketio.on('location_update')
    def on_location_update(data):
        """Real-time location broadcast."""
        group_id   = data.get('group_id')
        token      = data.get('token')
        lat        = data.get('lat')
        lng        = data.get('lng')
        place_name = data.get('place_name', '')

        if not group_id or not token or lat is None or lng is None:
            return

        try:
            import jwt as pyjwt
            secret  = os.environ.get('JWT_SECRET', 'yaply-secret-key')
            payload = pyjwt.decode(token, secret, algorithms=['HS256'])
            user_id = payload.get('user_id') or payload.get('id')

            db = get_db()
            if not is_member(db, group_id, user_id):
                db.close()
                return

            # Update in DB
            existing = db.execute(
                'SELECT id FROM live_locations WHERE user_id=? AND group_id=?',
                (user_id, group_id)
            ).fetchone()

            if existing:
                db.execute('''
                    UPDATE live_locations
                    SET lat=?, lng=?, place_name=?, updated_at=CURRENT_TIMESTAMP
                    WHERE user_id=? AND group_id=?
                ''', (lat, lng, place_name, user_id, group_id))
            else:
                db.execute('''
                    INSERT INTO live_locations
                        (user_id, group_id, lat, lng, place_name, is_sharing)
                    VALUES (?, ?, ?, ?, ?, 1)
                ''', (user_id, group_id, lat, lng, place_name))

            member = get_member_info(db, group_id, user_id)
            db.commit()
            db.close()

            # Broadcast location to group
            emit('location_updated', {
                'user_id':    user_id,
                'group_id':   group_id,
                'lat':        lat,
                'lng':        lng,
                'place_name': place_name,
                'emoji':      member['emoji'] if member else '🧑',
                'color':      member['color'] if member else '#00D4AA',
                'name':       member['name']  if member else 'Member',
                'updated_at': datetime.now().isoformat()
            }, room='group_' + str(group_id))

        except Exception as e:
            print('[WS] location_update error:', e)

    @socketio.on('typing')
    def on_typing(data):
        """Broadcast typing indicator."""
        group_id = data.get('group_id')
        token    = data.get('token')
        is_typing = data.get('is_typing', True)

        if not group_id or not token:
            return

        try:
            import jwt as pyjwt
            secret  = os.environ.get('JWT_SECRET', 'yaply-secret-key')
            payload = pyjwt.decode(token, secret, algorithms=['HS256'])
            user_id = payload.get('user_id') or payload.get('id')

            db = get_db()
            member = get_member_info(db, group_id, user_id)
            db.close()

            if member:
                emit('user_typing', {
                    'user_id':  user_id,
                    'name':     member['name'],
                    'emoji':    member['emoji'],
                    'is_typing': is_typing
                }, room='group_' + str(group_id), include_self=False)

        except Exception as e:
            print('[WS] typing error:', e)

    @socketio.on('ping_member')
    def on_ping_member(data):
        """Ping a specific member."""
        group_id  = data.get('group_id')
        target_id = data.get('target_user_id')
        token     = data.get('token')

        if not group_id or not target_id or not token:
            return

        try:
            import jwt as pyjwt
            secret  = os.environ.get('JWT_SECRET', 'yaply-secret-key')
            payload = pyjwt.decode(token, secret, algorithms=['HS256'])
            user_id = payload.get('user_id') or payload.get('id')

            db = get_db()
            sender = get_member_info(db, group_id, user_id)
            db.close()

            if sender:
                emit('you_were_pinged', {
                    'from_user_id': user_id,
                    'from_name':    sender['name'],
                    'from_emoji':   sender['emoji'],
                    'group_id':     group_id
                }, room='group_' + str(group_id))

        except Exception as e:
            print('[WS] ping_member error:', e)

    @socketio.on('send_sos')
    def on_send_sos(data):
        """Real-time SOS broadcast."""
        group_id = data.get('group_id')
        token    = data.get('token')
        sos_type = data.get('sos_type', 'general')
        message  = data.get('message', '')
        lat      = data.get('lat')
        lng      = data.get('lng')

        if not group_id or not token:
            return

        try:
            import jwt as pyjwt
            secret  = os.environ.get('JWT_SECRET', 'yaply-secret-key')
            payload = pyjwt.decode(token, secret, algorithms=['HS256'])
            user_id = payload.get('user_id') or payload.get('id')

            db = get_db()
            member = get_member_info(db, group_id, user_id)

            if not member:
                db.close()
                return

            name = member['name']
            sos_labels = {
                'lost':    name + ' is LOST and needs help!',
                'medical': name + ' has a MEDICAL EMERGENCY!',
                'safety':  name + ' needs immediate SAFETY help!',
                'general': name + ' sent an SOS alert!'
            }

            full_msg = sos_labels.get(sos_type, sos_labels['general'])
            if message:
                full_msg += ' — ' + message

            # Save to DB
            db.execute('''
                INSERT INTO sos_alerts
                    (group_id, user_id, sos_type, message, lat, lng)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (group_id, user_id, sos_type, message, lat, lng))

            db.execute('''
                INSERT INTO group_messages
                    (group_id, user_id, msg_type, content, metadata)
                VALUES (?, ?, 'sos', ?, ?)
            ''', (group_id, user_id, full_msg,
                  json.dumps({'sos_type': sos_type, 'lat': lat, 'lng': lng})))

            notify_all_members(
                db, group_id, user_id,
                'sos', '🆘 EMERGENCY — ' + name, full_msg,
                {'group_id': group_id, 'lat': lat, 'lng': lng}
            )

            db.commit()
            db.close()

            # BROADCAST to entire group — immediate
            emit('sos_alert', {
                'user_id':  user_id,
                'name':     name,
                'emoji':    member['emoji'],
                'sos_type': sos_type,
                'message':  full_msg,
                'lat':      lat,
                'lng':      lng,
                'group_id': group_id
            }, room='group_' + str(group_id))

        except Exception as e:
            print('[WS] send_sos error:', e)

    print('[Groups] WebSocket events registered ✓')