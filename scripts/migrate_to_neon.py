"""
One-off migration script to copy data from local SQLite to Neon Postgres.

Prereqs:
- Set SQLITE_URL (e.g., sqlite:///interview_platform.db)
- Set POSTGRES_URL (e.g., postgresql+psycopg2://...neon.../neondb?sslmode=require)
- Ensure Postgres schema exists (run flask db upgrade before migrating)

Run:
  python scripts/migrate_to_neon.py
"""
import os
import sys
import pathlib
from contextlib import contextmanager
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Load env
load_dotenv()

SQLITE_URL = os.getenv('SQLITE_URL', 'sqlite:///interview_platform.db')
POSTGRES_URL = os.getenv('POSTGRES_URL') or os.getenv('DATABASE_URL')

# Ensure the app can be imported when running from scripts/ directory
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# If POSTGRES_URL is provided, force the Flask app to use it
if POSTGRES_URL:
    os.environ['DATABASE_URL'] = POSTGRES_URL

# If using default SQLite but instance DB exists, prefer it automatically
if SQLITE_URL == 'sqlite:///interview_platform.db':
    instance_db = PROJECT_ROOT / 'instance' / 'interview_platform.db'
    if instance_db.exists():
        SQLITE_URL = 'sqlite:///instance/interview_platform.db'

print(f"Source (SQLite): {SQLITE_URL}")
print(f"Destination (Postgres): {os.getenv('DATABASE_URL')}")

if not POSTGRES_URL:
    print('ERROR: Set POSTGRES_URL or DATABASE_URL to your Neon connection string')
    sys.exit(1)

# Import Flask app and ORM models for Postgres session and model mapping
from app import app, db, User, Interview, InterviewSession, QuestionAttempt, UserSettings

# Source (SQLite) engine
src_engine = create_engine(SQLITE_URL)
# Destination (Postgres/Neon) uses existing db.session from app

SrcSession = sessionmaker(bind=src_engine)

@contextmanager
def src_session_scope():
    session = SrcSession()
    try:
        yield session
    finally:
        session.close()


def fetch_all(sql, session):
    try:
        return session.execute(text(sql)).mappings().all()
    except Exception as exc:
        print(f"ERROR executing query: {sql}")
        print("Hint: Ensure SQLITE_URL points to your real SQLite file (e.g., sqlite:///instance/interview_platform.db)")
        raise


def migrate_users():
    print('Migrating users...')
    with src_session_scope() as s, app.app_context():
        rows = fetch_all('SELECT id, name, email, password_hash FROM user', s)
        migrated = 0
        for row in rows:
            # Skip if email exists
            if User.query.filter_by(email=row['email']).first():
                continue
            u = User(id=row['id'], name=row['name'], email=row['email'], password_hash=row['password_hash'])
            db.session.add(u)
            migrated += 1
        db.session.commit()
        print(f'Users: inserted {migrated}, total source {len(rows)}')


def migrate_user_settings():
    print('Migrating user settings...')
    with src_session_scope() as s, app.app_context():
        try:
            rows = fetch_all('SELECT id, user_id, email_notifications, created_at, updated_at FROM user_settings', s)
        except Exception:
            rows = []
        migrated = 0
        for row in rows:
            if not User.query.get(row['user_id']):
                continue
            if UserSettings.query.filter_by(user_id=row['user_id']).first():
                continue
            us = UserSettings(
                id=row['id'],
                user_id=row['user_id'],
                email_notifications=bool(row['email_notifications']),
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            db.session.add(us)
            migrated += 1
        db.session.commit()
        print(f'UserSettings: inserted {migrated}, total source {len(rows)}')


def migrate_interviews():
    print('Migrating interviews...')
    with src_session_scope() as s, app.app_context():
        try:
            rows = fetch_all('SELECT id, user_id, domain, score, total_questions, correct_answers, performance_summary, created_at FROM interview', s)
        except Exception:
            rows = []
        migrated = 0
        for row in rows:
            if not User.query.get(row['user_id']):
                continue
            if Interview.query.get(row['id']):
                continue
            iv = Interview(
                id=row['id'],
                user_id=row['user_id'],
                domain=row['domain'],
                score=row['score'],
                total_questions=row['total_questions'],
                correct_answers=row['correct_answers'],
                performance_summary=row['performance_summary'],
                created_at=row['created_at']
            )
            db.session.add(iv)
            migrated += 1
        db.session.commit()
        print(f'Interviews: inserted {migrated}, total source {len(rows)}')


def migrate_sessions_and_attempts():
    print('Migrating interview sessions...')
    with src_session_scope() as s, app.app_context():
        try:
            sessions = fetch_all('SELECT id, user_id, domain, started_at, completed_at, current_index, total_questions FROM interview_session', s)
        except Exception:
            sessions = []
        migrated_sessions = 0
        for row in sessions:
            if not User.query.get(row['user_id']):
                continue
            if InterviewSession.query.get(row['id']):
                continue
            sess = InterviewSession(
                id=row['id'],
                user_id=row['user_id'],
                domain=row['domain'],
                started_at=row['started_at'],
                completed_at=row['completed_at'],
                current_index=row['current_index'],
                total_questions=row['total_questions']
            )
            db.session.add(sess)
            migrated_sessions += 1
        db.session.commit()
        print(f'Sessions: inserted {migrated_sessions}, total source {len(sessions)}')

        print('Migrating question attempts...')
        try:
            attempts = fetch_all('SELECT id, session_id, question_text, correct_answer, user_answer, is_correct, created_at, voice_clarity_score, voice_confidence_score, voice_tone_analysis FROM question_attempt', s)
        except Exception:
            attempts = []
        migrated_attempts = 0
        for row in attempts:
            if not InterviewSession.query.get(row['session_id']):
                continue
            if QuestionAttempt.query.get(row['id']):
                continue
            qa = QuestionAttempt(
                id=row['id'],
                session_id=row['session_id'],
                question_text=row['question_text'],
                correct_answer=row['correct_answer'],
                user_answer=row['user_answer'],
                is_correct=row['is_correct'],
                created_at=row['created_at'],
                voice_clarity_score=row.get('voice_clarity_score'),
                voice_confidence_score=row.get('voice_confidence_score'),
                voice_tone_analysis=row.get('voice_tone_analysis')
            )
            db.session.add(qa)
            migrated_attempts += 1
        db.session.commit()
        print(f'Attempts: inserted {migrated_attempts}, total source {len(attempts)}')


if __name__ == '__main__':
    print('Starting migration from SQLite to Neon Postgres')
    migrate_users()
    migrate_user_settings()
    migrate_interviews()
    migrate_sessions_and_attempts()
    print('Migration complete.')
