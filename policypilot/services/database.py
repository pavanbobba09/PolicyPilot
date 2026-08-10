import sqlite3
import logging
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Dict, Any, Iterable

from ..config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def get_db_connection():
    """Context manager for handling database connections."""
    conn = None
    try:
        database_path = Path(settings.DATABASE_URL)
        if database_path.parent != Path("."):
            database_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.DATABASE_URL)
        conn.row_factory = sqlite3.Row
        logger.debug("Database connection established.")
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed.")

def setup_database():
    """Creates/updates all necessary tables in the database."""
    logger.info("Setting up database schema...")
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            data_type TEXT NOT NULL,
            category TEXT,
            local_path TEXT,
            content_hash TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            metadata_json TEXT,
            vector_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES data_sources (id)
        );
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_chunks_vector_id ON knowledge_chunks(vector_id);",
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            profile_data_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """
    ]
    with get_db_connection() as conn:
        for statement in ddl_statements:
            conn.cursor().execute(statement)
        conn.commit()
    logger.info("Database schema setup complete.")

# --- Crawler-related Helpers ---

def initialize_crawl_jobs():
    """Populates the data_sources table with the initial crawl job start URLs."""
    logger.info("Initializing crawl jobs from config...")
    query = "INSERT OR IGNORE INTO data_sources (name, url, data_type, category, status, created_at) VALUES (?, ?, ?, ?, ?, ?)"
    with get_db_connection() as conn:
        for job in settings.CRAWLING_JOBS:
            conn.cursor().execute(query, (
                job['name'],
                job['start_url'],
                'start_url',
                job.get('domain_lock', 'General'),
                'pending',
                _utc_now()
            ))
        conn.commit()
    logger.info("Crawl jobs initialized successfully.")


def find_or_create_web_source(url: str, name: str) -> int:
    """
    Finds an existing data source by URL or creates a new one if it doesn't exist.
    Used for dynamically ingested web search results.

    Returns:
        The ID of the data source.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Check if it exists
        cursor.execute("SELECT id FROM data_sources WHERE url = ?", (url,))
        row = cursor.fetchone()
        if row:
            return row['id']

        # 2. If not, create it
        logger.info(f"Registering new dynamic web source: {url}")
        insert_query = """
        INSERT INTO data_sources (name, url, data_type, category, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_query, (
            name, url, 'web_search_result', 'Dynamic', 'ingested', _utc_now(), _utc_now()
        ))
        conn.commit()
        return cursor.lastrowid


def get_source_by_url(url: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM data_sources WHERE url = ?", (url,)).fetchone()
        return dict(row) if row else None


def get_vector_ids_for_source(source_id: int) -> list[str]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT vector_id FROM knowledge_chunks WHERE source_id = ? AND vector_id IS NOT NULL",
            (source_id,),
        ).fetchall()
        return [str(row["vector_id"]) for row in rows]


def replace_source_chunks(
    source_id: int,
    *,
    name: str,
    local_path: str,
    content_hash: str,
    chunks: Iterable[tuple[str, str, str]],
) -> None:
    """Atomically replace SQLite chunk metadata after vector storage succeeds."""

    now = _utc_now()
    with get_db_connection() as conn:
        conn.execute("DELETE FROM knowledge_chunks WHERE source_id = ?", (source_id,))
        conn.executemany(
            "INSERT INTO knowledge_chunks (source_id, chunk_text, metadata_json, vector_id) VALUES (?, ?, ?, ?)",
            ((source_id, text, metadata_json, vector_id) for text, metadata_json, vector_id in chunks),
        )
        conn.execute(
            """
            UPDATE data_sources
            SET name = ?, local_path = ?, content_hash = ?, status = 'ingested', updated_at = ?
            WHERE id = ?
            """,
            (name, local_path, content_hash, now, source_id),
        )
        conn.commit()


def database_is_ready() -> bool:
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


def add_discovered_source(url: str, category: str, data_type: str) -> int:
    """Adds a newly discovered URL to the database if it doesn't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM data_sources WHERE url = ?", (url,))
        row = cursor.fetchone()
        if row:
            return row['id']

        path_name = Path(urlparse(url).path).name
        if not path_name:
            path_name = urlparse(url).path.strip('/').replace('/', '_') or urlparse(url).netloc

        insert_query = "INSERT INTO data_sources (name, url, data_type, category, status, created_at) VALUES (?, ?, ?, ?, ?, ?)"
        cursor.execute(insert_query, (path_name, url, data_type, category, 'pending', _utc_now()))
        conn.commit()
        logger.debug(f"Discovered and added new source: {url}")
        return cursor.lastrowid

# --- User Management Helpers ---

def create_user(username: str, hashed_password: str, role: str = 'user') -> Optional[int]:
    """Creates a new user in the database."""
    query = "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)"
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (username, hashed_password, role))
            conn.commit()
            logger.info(f"User '{username}' created successfully.")
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        logger.warning(f"Attempted to create a user that already exists: {username}")
        return None
    except sqlite3.Error as e:
        logger.error(f"Database error while creating user {username}: {e}")
        return None

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user by their username."""
    query = "SELECT * FROM users WHERE username = ?"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (username,))
        row = cursor.fetchone()
        return dict(row) if row else None

# --- User Profile Helpers ---

def create_or_update_user_profile(user_id: int, profile_data: Dict[str, Any]) -> bool:
    """Creates or updates a user's 360-degree profile."""
    profile_json = json.dumps(profile_data)
    current_time = _utc_now()

    # Using INSERT OR REPLACE for simplicity (UPSERT)
    query = """
    INSERT INTO user_profiles (user_id, profile_data_json, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(user_id) DO UPDATE SET
        profile_data_json = excluded.profile_data_json,
        updated_at = excluded.updated_at;
    """
    try:
        with get_db_connection() as conn:
            conn.cursor().execute(query, (user_id, profile_json, current_time))
            conn.commit()
        logger.info(f"Profile for user_id {user_id} created/updated.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error while updating profile for user_id {user_id}: {e}")
        return False

def get_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a user's profile."""
    query = "SELECT profile_data_json FROM user_profiles WHERE user_id = ?"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (user_id,))
        row = cursor.fetchone()
        if row and row['profile_data_json']:
            return json.loads(row['profile_data_json'])
    return None
