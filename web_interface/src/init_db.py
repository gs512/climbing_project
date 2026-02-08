import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "routes.db")


def init_db() -> None:
    """Create the routes.db file and routes table if they don't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                name TEXT NOT NULL,
                difficulty TEXT DEFAULT 'V0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        print(f"Successfully ensured 'routes' table exists at {DB_PATH!r}.")


if __name__ == "__main__":
    init_db()
