import sqlite3
import os

def init_db():
    conn = sqlite3.connect('music_rec.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT, 
        password TEXT,
        username TEXT,
        age INTEGER,
        gender TEXT
    )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
    # Create static/audio directory if it doesn't exist
    os.makedirs('static/audio', exist_ok=True)
    print("Directories checked.")
