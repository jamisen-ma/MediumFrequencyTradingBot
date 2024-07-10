import sqlite3

def setup_database():
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS SentimentData (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            symbol TEXT,
            sentiment_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS FinancialData (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adj_close REAL
        )
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_database()
