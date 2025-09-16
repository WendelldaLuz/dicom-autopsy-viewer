import sqlite3
import logging

def safe_init_database():
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT,
                report_name TEXT,
                report_data BLOB,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                parameters TEXT
            )
        """)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Erro ao inicializar base de dados: {e}")
        return False

def save_report_to_db(user_email, report_name, report_data, parameters):
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (user_email, report_name, report_data, parameters)
            VALUES (?, ?, ?, ?)
        """, (user_email, report_name, report_data, parameters))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Erro ao salvar relatório: {e}")
        return False
