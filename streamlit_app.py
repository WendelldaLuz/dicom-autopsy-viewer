import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import pandas as pd
from datetime import datetime
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import json
import sqlite3
from PIL import Image
import logging
import hashlib

# Configurar logging de segurança
logging.basicConfig(
    filename='security.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suprimir warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Configuração da página
st.set_page_config(
    page_title="DICOM Autopsy Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado - Tema autópsia virtual
st.markdown("""
<style>
    .main { background: #1a1a1a; }
    .stApp { background: #0d0d0d; color: #ffffff; }
    .main-header { font-size: 2.5rem; color: #ffffff !important; text-align: center; font-weight: 700; }
    .sub-header { font-size: 1.5rem; color: #ffffff !important; font-weight: 600; }
    p, div, span, label { color: #e0e0e0 !important; }
    .card { background: #2d2d2d; padding: 20px; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid #00bcd4; }
    .patient-card { border-left: 4px solid #ff5252; }
    .tech-card { border-left: 4px solid #4caf50; }
    .image-card { border-left: 4px solid #9c27b0; }
    .stats-card { border-left: 4px solid #ff9800; }
    .login-card { border-left: 4px solid #00bcd4; background: #2d2d2d; padding: 30px; border-radius: 15px; }
    .feedback-card { border-left: 4px solid #ff9800; background: #2d2d2d; padding: 20px; border-radius: 12px; }
    .stButton>button { background: linear-gradient(45deg, #00bcd4, #00838f); color: white !important; border-radius: 8px; padding: 12px 24px; }
    .uploaded-file { background: #333333; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 3px solid #00bcd4; }
    .metric-value { font-size: 1.3rem; font-weight: 700; color: #00bcd4 !important; }
    .metric-label { font-size: 0.9rem; color: #b0b0b0 !important; font-weight: 500; }
    .file-size { color: #00bcd4; font-size: 0.8rem; margin-top: 5px; }
    .upload-info { background: #2d2d2d; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #4caf50; }
    .star-rating { font-size: 2rem; color: #ffd700; margin: 10px 0; }
    .security-alert { background: #ffebee; color: #c62828; padding: 10px; border-radius: 5px; border-left: 4px solid #c62828; }
</style>
""", unsafe_allow_html=True)

# Configuração do banco de dados para feedback
DB_PATH = "feedback_database.db"

# Configurações de email (USE SENHA DE APP DO GMAIL)
EMAIL_CONFIG = {
    'sender': 'wenndell.luz@gmail.com',
    'password': 'sua_senha_de_app_do_gmail',  # GERAR NO: https://myaccount.google.com/apppasswords
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
}

# Limite de rate limiting
UPLOAD_LIMITS = {
    'max_files': 10,
    'max_size_mb': 2000,
    'max_uploads_per_hour': 5
}

def init_database():
    """Inicializa o banco de dados para feedback"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS feedback
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_email TEXT,
                      feedback_text TEXT,
                      rating INTEGER,
                      report_data TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS system_learning
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      error_type TEXT,
                      error_message TEXT,
                      solution_applied TEXT,
                      occurrence_count INTEGER DEFAULT 1,
                      last_occurrence DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS security_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      event_type TEXT,
                      user_ip TEXT,
                      user_agent TEXT,
                      details TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
    except Exception as e:
        log_security_event("DATABASE_ERROR", f"Erro ao inicializar banco: {e}")

def log_security_event(event_type, details):
    """Registra eventos de segurança"""
    try:
        user_ip = "unknown"
        user_agent = "unknown"
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO security_logs (event_type, user_ip, user_agent, details)
                     VALUES (?, ?, ?, ?)''', 
                 (event_type, user_ip, user_agent, details))
        conn.commit()
        conn.close()
        
        logging.info(f"SECURITY - {event_type}: {details}")
    except Exception as e:
        logging.error(f"Erro ao registrar evento de segurança: {e}")

def validate_dicom_file(file):
    """Valida se é um arquivo DICOM real"""
    try:
        # Verifica assinatura DICOM
        file.seek(128)
        signature = file.read(4)
        file.seek(0)
        
        if signature == b'DICM':
            return True
        else:
            log_security_event("INVALID_FILE", "Arquivo não é DICOM válido")
            return False
    except Exception as e:
        log_security_event("FILE_VALIDATION_ERROR", f"Erro na validação: {e}")
        return False

def sanitize_patient_data(dataset):
    """Remove dados sensíveis se necessário"""
    try:
        sensitive_tags = ['PatientName', 'PatientID', 'PatientBirthDate', 'ReferringPhysicianName']
        
        for tag in sensitive_tags:
            if hasattr(dataset, tag):
                original_value = getattr(dataset, tag)
                setattr(dataset, tag, "REDACTED")
                log_security_event("DATA_SANITIZED", f"Campo {tag} redacted: {original_value}")
        
        return dataset
    except Exception as e:
        log_security_event("SANITIZATION_ERROR", f"Erro ao sanitizar dados: {e}")
        return dataset

def check_upload_limits(uploaded_files):
    """Verifica limites de upload"""
    try:
        total_size = sum(f.size for f in uploaded_files)
        
        if len(uploaded_files) > UPLOAD_LIMITS['max_files']:
            log_security_event("UPLOAD_LIMIT_EXCEEDED", f"Máximo de {UPLOAD_LIMITS['max_files']} arquivos excedido")
            return False, f"Máximo de {UPLOAD_LIMITS['max_files']} arquivos permitido"
            
        if total_size > UPLOAD_LIMITS['max_size_mb'] * 1024 * 1024:
            log_security_event("SIZE_LIMIT_EXCEEDED", f"Máximo de {UPLOAD_LIMITS['max_size_mb']}MB excedido")
            return False, f"Máximo de {UPLOAD_LIMITS['max_size_mb']}MB permitido"
            
        return True, "OK"
    except Exception as e:
        log_security_event("UPLOAD_LIMIT_ERROR", f"Erro ao verificar limites: {e}")
        return False, "Erro ao verificar limites"

init_database()

# ... (AS OUTRAS FUNÇÕES PERMANECEM IGUAIS: save_feedback, log_error, get_file_size, send_email_report, safe_dicom_value, create_medical_visualization, create_advanced_histogram, create_pdf_report, show_feedback_section, show_login_page, show_dashboard, show_main_app, main) ...

# NA FUNÇÃO show_main_app, ATUALIZE A PARTE DO UPLOAD:

def show_main_app():
    """Aplicativo principal após autenticação"""
    # ... (código anterior) ...
    
    with st.sidebar:
        # ... (código anterior) ...
        
        uploaded_files = st.file_uploader(
            "📤 Selecione os arquivos DICOM",
            type=['dcm', 'DCM'],
            accept_multiple_files=True,
            help=f"Selecione até {UPLOAD_LIMITS['max_files']} arquivos DICOM (máximo {UPLOAD_LIMITS['max_size_mb']}MB)"
        )
        
        if uploaded_files:
            # VERIFICA LIMITES DE SEGURANÇA
            is_valid, message = check_upload_limits(uploaded_files)
            
            if not is_valid:
                st.error(f"❌ {message}")
                log_security_event("UPLOAD_BLOCKED", message)
            else:
                total_size = sum(f.size for f in uploaded_files)
                
                # VALIDA CADA ARQUIVO
                valid_files = []
                for file in uploaded_files:
                    if validate_dicom_file(file):
                        valid_files.append(file)
                    else:
                        st.warning(f"⚠️ Arquivo {file.name} não é um DICOM válido e foi ignorado")
                
                if valid_files:
                    st.success(f"✅ {len(valid_files)} arquivo(s) válido(s) - {get_file_size(total_size)}")
                else:
                    st.error("❌ Nenhum arquivo DICOM válido encontrado")

    # ... (resto do código) ...

# NA FUNÇÃO PRINCIPAL, ADICIONE VERIFICAÇÃO DE SEGURANÇA:

def main():
    """Função principal"""
    try:
        log_security_event("APP_START", "Aplicativo iniciado")
        
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            st.session_state.user_data = {}
            st.session_state.feedback_submitted = False
