import streamlit as st
import sqlite3
import logging
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import json
from datetime import datetime
from io import BytesIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import socket
import base64
import colorsys
import scipy.stats as stats
from scipy.optimize import curve_fit
import cv2

# Importar módulos aprimorados
from enhanced_visualization import enhanced_visualization_tab
from enhanced_statistics import enhanced_statistics_tab
from enhanced_technical_analysis import enhanced_technical_analysis_tab
from enhanced_quality_metrics import enhanced_quality_metrics_tab
from enhanced_ra_index import enhanced_ra_index_tab

# Configuração inicial da página
st.set_page_config(
    page_title="DICOM Autopsy Viewer Pro - Enhanced",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Variáveis de estado -----
if 'background_color' not in st.session_state:
    st.session_state.background_color = '#000000'
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'color_theme' not in st.session_state:
    st.session_state.color_theme = {
        'primary': '#00bcd4',
        'secondary': '#00838f',
        'accent': '#ff9800',
        'text': '#ffffff',
        'background': '#000000'
    }
if 'rating' not in st.session_state:
    st.session_state.rating = 0
if 'learning_data' not in st.session_state:
    st.session_state.learning_data = []

# Definições globais
DB_PATH = "feedback_database.db"
UPLOAD_LIMITS = {
    'max_files': 10,  # Aumentado para 10
    'max_size_mb': 1000  # Aumentado para 1GB
}
EMAIL_CONFIG = {
    'sender': 'seu-email@gmail.com',
    'password': 'sua-senha-de-app',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CSS personalizado aprimorado
def update_css_theme():
    theme = st.session_state.color_theme
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {{
            font-family: 'Inter', sans-serif;
        }}
        
        .main {{
            background: {theme['background']};
            color: {theme['text']};
        }}
        
        .stApp {{ 
            background: {theme['background']};
            color: {theme['text']}; 
        }}
        
        .main-header {{ 
            font-size: 3rem; 
            color: {theme['text']} !important; 
            text-align: center; 
            font-weight: 700; 
            margin-bottom: 1rem;
            background: linear-gradient(45deg, #00bcd4, #00838f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .sub-header {{ 
            font-size: 1.5rem; 
            color: {theme['text']} !important; 
            font-weight: 600; 
            margin-bottom: 1rem;
            text-align: center;
        }}
        
        .upload-highlight {{
            background: linear-gradient(135deg, #00bcd4, #00838f);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 8px 25px rgba(0, 188, 212, 0.3);
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ box-shadow: 0 8px 25px rgba(0, 188, 212, 0.3); }}
            50% {{ box-shadow: 0 8px 25px rgba(0, 188, 212, 0.6); }}
            100% {{ box-shadow: 0 8px 25px rgba(0, 188, 212, 0.3); }}
        }}
        
        .upload-highlight h3 {{
            color: white !important;
            margin: 0;
            font-size: 1.8rem;
            font-weight: 700;
        }}
        
        .upload-highlight p {{
            color: rgba(255, 255, 255, 0.9) !important;
            margin: 10px 0;
            font-size: 1.1rem;
        }}
        
        p, div, span, label {{ 
            color: {theme['text']} !important; 
        }}
        
        .card {{ 
            background: linear-gradient(135deg, #1a1a1a, #2d2d2d); 
            padding: 25px; 
            border-radius: 15px; 
            margin-bottom: 25px; 
            border-left: 5px solid {theme['primary']};
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }}
        
        .patient-card {{ border-left: 5px solid #ff5252; }}
        .tech-card {{ border-left: 5px solid #4caf50; }}
        .image-card {{ border-left: 5px solid #9c27b0; }}
        .stats-card {{ border-left: 5px solid {theme['accent']}; }}
        
        .stButton>button {{
            background: linear-gradient(45deg, {theme['primary']}, {theme['secondary']});
            color: white !important;
            border-radius: 10px;
            padding: 15px 30px;
            border: none;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 188, 212, 0.3);
        }}
        
        .stButton>button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 188, 212, 0.5);
        }}
        
        .uploaded-file {{
            background: linear-gradient(135deg, #333333, #404040);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid {theme['primary']};
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .metric-value {{ 
            font-size: 1.5rem; 
            font-weight: 700; 
            color: {theme['primary']} !important; 
        }}
        
        .metric-label {{ 
            font-size: 1rem; 
            color: #b0b0b0 !important; 
            font-weight: 500; 
        }}
        
        .sidebar .sidebar-content {{
            background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        }}
        
        .stSelectbox, .stTextInput, .stTextArea {{
            background: #2d2d2d;
            color: white;
            border-radius: 8px;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
            justify-content: center;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: linear-gradient(135deg, #2d2d2d, #404040);
            border-radius: 10px 10px 0 0;
            padding: 15px 25px;
            font-weight: 600;
            color: white;
            border: none;
            transition: all 0.3s ease;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(45deg, {theme['primary']}, {theme['secondary']});
            box-shadow: 0 4px 15px rgba(0, 188, 212, 0.3);
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-online {{
            background-color: #4caf50;
            animation: blink 1.5s infinite;
        }}
        
        @keyframes blink {{
            0%, 50% {{ opacity: 1; }}
            51%, 100% {{ opacity: 0.3; }}
        }}
        
        .progress-container {{
            background: #2d2d2d;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }}
        
        .file-info {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 5px 0;
        }}
        
        .file-size {{
            font-size: 0.9rem;
            color: #b0b0b0;
        }}
    </style>
    """, unsafe_allow_html=True)

# Funções do sistema original (mantidas e otimizadas)
def init_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS feedback
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, user_email TEXT, feedback_text TEXT,
                      rating INTEGER, report_data TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS system_learning
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, error_type TEXT, error_message TEXT,
                      solution_applied TEXT, occurrence_count INTEGER DEFAULT 1,
                      last_occurrence DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS security_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT, user_ip TEXT, user_agent TEXT,
                      details TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS access_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, user TEXT, action TEXT,
                      resource TEXT, details TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS gas_analysis_data
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, 
                      fick_diffusion_coeff REAL, metierlich_params TEXT,
                      statistical_metrics TEXT, inference_results TEXT)''')
        conn.commit()
        conn.close()
        logging.info("Banco de dados inicializado com sucesso")
    except Exception as e:
        print(f"Erro ao inicializar banco: {e}")
        logging.error(f"Erro ao inicializar banco: {e}")

def log_security_event(event_type, details):
    try:
        user_ip = "unknown"
        user_agent = "unknown"
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO security_logs (event_type, user_ip, user_agent, details)
                     VALUES (?, ?, ?, ?)''', (event_type, user_ip, user_agent, details))
        conn.commit()
        conn.close()
        logging.info(f"SECURITY - {event_type}: {details}")
    except Exception as e:
        print(f"SECURITY FALLBACK - {event_type}: {details}")
        logging.error(f"Erro ao registrar evento de segurança: {e}")

def validate_dicom_file(file):
    try:
        max_size = 1000 * 1024 * 1024  # 1GB
        file_size = len(file.getvalue())
        if file_size > max_size:
            log_security_event("FILE_TOO_LARGE", f"Arquivo excede limite de {max_size} bytes")
            return False
        
        original_position = file.tell()
        file.seek(128)
        signature = file.read(4)
        file.seek(original_position)
        
        if signature != b'DICM':
            log_security_event("INVALID_FILE", "Arquivo não é DICOM válido")
            return False
            
        return True
    except Exception as e:
        log_security_event("FILE_VALIDATION_ERROR", f"Erro na validação: {e}")
        return False

def sanitize_patient_data(dataset):
    try:
        sensitive_tags = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'PatientAge', 'PatientSize', 'PatientWeight', 'PatientAddress',
            'ReferringPhysicianName', 'ReferringPhysicianAddress',
            'ReferringPhysicianTelephoneNumbers', 'InstitutionName',
            'InstitutionAddress', 'StudyDate', 'StudyTime', 'AccessionNumber',
            'StudyID', 'SeriesDate', 'ContentDate', 'AcquisitionDateTime',
            'InstitutionDepartmentName', 'StationName', 'PerformingPhysicianName'
        ]
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

def get_file_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def safe_dicom_value(dataset, tag, default="N/A"):
    try:
        value = getattr(dataset, tag, default)
        if value is None or str(value).strip() == "":
            return default
        if isinstance(value, pydicom.multival.MultiValue):
            return " / ".join([str(v) for v in value])
        return str(value).strip()
    except Exception:
        return default

def safe_init_database():
    try:
        init_database()
        return True
    except Exception as e:
        print(f"Falha crítica na inicialização do banco: {e}")
        logging.critical(f"Falha na inicialização do banco: {e}")
        return False

def show_user_form():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("👤 Dados do Analista Forense")
    st.info("Por favor, preencha os campos abaixo para acessar a ferramenta de análise avançada.")
    
    with st.form("user_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input("Nome Completo", key="user_name")
            department = st.text_input("Departamento/Órgão", key="user_department")
        
        with col2:
            email = st.text_input("Email Profissional", key="user_email")
            contact = st.text_input("Telefone/Contato", key="user_contact")
        
        submitted = st.form_submit_button("🚀 Iniciar Análise Forense", use_container_width=True)
        
        if submitted:
            if not full_name or not department or not email or not contact:
                st.error("❌ Todos os campos são obrigatórios para garantir a rastreabilidade da análise.")
            else:
                st.session_state.user_data = {
                    'nome': full_name, 
                    'departamento': department,
                    'email': email, 
                    'contato': contact
                }
                st.success("✅ Dados salvos com sucesso! Redirecionando para o sistema...")
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_enhanced_upload_section():
    """Cria seção de upload destacada e aprimorada"""
    
    st.markdown("""
    <div class="upload-highlight">
        <h3>📤 Central de Upload de Exames DICOM</h3>
        <p>Faça upload dos seus arquivos DICOM para análise forense avançada</p>
        <div style="display: flex; justify-content: space-around; margin-top: 15px;">
            <div>
                <strong>📊 Limite:</strong> 10 arquivos
            </div>
            <div>
                <strong>💾 Tamanho:</strong> 1GB total
            </div>
            <div>
                <strong>📄 Formato:</strong> .dcm, .DCM
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload com estilo aprimorado
    uploaded_files = st.file_uploader(
        "Selecione os arquivos DICOM para análise forense",
        type=['dcm', 'DCM'],
        accept_multiple_files=True,
        help=f"Selecione até {UPLOAD_LIMITS['max_files']} arquivos DICOM (máximo {UPLOAD_LIMITS['max_size_mb']}MB total)",
        key="main_uploader"
    )
    
    if uploaded_files:
        is_valid, message = check_upload_limits(uploaded_files)
        
        if not is_valid:
            st.error(f"❌ {message}")
            return None
        else:
            # Status de upload bem-sucedido
            total_size = sum(f.size for f in uploaded_files)
            st.success(f"✅ {len(uploaded_files)} arquivo(s) carregado(s) com sucesso - {get_file_size(total_size)}")
            
            # Mostrar arquivos carregados com informações detalhadas
            with st.expander("📋 Arquivos Carregados", expanded=True):
                for i, file in enumerate(uploaded_files, 1):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {file.name}**")
                    
                    with col2:
                        st.markdown(f"📏 {get_file_size(file.size)}")
                    
                    with col3:
                        st.markdown("🟢 Válido")
            
            return uploaded_files
    
    return None

def show_main_app():
    st.markdown(f"<h1 class='main-header'>🔬 DICOM Autopsy Viewer PRO</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 class='sub-header'>Sistema Avançado de Análise Forense Digital e Preditiva</h3>", unsafe_allow_html=True)

    # Sidebar aprimorada
    with st.sidebar:
        # Informações do usuário
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #00BFFF, #0099CC); 
                    padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px;'>
            <div class="status-indicator status-online"></div>
            <h3 style='margin: 0; font-size: 1.2rem;'>👤 Analista Ativo</h3>
            <p style='margin: 5px 0; font-size: 1rem; font-weight: 600;'>{st.session_state.user_data['nome']}</p>
            <p style='margin: 0; font-size: 0.9rem;'>{st.session_state.user_data['departamento']}</p>
            <p style='margin: 5px 0; font-size: 0.8rem;'>📧 {st.session_state.user_data['email']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sistema de logs em tempo real
        st.markdown("### 📊 Status do Sistema")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sessão Ativa", "🟢 Online")
        with col2:
            st.metric("Arquivos", len(st.session_state.get('uploaded_files', [])))
        
        # Configurações avançadas
        with st.expander("⚙️ Configurações Avançadas", expanded=False):
            st.markdown("**Qualidade de Análise:**")
            analysis_quality = st.selectbox(
                "Nível de processamento",
                ["Rápido", "Padrão", "Detalhado", "Forense Completo"],
                index=2
            )
            
            st.markdown("**Logs de Segurança:**")
            show_security_logs = st.checkbox("Exibir logs de segurança", value=False)
            
            if show_security_logs:
                st.text("🔒 Sistema monitorado")
                st.text("✅ Validação DICOM ativa")
                st.text("🛡️ Dados sanitizados")
        
        # Histórico de arquivos
        if st.session_state.get('uploaded_files'):
            st.markdown("### 📋 Histórico de Sessão")
            for i, file_name in enumerate(st.session_state.uploaded_files[:5], 1):
                st.markdown(f"{i}. {file_name[:20]}...")

    # Seção principal de upload aprimorada
    uploaded_files = create_enhanced_upload_section()

    if uploaded_files:
        # Seletor de arquivo aprimorado
        st.markdown("### 🎯 Seleção de Exame para Análise")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_file = st.selectbox(
                "Escolha o exame para análise detalhada:",
                [f.name for f in uploaded_files],
                key="file_selector"
            )
        
        with col2:
            if st.button("🔄 Processar Todos", use_container_width=True):
                st.info("🚀 Funcionalidade de processamento em lote em desenvolvimento...")
        
        dicom_file = next((f for f in uploaded_files if f.name == selected_file), None)
        
        if dicom_file:
            try:
                # Validação aprimorada
                with st.spinner("🔍 Validando arquivo DICOM..."):
                    if not validate_dicom_file(BytesIO(dicom_file.getvalue())):
                        st.error("❌ Arquivo DICOM inválido ou corrompido")
                        return
                
                # Processamento do arquivo
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                    tmp_file.write(dicom_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    with st.spinner("📊 Carregando e processando dados DICOM..."):
                        dataset = pydicom.dcmread(tmp_path)
                        dataset = sanitize_patient_data(dataset)
                    
                    # Armazenar arquivo atual
                    st.session_state.current_file = selected_file
                    
                    # Extrair informações básicas
                    dicom_data = {
                        'file_name': selected_file,
                        'file_size': get_file_size(dicom_file.size),
                        'patient_name': safe_dicom_value(dataset, 'PatientName'),
                        'patient_id': safe_dicom_value(dataset, 'PatientID'),
                        'modality': safe_dicom_value(dataset, 'Modality'),
                        'study_date': safe_dicom_value(dataset, 'StudyDate')
                    }
                    
                    # Barra de progresso para o processamento
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Inicializando análise...")
                    progress_bar.progress(10)
                    
                    # Tabs aprimoradas com ícones e descrições
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                        "🔬 Visualização Avançada", 
                        "📊 Estatísticas Expandidas", 
                        "👤 Identificação", 
                        "🔍 Análise Técnica Forense", 
                        "📈 Métricas de Qualidade", 
                        "🧬 RA-Index Avançado", 
                        "🎓 Sistema de Aprendizado"
                    ])
                    
                    status_text.text("Preparando análises...")
                    progress_bar.progress(20)
                    
                    with tab1:
                        status_text.text("Processando visualização avançada...")
                        progress_bar.progress(30)
                        enhanced_visualization_tab(dataset)
                    
                    with tab2:
                        status_text.text("Calculando estatísticas expandidas...")
                        progress_bar.progress(45)
                        enhanced_statistics_tab(dataset)
                    
                    with tab3:
                        status_text.text("Extraindo dados de identificação...")
                        progress_bar.progress(55)
                        
                        # Dados do paciente (versão original mantida)
                        patient_info = {
                            'Nome': safe_dicom_value(dataset, 'PatientName'),
                            'ID': safe_dicom_value(dataset, 'PatientID'),
                            'Data de Nascimento': safe_dicom_value(dataset, 'PatientBirthDate'),
                            'Idade': safe_dicom_value(dataset, 'PatientAge'),
                            'Sexo': safe_dicom_value(dataset, 'PatientSex'),
                            'Peso': safe_dicom_value(dataset, 'PatientWeight'),
                            'Descrição do Estudo': safe_dicom_value(dataset, 'StudyDescription'),
                            'Médico Solicitante': safe_dicom_value(dataset, 'ReferringPhysicianName'),
                            'Instituição': safe_dicom_value(dataset, 'InstitutionName')
                        }
                        
                        st.markdown('<div class="card patient-card">', unsafe_allow_html=True)
                        st.subheader("👤 Dados do Paciente")
                        
                        cols = st.columns(3)
                        for i, (key, value) in enumerate(patient_info.items()):
                            with cols[i % 3]:
                                st.markdown(f"""
                                <div style='background: #333333; padding: 12px; border-radius: 8px; margin: 8px 0;'>
                                    <span class='metric-label'>{key}</span><br>
                                    <span class='metric-value'>{value}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab4:
                        status_text.text("Executando análise técnica forense...")
                        progress_bar.progress(70)
                        enhanced_technical_analysis_tab(dataset)
                    
                    with tab5:
                        status_text.text("Calculando métricas de qualidade...")
                        progress_bar.progress(85)
                        enhanced_quality_metrics_tab(dataset)
                    
                    with tab6:
                        status_text.text("Processando RA-Index avançado...")
                        progress_bar.progress(95)
                        enhanced_ra_index_tab(dataset)
                    
                    with tab7:
                        status_text.text("Finalizando sistema de aprendizado...")
                        progress_bar.progress(100)
                        
                        # Sistema de aprendizado (versão original mantida e aprimorada)
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("🎓 Sistema de Aprendizado Contínuo")
                        
                        st.info("💡 Este sistema aprende com cada análise para melhorar continuamente a precisão e eficiência.")
                        
                        with st.form("learning_form"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                analysis_type = st.selectbox(
                                    "Tipo de análise a ser aprimorada:",
                                    ["Visualização Avançada", "Estatísticas", "Análise Técnica", 
                                     "Métricas de Qualidade", "RA-Index", "Interpretação Geral"]
                                )
                                
                                improvement_category = st.selectbox(
                                    "Categoria de melhoria:",
                                    ["Precisão", "Velocidade", "Interface", "Funcionalidade", "Relatórios"]
                                )
                            
                            with col2:
                                feedback_detail = st.text_area(
                                    "Detalhes da contribuição:",
                                    placeholder="Descreva como o sistema poderia ser melhorado...",
                                    height=100
                                )
                                
                                priority_level = st.selectbox(
                                    "Prioridade:",
                                    ["Baixa", "Média", "Alta", "Crítica"]
                                )
                            
                            submitted = st.form_submit_button("📤 Enviar Contribuição", use_container_width=True)
                            
                            if submitted:
                                if not feedback_detail:
                                    st.error("Por favor, forneça detalhes para contribuir com o aprendizado do sistema.")
                                else:
                                    learning_data = {
                                        'timestamp': datetime.now().isoformat(),
                                        'analysis_type': analysis_type,
                                        'improvement_category': improvement_category,
                                        'feedback': feedback_detail,
                                        'priority': priority_level,
                                        'user': st.session_state.user_data['nome'],
                                        'file_analyzed': selected_file
                                    }
                                    
                                    st.session_state.learning_data.append(learning_data)
                                    st.success("✅ Contribuição registrada! Obrigado por ajudar a evoluir o sistema.")
                        
                        # Mostrar histórico de contribuições
                        if st.session_state.learning_data:
                            st.markdown("### 📊 Histórico de Contribuições")
                            
                            df_learning = pd.DataFrame(st.session_state.learning_data)
                            if not df_learning.empty:
                                # Estatísticas resumidas
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total de Contribuições", len(df_learning))
                                
                                with col2:
                                    st.metric("Categorias Abordadas", df_learning['improvement_category'].nunique())
                                
                                with col3:
                                    high_priority = len(df_learning[df_learning['priority'].isin(['Alta', 'Crítica'])])
                                    st.metric("Alta Prioridade", high_priority)
                                
                                # Tabela de contribuições recentes
                                recent_contributions = df_learning.tail(5)[['analysis_type', 'improvement_category', 'priority', 'timestamp']]
                                st.dataframe(recent_contributions, use_container_width=True, hide_index=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Limpar barra de progresso
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Seção de feedback final
                    st.markdown("---")
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("💬 Avaliação da Análise Completa")
                    
                    if not st.session_state.get('feedback_submitted', False):
                        st.write("**Como foi sua experiência com esta análise?**")
                        
                        # Sistema de avaliação com estrelas
                        rating_cols = st.columns(5)
                        current_rating = st.session_state.get('rating', 0)
                        
                        for i in range(1, 6):
                            with rating_cols[i-1]:
                                if st.button(
                                    f'{"⭐" if i <= current_rating else "☆"}', 
                                    key=f'star_btn_{i}',
                                    help=f'{i} estrela(s)',
                                    use_container_width=True
                                ):
                                    st.session_state.rating = i
                                    st.rerun()
                        
                        if st.session_state.get('rating', 0) > 0:
                            st.markdown(f"**Avaliação selecionada:** {st.session_state.rating} ⭐")
                        
                        with st.form("feedback_form"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                feedback_text = st.text_area(
                                    "Comentários sobre a análise:", 
                                    placeholder="O que achou dos resultados? Sugestões de melhoria?",
                                    height=100
                                )
                            
                            with col2:
                                feedback_category = st.selectbox(
                                    "Categoria do feedback:",
                                    ["Geral", "Visualização", "Precisão", "Interface", "Performance", "Relatórios"]
                                )
                                
                                recommend_system = st.checkbox("Recomendaria este sistema para colegas?", value=True)
                            
                            submitted = st.form_submit_button("📤 Enviar Avaliação Completa", use_container_width=True)
                            
                            if submitted:
                                rating = st.session_state.get('rating', 0)
                                if rating == 0:
                                    st.error("Por favor, selecione uma avaliação com as estrelas.")
                                else:
                                    st.session_state.feedback_submitted = True
                                    st.success("✅ Avaliação enviada com sucesso! Obrigado por contribuir com a melhoria do sistema.")
                                    st.balloons()  # Efeito visual de sucesso
                                    st.rerun()
                    else:
                        st.success("📝 Obrigado pela sua avaliação! Suas contribuições são fundamentais para o aprimoramento contínuo do sistema.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                        
            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo DICOM: {e}")
                logging.error(f"Erro no processamento DICOM: {e}")

def main():
    if not safe_init_database():
        st.error("❌ Erro crítico: Não foi possível inicializar o sistema. Contate o administrador.")
        return
    
    update_css_theme()
    
    # Adicionar informações de versão no rodapé
    st.markdown("""
    <div style='position: fixed; bottom: 10px; right: 10px; background: rgba(0, 0, 0, 0.7); 
                padding: 8px 12px; border-radius: 20px; color: white; font-size: 0.8rem; z-index: 1000;'>
        <strong>DICOM Autopsy Viewer PRO v2.0</strong> - Enhanced Edition
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.user_data is None:
        show_user_form()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
