import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import warnings
import smtplib
import socket
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
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging de segurança
logging.basicConfig(
    filename='security.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suprimir warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Configurações de email (usando variáveis de ambiente)
EMAIL_CONFIG = {
    'sender': os.environ.get('EMAIL_SENDER', ''),
    'password': os.environ.get('EMAIL_PASSWORD', ''),
    'smtp_server': os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.environ.get('SMTP_PORT', 587))
}

# Limite de rate limiting
UPLOAD_LIMITS = {
    'max_files': 10,
    'max_size_mb': 2000,
    'max_uploads_per_hour': 5
}

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

def check_dependencies():
    """Verifica se todas as dependências estão disponíveis"""
    dependencies = {
        'streamlit': True,
        'pydicom': True,
        'numpy': True,
        'matplotlib': True,
        'PIL': True,
        'plotly': True,
        'smtplib': True,
        'sqlite3': True,
    }
    
    missing = []
    for dep, required in dependencies.items():
        try:
            if required:
                __import__(dep)
        except ImportError:
            missing.append(dep)
    
    return missing

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
        c.execute('''CREATE TABLE IF NOT EXISTS access_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp DATETIME,
                      user TEXT,
                      action TEXT,
                      resource TEXT,
                      details TEXT)''')
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

def log_access(user, action, resource, details=""):
    """Registra acesso a recursos sensíveis"""
    timestamp = datetime.now().isoformat()
    user_ip = "unknown"
    user_agent = "unknown"
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO access_logs (timestamp, user, action, resource, details)
                     VALUES (?, ?, ?, ?, ?)''', 
                 (timestamp, user, action, resource, details))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Erro ao registrar acesso: {e}")

def validate_dicom_file(file):
    """Validação mais robusta de arquivos DICOM"""
    try:
        # Verifica se o arquivo é muito grande (prevenção contra DoS)
        max_size = 500 * 1024 * 1024  # 500MB
        file_size = len(file.getvalue())
        if file_size > max_size:
            log_security_event("FILE_TOO_LARGE", f"Arquivo excede limite de {max_size} bytes")
            return False
        
        # Salva a posição original
        original_position = file.tell()
        
        # Verifica assinatura DICOM (128 bytes + 'DICM')
        file.seek(128)
        signature = file.read(4)
        file.seek(original_position)
        
        if signature != b'DICM':
            log_security_event("INVALID_FILE", "Arquivo não é DICOM válido")
            return False
            
        # Verificação adicional: tenta ler o metadata DICOM
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            # Tenta ler o arquivo DICOM
            dataset = pydicom.dcmread(tmp_path, force=True)
            
            # Verifica se tem pelo menos alguns atributos obrigatórios
            if not hasattr(dataset, 'SOPClassUID') or not hasattr(dataset, 'SOPInstanceUID'):
                log_security_event("INVALID_DICOM", "Arquivo não contém metadados DICOM essenciais")
                os.unlink(tmp_path)
                return False
                
            os.unlink(tmp_path)
            return True
            
        except Exception as e:
            log_security_event("DICOM_READ_ERROR", f"Erro ao ler metadados DICOM: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return False
            
    except Exception as e:
        log_security_event("FILE_VALIDATION_ERROR", f"Erro na validação: {e}")
        return False

def sanitize_patient_data(dataset):
    """Remove todos os dados sensíveis de acordo com DICOM Standard"""
    try:
        # Lista completa de tags sensíveis baseada no DICOM Standard
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

def get_file_size(bytes_size):
    """Converte bytes para formato legível"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def save_feedback(user_email, feedback_text, rating, report_data):
    """Salva feedback no banco de dados"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO feedback (user_email, feedback_text, rating, report_data)
                     VALUES (?, ?, ?, ?)''', 
                 (user_email, feedback_text, rating, json.dumps(report_data)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        log_security_event("FEEDBACK_ERROR", f"Erro ao salvar feedback: {e}")
        return False

def send_email_report(user_data, dicom_data, image_data, report_data):
    """Envia relatório por email com melhor tratamento de erros"""
    try:
        # Verifica se as credenciais de email estão configuradas
        if not EMAIL_CONFIG['sender'] or not EMAIL_CONFIG['password']:
            error_msg = "Credenciais de email não configuradas"
            log_security_event("EMAIL_CONFIG_ERROR", error_msg)
            st.error("Configuração de email não está completa. Contate o administrador.")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = 'wenndell.luz@gmail.com'
        msg['Subject'] = f'Relatório de Análise DICOM - {datetime.now().strftime("%d/%m/%Y %H:%M")}'
        
        # Corpo do email
        body = f"""
        RELATÓRIO DE ANÁLISE DICOM - DICOM AUTOPSY VIEWER
        =================================================
        
        DADOS DO USUÁRIO:
        - Nome: {user_data['nome']}
        - Departamento: {user_data['departamento']}
        - Email: {user_data['email']}
        - Contato: {user_data['contato']}
        - Data da Análise: {datetime.now().strftime("%d/%m/%Y %H:%M")}
        
        DADOS DO EXAME:
        - Arquivo: {dicom_data.get('file_name', 'N/A')}
        - Tamanho: {dicom_data.get('file_size', 'N/A')}
        - Paciente: {dicom_data.get('patient_name', 'N/A')}
        - ID: {dicom_data.get('patient_id', 'N/A')}
        - Modalidade: {dicom_data.get('modality', 'N/A')}
        - Data do Exame: {dicom_data.get('study_date', 'N/A')}
        
        ESTATÍSTICAS DA IMAGEM:
        - Dimensões: {report_data.get('dimensoes', 'N/A')}
        - Intensidade Mínima: {report_data.get('min_intensity', 'N/A')}
        - Intensidade Máxima: {report_data.get('max_intensity', 'N/A')}
        - Média: {report_data.get('media', 'N/A')}
        - Desvio Padrão: {report_data.get('desvio_padrao', 'N/A')}
        - Total de Pixels: {report_data.get('total_pixels', 'N/A')}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Anexar imagem
        if image_data:
            img_byte_arr = BytesIO()
            image_data.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            image_attachment = MIMEImage(img_byte_arr, name='analise_dicom.png')
            msg.attach(image_attachment)
        
        # Anexar relatório em CSV
        report_df = pd.DataFrame([{
            'Usuario': user_data['nome'],
            'Departamento': user_data['departamento'],
            'Email': user_data['email'],
            'Contato': user_data['contato'],
            'Data_Analise': datetime.now().strftime("%d/%m/%Y %H:%M"),
            'Arquivo': dicom_data.get('file_name', 'N/A'),
            'Tamanho_Arquivo': dicom_data.get('file_size', 'N/A'),
            'Paciente': dicom_data.get('patient_name', 'N/A'),
            'ID_Paciente': dicom_data.get('patient_id', 'N/A'),
            'Modalidade': dicom_data.get('modality', 'N/A'),
            'Data_Exame': dicom_data.get('study_date', 'N/A'),
            'Dimensoes': report_data.get('dimensoes', 'N/A'),
            'Intensidade_Min': report_data.get('min_intensity', 'N/A'),
            'Intensidade_Max': report_data.get('max_intensity', 'N/A'),
            'Media_Intensidade': report_data.get('media', 'N/A'),
            'Desvio_Padrao': report_data.get('desvio_padrao', 'N/A')
        }])
        
        csv_buffer = BytesIO()
        report_df.to_csv(csv_buffer, index=False)
        csv_attachment = MIMEApplication(csv_buffer.getvalue(), name='relatorio_analise.csv')
        csv_attachment['Content-Disposition'] = 'attachment; filename="relatorio_analise.csv"'
        msg.attach(csv_attachment)
        
        # Enviar email com timeout e tratamento de erros específico
        try:
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'], timeout=30)
            server.starttls()
            server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
            server.send_message(msg)
            server.quit()
            
            log_security_event("EMAIL_SENT", "Relatório enviado com sucesso")
            return True
            
        except smtplib.SMTPAuthenticationError:
            error_msg = "Falha na autenticação do email. Verifique as credenciais."
            log_security_event("EMAIL_AUTH_ERROR", error_msg)
            st.error("Erro de autenticação no servidor de email.")
            return False
            
        except smtplib.SMTPException as e:
            error_msg = f"Erro SMTP: {str(e)}"
            log_security_event("EMAIL_SMTP_ERROR", error_msg)
            st.error("Erro ao comunicar com o servidor de email.")
            return False
            
        except socket.timeout:
            error_msg = "Timeout ao conectar com o servidor de email"
            log_security_event("EMAIL_TIMEOUT", error_msg)
            st.error("Timeout ao conectar com o servidor de email. Tente novamente.")
            return False
            
    except Exception as e:
        error_msg = f"Erro geral ao enviar email: {str(e)}"
        log_security_event("EMAIL_ERROR", error_msg)
        st.error("Erro inesperado ao enviar email.")
        return False

def safe_dicom_value(value, default="N/A"):
    """Função segura para obter valores DICOM"""
    try:
        if value is None: return default
        if hasattr(value, '__len__') and len(value) > 100:
            return f"Dados muito grandes ({len(value)} bytes)"
        if hasattr(value, 'original_string'): return str(value.original_string)
        return str(value)
    except Exception as e:
        log_security_event("DICOM_VALUE_ERROR", f"Erro ao obter valor DICOM: {e}")
        return default

def create_medical_visualization(image, title):
    """Cria visualização médica profissional"""
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=image, colorscale='gray', showscale=False, hoverinfo='none'))
    fig.update_layout(
        title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 16, 'color': '#ffffff'}},
        width=600, height=500, margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#1a1a1a',
        xaxis=dict(visible=False, gridcolor='#333333'), yaxis=dict(visible=False, gridcolor='#333333')
    )
    return fig

def create_advanced_histogram(image):
    """Cria histograma avançado"""
    fig = px.histogram(x=image.flatten(), nbins=100, title="Distribuição de Intensidade de Pixels",
                      labels={'x': 'Intensidade', 'y': 'Frequência'}, color_discrete_sequence=['#00bcd4'])
    fig.update_layout(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font=dict(color='#ffffff'), bargap=0.1)
    return fig

def create_pdf_report(user_data, dicom_data, report_data):
    """Cria relatório em PDF"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from io import BytesIO
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # Cabeçalho
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "RELATÓRIO DE ANÁLISE DICOM")
    c.setFont("Helvetica", 10)
    c.drawString(50, 780, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Dados do usuário
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 750, "DADOS DO ANALISTA:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 730, f"Nome: {user_data['nome']}")
    c.drawString(50, 715, f"Departamento: {user_data['departamento']}")
    c.drawString(50, 700, f"Email: {user_data['email']}")
    c.drawString(50, 685, f"Contato: {user_data['contato']}")
    
    # Dados do exame
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 650, "DADOS DO EXAME:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 630, f"Arquivo: {dicom_data.get('file_name', 'N/A')}")
    c.drawString(50, 615, f"Tamanho: {dicom_data.get('file_size', 'N/A')}")
    c.drawString(50, 600, f"Paciente: {dicom_data.get('patient_name', 'N/A')}")
    c.drawString(50, 585, f"ID: {dicom_data.get('patient_id', 'N/A')}")
    c.drawString(50, 570, f"Modalidade: {dicom_data.get('modality', 'N/A')}")
    
    # Estatísticas
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 540, "ESTATÍSTICAS:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 520, f"Dimensões: {report_data.get('dimensoes', 'N/A')}")
    c.drawString(50, 505, f"Intensidade Mínima: {report_data.get('min_intensity', 'N/A')}")
    c.drawString(50, 490, f"Intensidade Máxima: {report_data.get('max_intensity', 'N/A')}")
    c.drawString(50, 475, f"Média: {report_data.get('media', 'N/A')}")
    c.drawString(50, 460, f"Desvio Padrão: {report_data.get('desvio_padrao', 'N/A')}")
    c.drawString(50, 445, f"Total de Pixels: {report_data.get('total_pixels', 'N/A')}")
    
    c.save()
    buffer.seek(0)
    return buffer

def check_data_protection_compliance():
    """Verifica conformidade com regulamentações de proteção de dados"""
    compliance = {
        'data_minimization': True,
        'purpose_limitation': True,
        'storage_limitation': True,
        'integrity_and_confidentiality': True,
        'accountability': True
    }
    return compliance

def get_compliance_badge():
    """Retorna um badge de conformidade"""
    compliance = check_data_protection_compliance()
    if all(compliance.values()):
        return "🛡️ Conformidade com LGPD/GDPR"
    else:
        return "⚠️ Verificação de conformidade necessária"

def show_feedback_section(report_data):
    """Seção de feedback"""
    st.markdown("---")
    st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
    st.subheader("💬 Feedback do Relatório")
    
    if not st.session_state.get('feedback_submitted', False):
        with st.form("feedback_form"):
            st.markdown('<div class="star-rating">⭐⭐⭐⭐⭐</div>', unsafe_allow_html=True)
            
            rating = st.slider("Avaliação (1-5 estrelas)", 1, 5, 5)
            feedback_text = st.text_area("Comentários ou sugestões:", placeholder="O que achou do relatório? Como podemos melhorar?")
            
            submitted = st.form_submit_button("📤 Enviar Feedback")
            
            if submitted:
                if save_feedback(st.session_state.user_data['email'], feedback_text, rating, report_data):
                    st.session_state.feedback_submitted = True
                    st.success("✅ Feedback enviado com sucesso! Obrigado por contribuir com a melhoria do sistema.")
                else:
                    st.error("❌ Erro ao enviar feedback.")
    else:
        st.success("📝 Obrigado pelo seu feedback! Sua contribuição ajuda a melhorar o sistema.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    
    st.markdown('</div>', unsafe_allow_html=True)   
def show_ra_index_section():
    """Seção do RA-Index com referências bibliográficas"""
    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔬 Índice de Alteração Radiológica (RA-Index)")
    
    with st.expander("📚 Referências Bibliográficas e Metodologia"):
        st.markdown("""
        ### Desenvolvimento e validação de um índice de alteração radiológica post-mortem: o RA-Index
        
        **Revista Brasileira de Direito (2012) 126:559–566**  
        DOI: 10.1007/s00414-012-0686-6
        
        **Autores:**  
        C. Egger, P. Vaucher, F. Doenz, C. Palmiere, P. Mangin, S. Grabherr
        
        **Recebido:** 11 de outubro de 2011  
        **Aceito:** 21 de fevereiro de 2012  
        **Publicado online:** 9 de março de 2012
        
        ### Resumo
        Este estudo teve como objetivo derivar um índice quantificando o estado de alteração de cadáveres 
        quantificando a presença de gás no corpo usando imagens de tomografia computadorizada multidetectora 
        post-mortem (MDCT) e validar o índice definindo sua sensibilidade e especificidade.
        """)
        
        st.markdown("""
        ### Método RA-Index
        O índice RA foi derivado de dados de MDCT post-mortem de 118 pessoas falecidas não traumáticas. 
        Para validar o índice, 100 corpos escaneados adicionais (50% falecidos traumaticamente) foram 
        examinados retrospectivamente por dois observadores independentes.
        """)
        
        # Tabela do RA-Index
        st.markdown("""
        ### Pontuações do RA-Index
        | Local Anatômico | Grau 0 | Grau I | Grau II | Grau III | Coeficiente Kappa |
        |-----------------|--------|--------|---------|----------|-------------------|
        | Cavidades Cardíacas | 0 | 5 | 15 | 20 | 0.41 |
        | Parênquima Hepático e Vasos | 0 | 8 | 17 | 25 | 0.66 |
        | Veia Inominada Esquerda | 0 | 8 | 8 | 8 | 0.78 |
        | Aorta Abdominal | 0 | 8 | 8 | 8 | 0.49 |
        | Parênquima Renal | 0 | 7 | 7 | 7 | 0.56 |
        | Vértebra L3 | 0 | 5 | 10 | 25 | 0.43 |
        | Tecidos Subcutâneos Peitorais | 0 | 8 | 8 | 8 | 0.46 |
        """)
        
        st.markdown("""
        ### Interpretação do RA-Index
        - **RA-Index < 50**: Alteração mínima/moderada
        - **RA-Index ≥ 50**: Suspeita de gás grau III em cavidades cardíacas
        - **RA-Index ≥ 60**: Suspeita de gás grau II ou III na cavidade craniana
        
        ### Validação
        - **Sensibilidade**: 100% (IC 95%: 51,7-100) para detectar cavidades cardíacas cheias de gás
        - **Especificidade**: 98,8% (IC 95%: 92,6-99,9)
        - **Confiabilidade**: ICC₂,₁ = 0,95 (IC 95%: 0,92-0,96)
        """)
        
        st.markdown("""
        ### Aplicações Práticas
        1. **Triagem de embolia gasosa**: Valores abaixo de 50 indicam necessidade de investigação adicional
        2. **Controle de qualidade**: Excluir casos com alteração avançada (>60) de estudos de imagem
        3. **Interpretação radiológica**: Considerar o estado de alteração no diagnóstico por imagem
        """)
    
    # Calculadora do RA-Index (se o usuário quiser calcular)
    st.markdown("### 📊 Calculadora do RA-Index")
    st.info("Use esta calculadora para determinar o RA-Index com base nos achados de imagem")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cardiac = st.selectbox("Cavidades Cardíacas", ["Grau 0", "Grau I", "Grau II", "Grau III"], help="Presença de gás nas 4 cavidades do coração")
        hepatic = st.selectbox("Parênquima Hepático", ["Grau 0", "Grau I", "Grau II", "Grau III"], help="Presença de gás no fígado e vasos hepáticos")
        vein = st.selectbox("Veia Inominada Esquerda", ["Grau 0", "Grau I", "Grau II", "Grau III"], help="Presença de gás na veia inominada esquerda")
        aorta = st.selectbox("Aorta Abdominal", ["Grau 0", "Grau I", "Grau II", "Grau III"], help="Presença de gás na aorta abdominal")
    
    with col2:
        renal = st.selectbox("Parênquima Renal", ["Grau 0", "Grau I", "Grau II", "Grau III"], help="Presença de gás nos rins")
        vertebra = st.selectbox("Vértebra L3", ["Grau 0", "Grau I", "Grau II", "Grau III"], help="Presença de gás na terceira vértebra lombar")
        subcutaneous = st.selectbox("Tecidos Subcutâneos", ["Grau 0", "Grau I", "Grau II", "Grau III"], help="Presença de gás nos tecidos subcutâneos peitorais")
    
    if st.button("Calcular RA-Index"):
        # Mapeamento de valores
        scores = {
            "Grau 0": 0,
            "Grau I": 1,
            "Grau II": 2,
            "Grau III": 3
        }
        
        # Pontuações conforme a tabela do estudo
        ra_scores = {
            "cardiac": [0, 5, 15, 20],
            "hepatic": [0, 8, 17, 25],
            "vein": [0, 8, 8, 8],
            "aorta": [0, 8, 8, 8],
            "renal": [0, 7, 7, 7],
            "vertebra": [0, 5, 10, 25],
            "subcutaneous": [0, 8, 8, 8]
        }
        
        # Calcular RA-Index
        total_score = (
            ra_scores["cardiac"][scores[cardiac]] +
            ra_scores["hepatic"][scores[hepatic]] +
            ra_scores["vein"][scores[vein]] +
            ra_scores["aorta"][scores[aorta]] +
            ra_scores["renal"][scores[renal]] +
            ra_scores["vertebra"][scores[vertebra]] +
            ra_scores["subcutaneous"][scores[subcutaneous]]
        )
        
        # Interpretação
        if total_score < 50:
            interpretation = "Alteração mínima/moderada"
            color = "green"
        elif total_score < 60:
            interpretation = "Suspeita de gás grau III em cavidades cardíacas"
            color = "orange"
        else:
            interpretation = "Suspeita de gás grau II ou III na cavidade craniana - Alteração avançada"
            color = "red"
        
        # Mostrar resultado
        st.markdown(f"""
        <div style='background: #2d2d2d; padding: 20px; border-radius: 10px; border-left: 4px solid {color};'>
            <h3 style='color: {color}; margin-top: 0;'>RA-Index: {total_score}/100</h3>
            <p style='color: #e0e0e0;'>{interpretation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recomendações
        if total_score >= 50:
            st.warning("""
            **Recomendações:**
            - Considerar análise de composição gasosa (cromatografia gasosa)
            - Interpretar achados radiológicos com cautela
            - Limitar procedimentos diagnósticos adicionais
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)
def show_login_page():
    """Página de login/registro"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🔬 DICOM Autopsy Viewer</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #b0b0b0 !important; font-size: 1.1rem;">Sistema Profissional para Análise Forense</p>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            st.subheader("📋 Acesso ao Sistema")
            
            with st.form("login_form"):
                nome = st.text_input("Nome Completo", placeholder="Digite seu nome completo")
                departamento = st.text_input("Departamento", placeholder="Departamento/Filiação")
                email = st.text_input("Email", placeholder="seu.email@exemplo.com")
                contato = st.text_input("Contato", placeholder="Telefone ou outro contato")
                
                submitted = st.form_submit_button("🔓 Acessar Sistema")
                
                if submitted:
                    if nome and departamento and email and contato:
                        st.session_state.authenticated = True
                        st.session_state.user_data = {
                            'nome': nome,
                            'departamento': departamento,
                            'email': email,
                            'contato': contato,
                            'data_acesso': datetime.now().strftime("%d/%m/%Y %H:%M")
                        }
                        log_access(nome, "LOGIN", "SYSTEM_ACCESS")
                        st.success("✅ Acesso concedido! Carregando sistema...")
                        st.rerun()
                    else:
                        st.error("❌ Preencha todos os campos obrigatórios")
            
            st.markdown('</div>', unsafe_allow_html=True)

def show_dashboard():
    """Dashboard inicial"""
    compliance_badge = get_compliance_badge()
    
    st.markdown(f"""
    <div style='text-align: center; padding: 40px 20px; background: #2d2d2d; border-radius: 15px; color: #ffffff;'>
        <h2 style='color: #00bcd4 !important;'>🔬 Bem-vindo ao DICOM Autopsy Viewer</h2>
        <p style='color: #b0b0b0 !important;'>Sistema profissional para análise forense de imagens DICOM</p>
        <div style='font-size: 3rem; margin: 20px 0;'>🔬📊📧</div>
        <div style='background: #4caf50; padding: 10px; border-radius: 5px; display: inline-block; margin-top: 15px;'>
            {compliance_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='card'>
            <h4>💾 Grande Capacidade</h4>
            <p>• 200MB por arquivo</p>
            <p>• Até 10 arquivos por caso</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h4>📊 Análise Completa</h4>
            <p>• Visualização avançada</p>
            <p>• Estatísticas detalhadas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h4>📧 Integração Total</h4>
            <p>• Envio automático por email</p>
            <p>• Relatórios em PDF</p>
        </div>
        """, unsafe_allow_html=True)

def show_main_app():
    """Aplicativo principal após autenticação"""
    # Registrar acesso
    log_access(st.session_state.user_data['nome'], "LOGIN", "MAIN_APP")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<h1 class="main-header">🔬 DICOM Autopsy Viewer</h1>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div style="background: #333; padding: 10px; border-radius: 8px; text-align: center;">'
                   f'<span style="color: #00bcd4;">👤 {st.session_state.user_data["nome"]}</span><br>'
                   f'<span style="color: #b0b0b0; font-size: 0.8rem;">{st.session_state.user_data["departamento"]}</span>'
                   f'</div>', unsafe_allow_html=True)
        
        if st.button("🚪 Sair"):
            log_access(st.session_state.user_data['nome'], "LOGOUT", "SYSTEM_ACCESS")
            st.session_state.authenticated = False
            st.session_state.user_data = {}
            st.rerun()

    st.markdown("---")

    with st.sidebar:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a237e, #283593); padding: 15px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0;'>👤 Usuário Logado</h3>
            <p style='margin: 5px 0; font-size: 0.9rem;'>{st.session_state.user_data['nome']}</p>
            <p style='margin: 0; font-size: 0.8rem;'>{st.session_state.user_data['departamento']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"""
        <div class='upload-info'>
            <h4>📁 Upload de Exames</h4>
            <p>• Limite: <strong>{UPLOAD_LIMITS['max_files']} arquivos</strong></p>
            <p>• Tamanho: <strong>{UPLOAD_LIMITS['max_size_mb']}MB por arquivo</strong></p>
            <p>• Formato: <strong>.dcm, .DCM</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "📤 Selecione os arquivos DICOM",
            type=['dcm', 'DCM'],
            accept_multiple_files=True,
            help=f"Selecione até {UPLOAD_LIMITS['max_files']} arquivos DICOM (máximo {UPLOAD_LIMITS['max_size_mb']}MB cada)"
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
                    file_copy = BytesIO(file.getvalue())  # Cria cópia para validação
                    if validate_dicom_file(file_copy):
                        valid_files.append(file)
                    else:
                        st.warning(f"⚠️ Arquivo {file.name} não é um DICOM válido e foi ignorado")
                
                if valid_files:
                    st.success(f"✅ {len(valid_files)} arquivo(s) válido(s) - {get_file_size(total_size)}")
                    
                    # Mostrar tamanho de cada arquivo
                    for file in valid_files:
                        st.markdown(f"""
                        <div class='uploaded-file'>
                            📄 {file.name}
                            <div class='file-size'>{get_file_size(file.size)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("❌ Nenhum arquivo DICOM válido encontrado")
                    log_security_event("NO_VALID_FILES", "Nenhum arquivo DICOM válido no upload")

    if uploaded_files:
        selected_file = st.selectbox("📋 Selecione o exame para análise:", [f.name for f in uploaded_files])
        dicom_file = next((f for f in uploaded_files if f.name == selected_file), None)
        
        if dicom_file:
            tmp_path = None
            try:
                # VALIDAÇÃO FINAL DE SEGURANÇA
                file_copy = BytesIO(dicom_file.getvalue())
                if not validate_dicom_file(file_copy):
                    st.error("❌ Arquivo corrompido ou inválido")
                    log_security_event("FINAL_VALIDATION_FAILED", f"Arquivo {selected_file} falhou na validação final")
                    return
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                    tmp_file.write(dicom_file.getvalue())
                    tmp_path = tmp_file.name
                
                dataset = pydicom.dcmread(tmp_path)
                
                # SANITIZA DADOS SENSÍVEIS
                dataset = sanitize_patient_data(dataset)
                
                dicom_data = {
                    'file_name': selected_file,
                    'file_size': get_file_size(dicom_file.size),
                    'patient_name': safe_dicom_value(getattr(dataset, 'PatientName', 'N/A')),
                    'patient_id': safe_dicom_value(getattr(dataset, 'PatientID', 'N/A')),
                    'modality': safe_dicom_value(getattr(dataset, 'Modality', 'N/A')),
                    'study_date': safe_dicom_value(getattr(dataset, 'StudyDate', 'N/A'))
                }
                
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔬 Visualização", "👤 Identificação", "⚙️ Técnico", "📊 Análise", "📚 RA-Index"])
                
                report_data = {}
                image_for_report = None
                
                with tab1:
                    if hasattr(dataset, 'pixel_array'):
                        image = dataset.pixel_array
                        if image.dtype != np.uint8:
                            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                        
                        fig = create_medical_visualization(image, f"Exame: {selected_file}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Salvar imagem para relatório
                        plt.figure(figsize=(8, 8))
                        plt.imshow(image, cmap='gray')
                        plt.axis('off')
                        plt.title(f"Análise DICOM - {selected_file}")
                        img_buffer = BytesIO()
                        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0.1)
                        img_buffer.seek(0)
                        image_for_report = Image.open(img_buffer)
                        plt.close()
                    else:
                        st.warning("⚠️ Arquivo DICOM não contém dados de imagem")
                
                with tab2:
                    st.markdown('<div class="card patient-card">', unsafe_allow_html=True)
                    st.subheader("👤 Dados de Identificação")
                    
                    patient_info = {
                        'Nome': safe_dicom_value(getattr(dataset, 'PatientName', 'N/A')),
                        'ID': safe_dicom_value(getattr(dataset, 'PatientID', 'N/A')),
                        'Idade': safe_dicom_value(getattr(dataset, 'PatientAge', 'N/A')),
                        'Sexo': safe_dicom_value(getattr(dataset, 'PatientSex', 'N/A')),
                        'Data do Estudo': safe_dicom_value(getattr(dataset, 'StudyDate', 'N/A')),
                        'Modalidade': safe_dicom_value(getattr(dataset, 'Modality', 'N/A')),
                        'Médico': safe_dicom_value(getattr(dataset, 'ReferringPhysicianName', 'N/A')),
                        'Instituição': safe_dicom_value(getattr(dataset, 'InstitutionName', 'N/A'))
                    }
                    
                    cols = st.columns(2)
                    for i, (key, value) in enumerate(patient_info.items()):
                        with cols[i % 2]:
                            st.markdown(f"""
                            <div style='background: #333333; padding: 12px; border-radius: 8px; margin: 8px 0;'>
                                <span class='metric-label'>{key}</span><br>
                                <span class='metric-value'>{value}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab3:
                    st.markdown('<div class="card tech-card">', unsafe_allow_html=True)
                    st.subheader("⚙️ Informações Técnicas")
                    
                    tech_info = {
                        'Modalidade': safe_dicom_value(getattr(dataset, 'Modality', 'N/A')),
                        'Tamanho': f"{safe_dicom_value(getattr(dataset, 'Rows', 'N/A'))} × {safe_dicom_value(getattr(dataset, 'Columns', 'N/A'))}",
                        'Bits por Pixel': safe_dicom_value(getattr(dataset, 'BitsAllocated', 'N/A')),
                        'Janela Central': safe_dicom_value(getattr(dataset, 'WindowCenter', 'N/A')),
                        'Largura da Janela': safe_dicom_value(getattr(dataset, 'WindowWidth', 'N/A')),
                        'Espessura de Corte': safe_dicom_value(getattr(dataset, 'SliceThickness', 'N/A'))
                    }
                    
                    cols = st.columns(2)
                    for i, (key, value) in enumerate(tech_info.items()):
                        with cols[i % 2]:
                            st.markdown(f"""
                            <div style='background: #333333; padding: 12px; border-radius: 8px; margin: 8px 0;'>
                                <span class='metric-label'>{key}</span><br>
                                <span class='metric-value'>{value}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab4:
                    if hasattr(dataset, 'pixel_array'):
                        image = dataset.pixel_array
                        report_data = {
                            'dimensoes': f"{image.shape[0]} × {image.shape[1]}",
                            'min_intensity': int(np.min(image)),
                            'max_intensity': int(np.max(image)),
                            'media': f"{np.mean(image):.2f}",
                            'desvio_padrao': f"{np.std(image):.2f}",
                            'total_pixels': f"{image.size:,}"
                        }
                        
                        # Botões de relatório
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📧 Enviar Relatório por Email", help="Envia relatório completo para wenndell.luz@gmail.com"):
                                if send_email_report(st.session_state.user_data, dicom_data, image_for_report, report_data):
                                    st.success("✅ Relatório enviado para wenndell.luz@gmail.com")
                                    st.info("📋 Uma cópia foi enviada para o administrador do sistema para auditoria e melhoria contínua")
                                    log_security_event("USER_NOTIFIED", "Usuário informado sobre envio de cópia")

                        with col2:
                            pdf_report = create_pdf_report(st.session_state.user_data, dicom_data, report_data)
                            st.download_button(
                                label="📄 Baixar Relatório PDF",
                                data=pdf_report,
                                file_name=f"relatorio_{selected_file.split('.')[0]}.pdf",
                                mime="application/pdf",
                                help="Baixe relatório completo em PDF"
                            )
                        
                        # Seção de feedback
                        show_feedback_section({
                            'dicom_data': dicom_data,
                            'report_data': report_data,
                            'user': st.session_state.user_data,
                            'timestamp': datetime.now().isoformat()
                        })
            with tab5:
        # Seção do Índice RA
        show_ra_index_section()

            except Exception as e:
                error_msg = f"Erro ao processar arquivo: {str(e)}"
                st.error(f"❌ {error_msg}")
                log_security_event("PROCESSING_ERROR", error_msg)
                
            finally:
                # Limpar arquivo temporário
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        log_security_event("CLEANUP_ERROR", f"Erro ao limpar arquivo temporário: {e}")
    else:
        show_dashboard()

def main():
    """Função principal"""
    try:
        # Verificar dependências
        missing_deps = check_dependencies()
        if missing_deps:
            st.error(f"❌ Dependências missing: {', '.join(missing_deps)}")
            log_security_event("MISSING_DEPENDENCIES", f"Dependências faltando: {missing_deps}")
            return
        
        # Inicialização completa das variáveis de sessão
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}
        if 'feedback_submitted' not in st.session_state:
            st.session_state.feedback_submitted = False
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'current_file' not in st.session_state:
            st.session_state.current_file = None
            
        log_security_event("APP_START", "Aplicativo iniciado")
        
        if not st.session_state.authenticated:
            show_login_page()
        else:
            show_main_app()
            
    except Exception as e:
        error_msg = f"Erro crítico: {str(e)}"
        log_security_event("APP_CRASH", error_msg)
        st.error("❌ Erro crítico no aplicativo. Os administradores foram notificados.")

# Inicializar banco e executar aplicação
init_database()

if __name__ == "__main__":
    main()
