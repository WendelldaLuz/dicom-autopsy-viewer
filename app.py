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
</style>
""", unsafe_allow_html=True)

# Configuração do banco de dados para feedback
DB_PATH = "feedback_database.db"

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
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erro ao inicializar banco: {e}")

init_database()

# Configurações de email
EMAIL_CONFIG = {
    'sender': 'wenndell.luz@gmail.com',
    'password': 'sua_senha_do_email',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
}

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
        print(f"Erro ao salvar feedback: {e}")
        return False

def log_error(error_type, error_message, solution=""):
    """Registra erros para aprendizado do sistema"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''SELECT id, occurrence_count FROM system_learning 
                     WHERE error_type = ? AND error_message = ?''', 
                 (error_type, error_message))
        result = c.fetchone()
        
        if result:
            c.execute('''UPDATE system_learning 
                         SET occurrence_count = occurrence_count + 1, 
                             last_occurrence = CURRENT_TIMESTAMP,
                             solution_applied = ?
                         WHERE id = ?''', (solution, result[0]))
        else:
            c.execute('''INSERT INTO system_learning (error_type, error_message, solution_applied)
                         VALUES (?, ?, ?)''', (error_type, error_message, solution))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erro ao registrar erro: {e}")

def get_file_size(bytes_size):
    """Converte bytes para formato legível"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def send_email_report(user_data, dicom_data, image_data, report_data):
    """Envia relatório por email"""
    try:
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
        
        # Enviar email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        error_msg = f"Erro ao enviar email: {str(e)}"
        log_error("EMAIL_ERROR", error_msg, "Verificar configurações SMTP")
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
        log_error("DICOM_VALUE_ERROR", str(e), "Valor padrão aplicado")
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
                        st.success("✅ Acesso concedido! Carregando sistema...")
                        st.rerun()
                    else:
                        st.error("❌ Preencha todos os campos obrigatórios")
            
            st.markdown('</div>', unsafe_allow_html=True)

def show_dashboard():
    """Dashboard inicial"""
    st.markdown("""
    <div style='text-align: center; padding: 40px 20px; background: #2d2d2d; border-radius: 15px; color: #ffffff;'>
        <h2 style='color: #00bcd4 !important;'>🔬 Bem-vindo ao DICOM Autopsy Viewer</h2>
        <p style='color: #b0b0b0 !important;'>Sistema profissional para análise forense de imagens DICOM</p>
        <div style='font-size: 3rem; margin: 20px 0;'>🔬📊📧</div>
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
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<h1 class="main-header">🔬 DICOM Autopsy Viewer</h1>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div style="background: #333; padding: 10px; border-radius: 8px; text-align: center;">'
                   f'<span style="color: #00bcd4;">👤 {st.session_state.user_data["nome"]}</span><br>'
                   f'<span style="color: #b0b0b0; font-size: 0.8rem;">{st.session_state.user_data["departamento"]}</span>'
                   f'</div>', unsafe_allow_html=True)
        
        if st.button("🚪 Sair"):
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
        
        st.markdown("""
        <div class='upload-info'>
            <h4>📁 Upload de Exames</h4>
            <p>• Limite: <strong>200MB por arquivo</strong></p>
            <p>• Máximo: <strong>10 arquivos por caso</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "📤 Selecione os arquivos DICOM",
            type=['dcm', 'DCM'],
            accept_multiple_files=True,
            help="Selecione até 10 arquivos DICOM"
        )
        
        if uploaded_files:
            total_size = sum(f.size for f in uploaded_files)
            st.success(f"✅ {len(uploaded_files)} arquivo(s) - {get_file_size(total_size)}")

    if uploaded_files:
        selected_file = st.selectbox("📋 Selecione o exame para análise:", [f.name for f in uploaded_files])
        dicom_file = next((f for f in uploaded_files if f.name == selected_file), None)
        
        if dicom_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                    tmp_file.write(dicom_file.getvalue())
                    tmp_path = tmp_file.name
                
                dataset = pydicom.dcmread(tmp_path)
                
                dicom_data = {
                    'file_name': selected_file,
                    'file_size': get_file_size(dicom_file.size),
                    'patient_name': safe_dicom_value(getattr(dataset, 'PatientName', 'N/A')),
                    'patient_id': safe_dicom_value(getattr(dataset, 'PatientID', 'N/A')),
                    'modality': safe_dicom_value(getattr(dataset, 'Modality', 'N/A')),
                    'study_date': safe_dicom_value(getattr(dataset, 'StudyDate', 'N/A'))
                }
                
                tab1, tab2, tab3, tab4 = st.tabs(["🔬 Visualização", "👤 Identificação", "⚙️ Técnico", "📊 Análise"])
                
                report_data = {}
                image_for_report = None
                
                with tab1:
                    if hasattr(dataset, 'pixel_array'):
                        image = dataset.pixel_array
                        if image.dtype != np.uint8:
                            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                        
                        fig = create_medical_visualization(image, f"Exame: {selected_file}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        plt.figure(figsize=(8, 8))
                        plt.imshow(image, cmap='gray')
                        plt.axis('off')
                        plt.title(f"Análise DICOM - {selected_file}")
                        img_buffer = BytesIO()
                        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0.1)
                        img_buffer.seek(0)
                        image_for_report = Image.open(img_buffer)
                        plt.close()
                
                with tab4:
                    if hasattr(dataset, 'pixel_array'):
                        report_data = {
                            'dimensoes': f"{image.shape[0]} × {image.shape[1]}",
                            'min_intensity': int(np.min(image)),
                            'max_intensity': int(np.max(image)),
                            'media': f"{np.mean(image):.2f}",
                            'desvio_padrao': f"{np.std(image):.2f}",
                            'total_pixels': f"{image.size:,}"
                        }
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📧 Enviar Relatório por Email"):
                                if send_email_report(st.session_state.user_data, dicom_data, image_for_report, report_data):
                                    st.success("✅ Relatório enviado para wenndell.luz@gmail.com")
                        
                        with col2:
                            pdf_report = create_pdf_report(st.session_state.user_data, dicom_data, report_data)
                            st.download_button(
                                label="📄 Baixar Relatório PDF",
                                data=pdf_report,
                                file_name=f"relatorio_{selected_file.split('.')[0]}.pdf",
                                mime="application/pdf"
                            )
                        
                        show_feedback_section({
                            'dicom_data': dicom_data,
                            'report_data': report_data,
                            'user': st.session_state.user_data
                        })
                
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo: {str(e)}")
    else:
        show_dashboard()

def main():
    """Função principal"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_data = {}
        st.session_state.feedback_submitted = False
    
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_app()

# EXECUTAR A APLICAÇÃO
if __name__ == "__main__":
    main()
