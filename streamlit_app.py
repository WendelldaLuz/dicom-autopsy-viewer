import sqlite3
import logging
import pydicom
import streamlit as st
import numpy as np
import tempfile
import os
from datetime import datetime

# Configurações iniciais da página
st.set_page_config(
    page_title="DICOM Autopsy Viewer PRO",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração básica de logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Banco de dados ---

def safe_init_database():
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL,
                department TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT,
                action TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                details TEXT
            )
        """)
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

def log_security_event(user_email, action, details=""):
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        ip_address = "127.0.0.1"  # Pode ser substituído por IP real
        cursor.execute("""
            INSERT INTO security_logs (user_email, action, ip_address, details)
            VALUES (?, ?, ?, ?)
        """, (user_email, action, ip_address, details))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Erro ao registrar evento de segurança: {e}")

def save_user(name, email, role, department):
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (name, email, role, department)
            VALUES (?, ?, ?, ?)
        """, (name, email, role, department))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        st.error("Email já cadastrado. Por favor, use outro email.")
        return False
    except Exception as e:
        st.error(f"Erro ao salvar usuário: {e}")
        return False

def get_user_reports(user_email):
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, report_name, generated_at
            FROM reports
            WHERE user_email = ?
            ORDER BY generated_at DESC
        """, (user_email,))
        reports = cursor.fetchall()
        conn.close()
        return reports
    except Exception as e:
        logging.error(f"Erro ao recuperar relatórios: {e}")
        return []

# --- CSS para tema claro e profissional ---

def update_css_theme():
    st.markdown("""
    <style>
    body, .main, .stApp {
        background-color: #FFFFFF;
        color: #000000;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #333333 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border-bottom: 2px solid #000000;
    }
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        background-color: #000000;
        color: #FFFFFF;
        padding: 8px 16px;
        border-radius: 4px 0 0 0;
        font-size: 0.8rem;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        DICOM Autopsy Viewer PRO v3.0 | Interface Profissional | © 2025
    </div>
    """, unsafe_allow_html=True)

# --- Formulário de registro de usuário ---

def show_user_form():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>DICOM Autopsy Viewer PRO</h1>
        <h3>Sistema Avançado de Análise Forense Digital</h3>
    </div>
    """, unsafe_allow_html=True)
    with st.form("user_registration"):
        name = st.text_input("Nome Completo *", placeholder="Dr. João Silva")
        email = st.text_input("Email Institucional *", placeholder="joao.silva@hospital.com")
        col1, col2 = st.columns(2)
        with col1:
            role = st.selectbox("Função *", ["Radiologista", "Médico Legista", "Técnico em Radiologia", "Pesquisador", "Estudante", "Outro"])
        with col2:
            department = st.text_input("Departamento/Instituição", placeholder="Departamento de Radiologia")
        with st.expander("Termos de Uso e Política de Privacidade"):
            st.markdown("""
            **Termos de Uso:**
            1. Utilização autorizada apenas para fins educacionais e de pesquisa
            2. Proibido o carregamento de dados de pacientes reais sem autorização apropriada
            3. Compromisso com a confidencialidade das informações processadas
            4. Os relatórios gerados são de responsabilidade do usuário
            5. O sistema não armazena imagens médicas, apenas metadados anônimos

            **Política de Privacidade:**
            - Seus dados de registro são armazenados de forma segura
            - As análises realizadas são confidenciais
            - Metadados das imagens são anonimizados para análise estatística
            - Relatórios gerados podem ser excluídos a qualquer momento
            """)
            terms_accepted = st.checkbox("Eu concordo com os termos de uso e política de privacidade")
        submitted = st.form_submit_button("Iniciar Sistema →")
        if submitted:
            if not all([name, email, terms_accepted]):
                st.error("Por favor, preencha todos os campos obrigatórios e aceite os termos de uso.")
            else:
                if save_user(name, email, role, department):
                    st.session_state.user_data = {'name': name, 'email': email, 'role': role, 'department': department}
                    log_security_event(email, "USER_REGISTRATION", f"Role: {role}")
                    st.success("Usuário registrado com sucesso!")
                    st.experimental_rerun()

# --- Tela principal após login ---

def show_main_app():
    user_data = st.session_state.user_data
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 1rem; border-bottom: 1px solid #E0E0E0; margin-bottom: 1rem;">
            <h3>{user_data['name']}</h3>
            <p><strong>Função:</strong> {user_data['role']}</p>
            <p><strong>Email:</strong> {user_data['email']}</p>
            {f'<p><strong>Departamento:</strong> {user_data["department"]}</p>' if user_data['department'] else ''}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Navegação")
        uploaded_file = st.file_uploader(
            "Selecione um arquivo DICOM:",
            type=['dcm', 'dicom'],
            help="Carregue um arquivo DICOM para análise forense avançada"
        )

        st.markdown("---")
        st.markdown("### Relatórios Salvos")
        user_reports = get_user_reports(user_data['email'])
        if user_reports:
            for report_id, report_name, generated_at in user_reports:
                if st.button(f"{report_name} - {generated_at.split()[0]}", key=f"report_{report_id}"):
                    st.session_state.selected_report = report_id
        else:
            st.info("Nenhum relatório salvo ainda.")

        st.markdown("---")
        with st.expander("Informações do Sistema"):
            st.write("**Versão:** 3.0 Professional")
            st.write("**Última Atualização:** 2025-09-15")
            st.write("**Status:** Online")
            st.write("**Armazenamento:** 2.5 GB disponíveis")

        if st.button("Trocar Usuário", use_container_width=True):
            st.session_state.user_data = None
            st.experimental_rerun()

    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <h1>DICOM Autopsy Viewer</h1>
        <span style="background-color: #000; color: #fff; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
            v3.0 Professional
        </span>
    </div>
    <p>Bem-vindo, <strong>{user_data['name']}</strong>! Utilize as ferramentas abaixo para análise forense avançada de imagens DICOM.</p>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            log_security_event(user_data['email'], "FILE_UPLOAD", f"Filename: {uploaded_file.name}")

            try:
                dicom_data = pydicom.dcmread(tmp_path)
                image_array = dicom_data.pixel_array

                st.session_state.dicom_data = dicom_data
                st.session_state.image_array = image_array
                st.session_state.uploaded_file_name = uploaded_file.name

                st.markdown("### Informações do Arquivo")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Dimensões", f"{image_array.shape[0]} × {image_array.shape[1]}")
                with col2:
                    st.metric("Tipo de Dados", str(image_array.dtype))
                with col3:
                    st.metric("Faixa de Valores", f"{image_array.min()} → {image_array.max()}")
                with col4:
                    st.metric("Tamanho do Arquivo", f"{uploaded_file.size / 1024:.1f} KB")

                # Aqui você pode chamar as funções para as abas, ex:
                # enhanced_visualization_tab(dicom_data, image_array)
                # enhanced_statistics_tab(dicom_data, image_array)
                # etc.

                st.info("Funcionalidades das abas devem ser implementadas conforme necessidade.")

            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            st.error(f"Erro ao processar arquivo DICOM: {e}")
            logging.error(f"Erro no processamento DICOM: {e}")
    else:
        st.info("Carregue um arquivo DICOM na sidebar para começar a análise.")

# --- Main ---

def main():
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'dicom_data' not in st.session_state:
        st.session_state.dicom_data = None
    if 'image_array' not in st.session_state:
        st.session_state.image_array = None
    if 'current_report' not in st.session_state:
        st.session_state.current_report = None

    if not safe_init_database():
        st.error("Erro crítico: Não foi possível inicializar o sistema. Contate o administrador.")
        return

    update_css_theme()

    if st.session_state.user_data is None:
        show_user_form()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
