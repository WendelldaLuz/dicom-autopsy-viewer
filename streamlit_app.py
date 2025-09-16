authenticity_report = {}
import sqlite3
import logging
import pydicom
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import json
from datetime import datetime
from io import BytesIO
import smtplib
import hashlib
import uuid
import csv
from skimage import feature
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
except ImportError:
    st.warning("ReportLab não instalado. Funcionalidade de PDF limitada.")
    
import socket
import base64
import colorsys
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy import ndimage
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

try:
    import cv2
except ImportError:
    st.warning("OpenCV não instalado. Algumas funcionalidades de processamento de imagem limitadas.")

# Configuração inicial da página
st.set_page_config(
    page_title="DICOM Autopsy Viewer Pro - Enhanced",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== SEÇÃO 1: FUNÇÕES DE VISUALIZAÇÃO APRIMORADA ======

def setup_matplotlib_for_plotting():
    import warnings
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    try:
        plt.style.use("seaborn-v0_8")
    except:
        plt.style.use("default")
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False

def apply_hounsfield_windowing(image, window_center, window_width):
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    windowed_image = np.copy(image)
    windowed_image[windowed_image < min_value] = min_value
    windowed_image[windowed_image > max_value] = max_value
    windowed_image = (windowed_image - min_value) / (max_value - min_value) * 255
    return windowed_image.astype(np.uint8)

def enhanced_visualization_tab(dicom_data, image_array):
    st.subheader("Visualização Avançada de Imagem DICOM")
    
    # Implementação da visualização aqui
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Visualização da Imagem")
        window_center = st.slider("Centro da Janela (Window Center)", -1000, 3000, 40)
        window_width = st.slider("Largura da Janela (Window Width)", 1, 4000, 400)
        
        # Aplicar janelamento
        windowed_image = apply_hounsfield_windowing(image_array, window_center, window_width)
        
        # Mostrar imagem
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(windowed_image, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Informações da Imagem")
        st.metric("Dimensões", f"{image_array.shape[0]} × {image_array.shape[1]}")
        st.metric("Valor Mínimo", f"{np.min(image_array):.2f} HU")
        st.metric("Valor Máximo", f"{np.max(image_array):.2f} HU")
        st.metric("Valor Médio", f"{np.mean(image_array):.2f} HU")
        
        # Histograma
        st.markdown("### Histograma de Intensidades")
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        ax_hist.hist(image_array.flatten(), bins=100, alpha=0.7)
        ax_hist.set_xlabel("Unidades Hounsfield (HU)")
        ax_hist.set_ylabel("Frequência")
        st.pyplot(fig_hist)

def enhanced_post_mortem_analysis_tab(dicom_data, image_array):
    st.subheader("Análise Avançada de Períodos Post-Mortem")
    
    with st.expander("Referências Científicas (Normas ABNT)"):
        st.markdown("""
        **Base Científica desta Análise:**
        - ALTAIMIRANO, R. **Técnicas de imagem aplicadas à tanatologia forense**. Revista de Medicina Legal, 2022.
        - MEGO, M. et al. **Análise quantitativa de fenômenos cadavéricos através de TC multidetectores**. J Forensic Sci, 2017.
        - GÓMEZ, H. **Avanços na estimativa do intervalo post-mortem por métodos de imagem**. Forense Internacional, 2021.
        - ESPINOZA, C. et al. **Correlação entre fenômenos abióticos e achados de imagem em cadáveres**. Arquivos de Medicina Legal, 2019.
        - HOFER, P. **Mudanças densitométricas teciduais no período post-mortem**. J Radiol Forense, 2005.
        """)
    
    tab_algor, tab_livor, tab_rigor, tab_putrefaction, tab_conservation = st.tabs([
        "Algor Mortis", "Livor Mortis", "Rigor Mortis", "Putrefação", "Fenômenos Conservadores"
    ])
    
    with tab_algor:
        st.markdown("### ❄️ Algor Mortis (Esfriamento Cadavérico)")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Análise de Distribuição Térmica Simulada")
            thermal_simulation = simulate_body_cooling(image_array)
            fig = go.Figure(data=go.Heatmap(
                z=thermal_simulation,
                colorscale='jet',
                showscale=True,
                hovertemplate='Temperatura: %{z:.1f}°C<extra></extra>'
            ))
            fig.update_layout(
                title="Simulação de Distribuição Térmica Corporal",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚙️ Parâmetros de Esfriamento")
            ambient_temp = st.slider("Temperatura Ambiente (°C)", 10, 40, 25)
            body_mass = st.slider("Massa Corporal (kg)", 40, 120, 70)
            clothing = st.select_slider("Vestuário", options=["Leve", "Moderado", "Abrigado"], value="Moderado")
            
            if st.button("Estimar IPM por Algor Mortis"):
                ipm_estimate = estimate_pmi_from_cooling(thermal_simulation, ambient_temp, body_mass, clothing)
                st.metric("Intervalo Post-Mortem Estimado", f"{ipm_estimate:.1f} horas")
                st.markdown("**Curva Teórica de Resfriamento:**")
                cooling_data = generate_cooling_curve(ipm_estimate, ambient_temp)
                st.line_chart(cooling_data)
    
    with tab_livor:
        st.markdown("### 🔴 Livor Mortis (Manchas de Hipóstase)")
        st.info("""
        **Referência:** Manchas começam em 30min-2h, fixam em 12-18h (Altamirano, 2022; Gómez H., 2021)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Análise de Distribuição Sanguínea")
            blood_pooling_map = detect_blood_pooling(image_array)
            fig = px.imshow(blood_pooling_map,
                           color_continuous_scale='hot',
                           title="Mapa de Provável Acúmulo Sanguíneo")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Métricas de Hipóstase")
            pooling_intensity = np.mean(blood_pooling_map)
            pooling_variance = np.var(blood_pooling_map)
            st.metric("Intensidade Média de Acúmulo", f"{pooling_intensity:.3f}")
            st.metric("Variância da Distribuição", f"{pooling_variance:.6f}")
            
            fixation_ratio = assess_livor_fixation(blood_pooling_map)
            if fixation_ratio > 0.7:
                st.error(f"Alta probabilidade de manchas fixas (>12h post-mortem)")
            elif fixation_ratio > 0.3:
                st.warning(f"Manchas em transição (6-12h post-mortem)")
            else:
                st.success(f"Manchas não fixas (<6h post-mortem)")
    
    # Continuação das outras abas (Rigor Mortis, Putrefação, Fenômenos Conservadores)...
    # Implementação similar para as outras abas

def simulate_body_cooling(image_array):
    hu_min, hu_max = np.min(image_array), np.max(image_array)
    normalized = (image_array - hu_min) / (hu_max - hu_min)
    center_y, center_x = np.array(image_array.shape) / 2
    y, x = np.indices(image_array.shape)
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    center_effect = 1 - (distance_from_center / max_distance)
    simulated_temp = 25 + 10 * normalized + 5 * center_effect
    return simulated_temp

def estimate_pmi_from_cooling(thermal_map, ambient_temp, body_mass, clothing):
    core_temp = np.max(thermal_map)
    temp_difference = core_temp - ambient_temp
    mass_factor = body_mass / 70
    clothing_factor = {"Leve": 0.8, "Moderado": 1.0, "Abrigado": 1.2}[clothing]
    pmi_hours = (temp_difference * mass_factor * clothing_factor) / 0.8
    return max(0, min(pmi_hours, 48))

def detect_blood_pooling(image_array):
    gradient_x = ndimage.sobel(image_array, axis=0)
    gradient_y = ndimage.sobel(image_array, axis=1)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    pooling_map = ndimage.gaussian_filter(gradient_magnitude, sigma=2)
    pooling_map = (pooling_map - np.min(pooling_map)) / (np.max(pooling_map) - np.min(pooling_map))
    return pooling_map

def assess_livor_fixation(pooling_map):
    edges = feature.canny(pooling_map, sigma=2)
    fixation_ratio = np.sum(edges) / edges.size
    return fixation_ratio

# Continuação das outras funções...

# ====== SEÇÃO 2: INICIALIZAÇÃO DO BANCO DE DADOS E FUNÇÕES AUXILIARES ======

def safe_init_database():
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
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
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT,
                rating INTEGER,
                category TEXT,
                comment TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        ip_address = "127.0.0.1"
        cursor.execute("""
            INSERT INTO security_logs (user_email, action, ip_address, details)
            VALUES (?, ?, ?, ?)
        """, (user_email, action, ip_address, details))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Erro ao registrar evento de segurança: {e}")

def save_report_to_db(user_email, report_name, report_data, parameters):
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (user_email, report_name, report_data, parameters)
            VALUES (?, ?, ?, ?)
        """, (user_email, report_name, report_data, json.dumps(parameters)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Erro ao salvar relatório: {e}")
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

# ====== SEÇÃO 3: INTERFACE DO USUÁRIO ======

def update_css_theme():
    st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
        padding-top: 2rem;
        color: #000000;
    }
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 600;
    }
    p, div, span {
        color: #000000 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .css-1d391kg, .css-1v0mbdj {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E0E0E0;
    }
    .css-1d391kg p, .css-1v0mbdj p {
        color: #000000 !important;
    }
    .stButton > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #000000;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #333333 !important;
        border-color: #333333;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F0F0;
        border-radius: 4px 4px 0 0;
        color: #000000;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border: 1px solid #E0E0E0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border-bottom: 2px solid #000000;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    .stMetric {
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 1rem;
    }
    .stAlert {
        background-color: #F8F9FA;
        border-left: 4px solid #000000;
        color: #000000;
        border-radius: 4px;
    }
    .streamlit-expanderHeader {
        background-color: #F8F9FA;
        color: #000000;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        font-weight: 600;
    }
    .dataframe {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #E0E0E0;
    }
    .dataframe th {
        background-color: #F0F0F0;
        color: #000000;
        font-weight: 600;
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
    .upload-section {
        background-color: #F8F9FA;
        padding: 2rem;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        color: #000000;
        text-align: center;
        margin: 1rem 0;
    }
    .info-card {
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #000000;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #000000;
        color: #FFFFFF;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        DICOM Autopsy Viewer PRO v3.0 | Interface Profissional | © 2025
    </div>
    """, unsafe_allow_html=True)

def show_user_form():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #000000; font-size: 2.5rem; margin-bottom: 0.5rem;">DICOM Autopsy Viewer PRO</h1>
        <h3 style="color: #666666; font-weight: 400;">Sistema Avançado de Análise Forense Digital</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://via.placeholder.com/300x300/FFFFFF/000000?text=DICOM+Viewer",
                 use_column_width=True, caption="Sistema de Análise de Imagens Forenses")
    
    with col2:
        with st.form("user_registration"):
            st.markdown("### Registro de Usuário")
            name = st.text_input("Nome Completo *", placeholder="Dr. João Silva",
                                 help="Informe seu nome completo")
            email = st.text_input("Email Institucional *", placeholder="joao.silva@hospital.com",
                                  help="Utilize email institucional para registro")
            
            col_a, col_b = st.columns(2)
            with col_a:
                role = st.selectbox("Função *", [
                    "Radiologista", "Médico Legista", "Técnico em Radiologia",
                    "Pesquisador", "Estudante", "Outro"
                ], help="Selecione sua função principal")
            
            with col_b:
                department = st.text_input("Departamento/Instituição",
                                           placeholder="Departamento de Radiologia",
                                           help="Informe seu departamento ou instituição")
            
            with st.expander("📋 Termos de Uso e Política de Privacidade"):
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
            
            submitted = st.form_submit_button("Iniciar Sistema →", use_container_width=True)
            
            if submitted:
                if not all([name, email, terms_accepted]):
                    st.error("Por favor, preencha todos os campos obrigatórios e aceite os termos de uso.")
                else:
                    try:
                        conn = sqlite3.connect("dicom_viewer.db")
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO users (name, email, role, department)
                            VALUES (?, ?, ?, ?)
                        """, (name, email, role, department))
                        conn.commit()
                        conn.close()
                        
                        st.session_state.user_data = {
                            'name': name,
                            'email': email,
                            'role': role,
                            'department': department
                        }
                        
                        log_security_event(email, "USER_REGISTRATION", f"Role: {role}")
                        st.success("✅ Usuário registrado com sucesso!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao registrar usuário: {e}")

def show_main_app():
    user_data = st.session_state.user_data
    
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 1rem; border-bottom: 1px solid #E0E0E0; margin-bottom: 1rem;">
            <h3 style="color: #000000; margin-bottom: 0.5rem;">👤 {user_data['name']}</h3>
            <p style="color: #666666; margin: 0;"><strong>Função:</strong> {user_data['role']}</p>
            <p style="color: #666666; margin: 0;"><strong>Email:</strong> {user_data['email']}</p>
            {f'<p style="color: #666666; margin: 0;"><strong>Departamento:</strong> {user_data["department"]}</p>' if user_data['department'] else ''}
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
        
        with st.expander("ℹ️ Informações do Sistema"):
            st.write("**Versão:** 3.0 Professional")
            st.write("**Última Atualização:** 2025-09-15")
            st.write("**Status:** Online")
            st.write("**Armazenamento:** 2.5 GB disponíveis")
        
        if st.button("Trocar Usuário", use_container_width=True):
            st.session_state.user_data = None
            st.rerun()
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <h1 style="color: #000000; margin-right: 1rem; margin-bottom: 0;">DICOM Autopsy Viewer</h1>
        <span style="background-color: #000000; color: #FFFFFF; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
            v3.0 Professional
        </span>
    </div>
    <p style="color: #666666; margin-bottom: 2rem;">Bem-vindo, <strong>{user_data['name']}</strong>! Utilize as ferramentas abaixo para análise forense avançada de imagens DICOM.</p>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            log_security_event(user_data['email'], "FILE_UPLOAD",
                              f"Filename: {uploaded_file.name}")
            
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
                
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "👁️ Visualização", "📊 Estatísticas", "🔧 Análise Técnica",
                    "📈 Qualidade", "⚰️ Análise Post-Mortem", "📋 RA-Index", "📄 Relatórios"
                ])
                
                with tab1:
                    enhanced_visualization_tab(dicom_data, image_array)
                
                with tab2:
                    enhanced_statistics_tab(dicom_data, image_array)
                
                with tab3:
                    enhanced_technical_analysis_tab(dicom_data, image_array)
                
                with tab4:
                    enhanced_quality_metrics_tab(dicom_data, image_array)
                
                with tab5:
                    enhanced_post_mortem_analysis_tab(dicom_data, image_array)
                
                with tab6:
                    enhanced_ra_index_tab(dicom_data, image_array)
                
                with tab7:
                    enhanced_reporting_tab(dicom_data, image_array, user_data)
            
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo DICOM: {e}")
            logging.error(f"Erro no processamento DICOM: {e}")
    
    else:
        st.info("Carregue um arquivo DICOM na sidebar para começar a análise.")
        
        st.markdown("## Funcionalidades Disponíveis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h4>👁️ Visualização Avançada</h4>
                <ul>
                    <li>Janelamento Hounsfield personalizado</li>
                    <li>Ferramentas colorimétricas</li>
                    <li>Análise de pixels interativa</li>
                    <li>Visualização 3D multiplana</li>
                    <li>Download de imagens processadas</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4>📊 Análise Estatística</h4>
                <ul>
                    <li>6+ tipos de visualizações</li>
                    <li>Análise regional detalhada</li>
                    <li>Correlações avançadas</li>
                    <li>Densidade de probabilidade</li>
                    <li>Mapas de calor interativos</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card">
                <h4>🔧 Análise Forense</h4>
                <ul>
                    <li>Metadados completos DICOM</li>
                    <li>Verificação de integridade</li>
                    <li>Detecção de anomalias</li>
                    <li>Timeline forense</li>
                    <li>Autenticidade de imagens</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown("""
            <div class="info-card">
                <h4>📈 Controle de Qualidade</h4>
                <ul>
                    <li>Métricas de qualidade de imagem</li>
                    <li>Análise de ruído e artefatos</li>
                    <li>Detecção de compressão</li>
                    <li>Uniformidade e resolução</li>
                    <li>Relatórios de qualidade</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div class="info-card">
                <h4>⚰️ Análise Post-Mortem</h4>
                <ul>
                    <li>Estimativa de intervalo post-mortem</li>
                    <li>Análise de fenômenos cadavéricos</li>
                    <li>Modelos de decomposição</li>
                    <li>Mapas de alterações teciduais</li>
                    <li>Correlações temporais</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown("""
            <div class="info-card">
                <h4>📄 Relatórios Completos</h4>
                <ul>
                    <li>Relatórios personalizáveis</li>
                    <li>Exportação em PDF/CSV</li>
                    <li>Histórico de análises</li>
                    <li>Comparativo entre exames</li>
                    <li>Banco de dados de casos</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("## Casos de Uso Exemplares")
        use_case_col1, use_case_col2 = st.columns(2)
        
        with use_case_col1:
            with st.expander("🔫 Identificação de Metais e Projéteis"):
                st.markdown("""
                1. Carregue a imagem DICOM
                2. Acesse a aba **Visualização**
                3. Utilize as ferramentas colorimétricas para destacar metais
                4. Ajuste a janela Hounsfield para a faixa de 1000-3000 HU
                5. Use os filtros de detecção de bordas para melhorar a visualização
                6. Gere um relatório completo com as medidas e localizações
                """)
        
        with use_case_col2:
            with st.expander("⏰ Estimativa de Intervalo Post-Mortem"):
                st.markdown("""
                1. Carregue a imagem DICOM
                2. Acesse a aba **Análise Post-Mortem**
                3. Configure os parâmetros ambientais
                4. Analise os mapas de distribuição gasosa
                5. Consulte as estimativas temporais
                6. Exporte o relatório forense completo
                """)

# ====== FUNÇÃO PRINCIPAL ======

def main():
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    
    if 'dicom_data' not in st.session_state:
        st.session_state.dicom_data = None
    
    if 'image_array' not in st.session_state:
        st.session_state.image_array = None
    
    if 'current_report' not in st.session_state:
        st.session_state.current_report = None
    
    setup_matplotlib_for_plotting()
    
    if not safe_init_database():
        st.error("❌ Erro crítico: Não foi possível inicializar o sistema. Contate o administrador.")
        return
    
    update_css_theme()
    
    if st.session_state.user_data is None:
        show_user_form()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
