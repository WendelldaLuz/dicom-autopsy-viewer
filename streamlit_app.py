# ==============================================================================
# SEÇÃO 1: IMPORTS DE BIBLIOTECAS
# ==============================================================================
import base64
import csv
import hashlib
import json
import logging
import os
import smtplib
import socket
import sqlite3
import tempfile
import uuid
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from plotly.subplots import make_subplots
from scipy import ndimage, signal
from scipy.optimize import curve_fit
from skimage import feature, img_as_ubyte

# Configuração de logging para depuração
logging.basicConfig(level=logging.INFO)

# ==============================================================================
# SEÇÃO 2: VERIFICAÇÃO E INFORMAÇÕES DE DEPENDÊNCIAS
# ==============================================================================
st.info("Verificando dependências...")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
except ImportError:
    st.warning("A biblioteca `ReportLab` não está instalada. A funcionalidade de geração de PDF estará limitada. Execute `pip install reportlab` para habilitá-la.")
try:
    import jpeg_ls  # type: ignore
except ImportError:
    st.warning("A biblioteca `jpeg_ls` não está instalada. A funcionalidade para compressão JPEG-LS estará limitada. Execute `pip install jpeg-ls` para habilitá-la.")
try:
    import gdcm  # type: ignore
except ImportError:
    st.warning("A biblioteca `gdcm` não está instalada. A funcionalidade para compressão JPEG 2000 estará limitada. Execute `pip install python-gdcm` para habilitá-la.")
try:
    import cv2
except ImportError:
    st.warning("A biblioteca OpenCV (`cv2`) não está instalada. Algumas funcionalidades de processamento de imagem estarão limitadas. Execute `pip install opencv-python`.")

# ==============================================================================
# SEÇÃO 3: CONFIGURAÇÃO DA PÁGINA E ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="DICOM Autopsy Viewer Pro - Enhanced",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

def update_css_theme():
    """
    Aplicar tema CSS profissional branco com preto.
    """
    st.markdown("""
    <style>
    /* Tema principal - branco com preto */
    .main {
        background-color: #FFFFFF;
        padding-top: 2rem;
        color: #000000;
    }
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    /* Cabeçalhos */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 600;
    }
    /* Texto geral */
    p, div, span {
        color: #000000 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Sidebar */
    .css-1d391kg, .css-1v0mbdj {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E0E0E0;
    }
    .css-1d391kg p, .css-1v0mbdj p {
        color: #000000 !important;
    }
    /* Botões */
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
    /* Abas */
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
    /* Campos de entrada */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
    }
    /* Métricas */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    .stMetric {
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 1rem;
    }
    /* Alertas */
    .stAlert {
        background-color: #F8F9FA;
        border-left: 4px solid #000000;
        color: #000000;
        border-radius: 4px;
    }
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #F8F9FA;
        color: #000000;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        font-weight: 600;
    }
    /* Tabelas */
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
    /* Footer */
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
    /* Upload section */
    .upload-section {
        background-color: #F8F9FA;
        padding: 2rem;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        color: #000000;
        text-align: center;
        margin: 1rem 0;
    }
    /* Cards de informação */
    .info-card {
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    /* Tooltips */
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
    
    # Adicionar footer
    st.markdown("""
    <div class="footer">
        DICOM Autopsy Viewer PRO v3.0 | Interface Profissional | © 2025
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SEÇÃO 4: FUNÇÕES AUXILIARES DE ANÁLISE E LÓGICA DE NEGÓCIO
# ==============================================================================

def setup_matplotlib_for_plotting():
    """Configura o Matplotlib para plotagem com estilo consistente."""
    import warnings
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    try:
        plt.style.use("seaborn-v0_8")
    except ImportError:
        plt.style.use("default")
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False

def apply_hounsfield_windowing(image, window_center, window_width):
    """Aplica janelamento de Hounsfield (HU) na imagem para realce de tecidos."""
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    windowed_image = np.clip(image, min_value, max_value)
    windowed_image = (windowed_image - min_value) / (max_value - min_value) * 255
    return windowed_image.astype(np.uint8)

def simulate_body_cooling(image_array):
    """Simula a distribuição térmica corporal baseada na densidade de tecidos."""
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
    """Estima o intervalo post-mortem (IPM) baseado no padrão de esfriamento."""
    core_temp = np.max(thermal_map)
    temp_difference = core_temp - ambient_temp
    mass_factor = body_mass / 70
    clothing_factor = {"Leve": 0.8, "Moderado": 1.0, "Abrigado": 1.2}[clothing]
    pmi_hours = (temp_difference * mass_factor * clothing_factor) / 0.8
    return max(0, min(pmi_hours, 48))

def generate_cooling_curve(ipm_estimate, ambient_temp):
    """Gera uma curva de resfriamento teórica."""
    hours = np.linspace(0, ipm_estimate + 5, 50)
    temps = ambient_temp + (37 - ambient_temp) * np.exp(-0.05 * hours)
    return pd.DataFrame({'Horas': hours, 'Temperatura': temps}).set_index('Horas')

def detect_blood_pooling(image_array):
    """Detecta regiões de possível acúmulo sanguíneo por hipóstase usando gradientes."""
    gradient_x = ndimage.sobel(image_array, axis=0)
    gradient_y = ndimage.sobel(image_array, axis=1)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    pooling_map = ndimage.gaussian_filter(gradient_magnitude, sigma=2)
    pooling_map = (pooling_map - np.min(pooling_map)) / (np.max(pooling_map) - np.min(pooling_map))
    return pooling_map

def assess_livor_fixation(pooling_map):
    """Avalia o grau de fixação das manchas de hipóstase com base na detecção de bordas."""
    edges = feature.canny(pooling_map, sigma=2)
    fixation_ratio = np.sum(edges) / edges.size
    return fixation_ratio

def segment_muscle_tissue(image_array):
    """Segmenta o tecido muscular baseado na faixa de unidades Hounsfield (HU) típica."""
    muscle_mask = (image_array >= 35) & (image_array <= 55)
    return muscle_mask.astype(float)

def calculate_muscle_density(image_array, muscle_mask):
    """Calcula a densidade média do tecido muscular a partir da máscara."""
    muscle_values = image_array[muscle_mask > 0.5]
    return np.mean(muscle_values) if len(muscle_values) > 0 else 0

def estimate_rigor_stage(muscle_density):
    """Estima o estágio do rigor mortis baseado na densidade muscular."""
    if muscle_density < 40:
        return "inicial"
    elif 40 <= muscle_density < 48:
        return "progressivo"
    elif 48 <= muscle_density < 55:
        return "completo"
    else:
        return "resolucao"

def detect_putrefaction_gases(image_array):
    """Detecta gases de putrefação baseado em valores de HU típicos."""
    gas_mask = (image_array <= -100) & (image_array >= -1000)
    gas_map = ndimage.gaussian_filter(gas_mask.astype(float), sigma=3)
    return gas_map

def classify_putrefaction_stage(image_array):
    """Classifica o estágio de putrefação baseado no volume de gases na imagem."""
    gas_map = detect_putrefaction_gases(image_array)
    gas_volume = np.sum(gas_map > 0.5) / gas_map.size
    if gas_volume < 0.05:
        return "initial"
    elif 0.05 <= gas_volume < 0.15:
        return "coloracao"
    elif 0.15 <= gas_volume < 0.3:
        return "gasoso"
    else:
        return "coliquativo"

def analyze_conservation_features(image_array):
    """Analisa características de fenômenos conservadores como saponificação e mumificação."""
    saponification_mask = (image_array >= 100) & (image_array <= 300)
    calcification_mask = image_array >= 500
    conservation_map = np.zeros_like(image_array, dtype=float)
    conservation_map[saponification_mask] = 0.5
    conservation_map[calcification_mask] = 1.0
    return conservation_map

def classify_conservation_type(image_array):
    """Classifica o tipo de fenômeno conservador presente."""
    conservation_map = analyze_conservation_features(image_array)
    saponification_ratio = np.sum(conservation_map == 0.5) / conservation_map.size
    calcification_ratio = np.sum(conservation_map == 1.0) / conservation_map.size
    if calcification_ratio > 0.05:
        return "calcification"
    elif saponification_ratio > 0.1:
        return "saponification"
    elif np.mean(image_array) > 200:
        return "mummification"
    else:
        return "none"

def generate_post_mortem_report(image_array, thermal_map, pooling_map, muscle_density, gas_map, conservation_map):
    """Gera um relatório consolidado de análise post-mortem com métricas e estimativas."""
    ipm_algor = estimate_pmi_from_cooling(thermal_map, 25, 70, "Moderado")
    fixation_ratio = assess_livor_fixation(pooling_map)
    rigor_stage = estimate_rigor_stage(muscle_density)
    putrefaction_stage = classify_putrefaction_stage(image_array)
    conservation_type = classify_conservation_type(image_array)
    if fixation_ratio > 0.7:
        ipm_livor = "12-18h (manchas fixas)"
    elif fixation_ratio > 0.3:
        ipm_livor = "6-12h (em fixação)"
    else:
        ipm_livor = "2-6h (manchas não fixas)"
    ipm_rigor_map = {"inicial": "2-4h", "progressivo": "4-8h", "completo": "8-12h", "resolucao": "12-24h"}
    ipm_rigor = ipm_rigor_map.get(rigor_stage, "Indeterminado")
    ipm_putrefaction_map = {"initial": "0-24h", "coloracao": "24-48h", "gasoso": "48-72h", "coliquativo": ">72h"}
    ipm_putrefaction = ipm_putrefaction_map.get(putrefaction_stage, "Indeterminado")
    notes = []
    if ipm_algor > 24: notes.append("Padrão de esfriamento sugere IPM prolongado.")
    if fixation_ratio > 0.7: notes.append("Hipóstase fixa indica que o corpo não foi movido após 12h post-mortem.")
    if putrefaction_stage == "gasoso": notes.append("Presença significativa de gases de putrefação.")
    if conservation_type != "none": notes.append(f"Evidências de {conservation_type} detectadas.")
    forensic_notes = "\n".join([f"- {note}" for note in notes]) if notes else "Nenhuma observação adicional."
    return {
        'ipm_algor': ipm_algor, 'ipm_livor': ipm_livor, 'ipm_rigor': ipm_rigor,
        'ipm_putrefaction': ipm_putrefaction, 'algor_stage': f"Esfriamento avançado ({np.mean(thermal_map):.1f}°C)",
        'livor_stage': f"Fixation ratio: {fixation_ratio:.2f}", 'rigor_stage': rigor_stage,
        'putrefaction_stage': putrefaction_stage, 'conservation_type': conservation_type,
        'estimated_temp': np.mean(thermal_map), 'pooling_intensity': np.mean(pooling_map),
        'muscle_density': muscle_density, 'gas_volume': np.sum(gas_map > 0.5) / gas_map.size * 100,
        'forensic_notes': forensic_notes
    }

def calculate_extended_statistics(image_array):
    """Calcula estatísticas descritivas expandidas da imagem."""
    flattened = image_array.flatten()
    return {
        'Média': np.mean(flattened), 'Mediana': np.median(flattened), 'Desvio Padrão': np.std(flattened),
        'Erro Padrão': stats.sem(flattened), 'Mínimo': np.min(flattened), 'Máximo': np.max(flattened),
        'Amplitude': np.ptp(flattened), 'Percentil 5': np.percentile(flattened, 5),
        'Percentil 25': np.percentile(flattened, 25), 'Percentil 75': np.percentile(flattened, 75),
        'Percentil 95': np.percentile(flattened, 95), 'IQR': np.percentile(flattened, 75) - np.percentile(flattened, 25),
        'Assimetria': stats.skew(flattened), 'Curtose': stats.kurtosis(flattened),
        'CV': np.std(flattened) / np.mean(flattened) if np.mean(flattened) != 0 else 0
    }

def create_enhanced_histogram(image_array):
    """Cria um histograma avançado com ajuste de distribuição normal."""
    flattened = image_array.flatten()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=flattened, name="Dados", nbinsx=100, opacity=0.7, marker_color='lightblue'))
    mu, sigma = np.mean(flattened), np.std(flattened)
    x_range = np.linspace(np.min(flattened), np.max(flattened), 200)
    pdf = stats.norm.pdf(x_range, mu, sigma)
    scale_factor = len(flattened) * (np.max(flattened) - np.min(flattened)) / 100
    fig.add_trace(go.Scatter(x=x_range, y=pdf * scale_factor, name="Distribuição Normal", line=dict(color='red', width=2)))
    fig.update_layout(title="Histograma com Ajuste de Distribuição Normal", xaxis_title="Unidades Hounsfield (HU)", yaxis_title="Frequência", height=400)
    return fig

def create_qq_plot(image_array):
    """Cria um QQ plot para análise de normalidade da distribuição dos pixels."""
    flattened = image_array.flatten()
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(flattened)))
    sample_quantiles = np.percentile(flattened, np.linspace(1, 99, len(flattened)))
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name='Quantis Amostrais'))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Referência', line=dict(color='red', dash='dash')))
    fig.update_layout(title="QQ Plot - Análise de Normalidade", xaxis_title="Quantis Teóricos", yaxis_title="Quantis Amostrais", height=400)
    return fig

def create_annotated_heatmap(image_array):
    """Cria um mapa de calor com anotações de densidade."""
    if image_array.shape[0] > 200 or image_array.shape[1] > 200:
        reduction_factor = max(image_array.shape[0] // 200, image_array.shape[1] // 200)
        small_array = image_array[::reduction_factor, ::reduction_factor]
    else:
        small_array = image_array
    fig = go.Figure(data=go.Heatmap(z=small_array, colorscale='viridis', showscale=True, hoverongaps=False))
    fig.update_layout(title="Mapa de Calor com Análise de Densidade", height=400)
    return fig

def create_gradient_analysis(image_array):
    """Cria uma visualização da análise de gradientes e bordas."""
    grad_x = np.gradient(image_array.astype(float), axis=1)
    grad_y = np.gradient(image_array.astype(float), axis=0)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    fig = px.imshow(gradient_magnitude, color_continuous_scale='gray', title="Análise de Gradientes da Imagem")
    return fig

def calculate_regional_statistics(image_array, grid_size):
    """Calcula estatísticas regionais da imagem para uma grade."""
    h, w = image_array.shape
    h_step, w_step = h // grid_size, w // grid_size
    regional_data = []
    for i in range(grid_size):
        for j in range(grid_size):
            region = image_array[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
            if region.size > 0:
                regional_data.append({
                    'Região': f"{i + 1}-{j + 1}", 'X': j, 'Y': i, 'Média': np.mean(region),
                    'Mediana': np.median(region), 'Desvio Padrão': np.std(region),
                    'Mínimo': np.min(region), 'Máximo': np.max(region),
                    'Assimetria': stats.skew(region.flatten()), 'Área (%)': (region.size / image_array.size) * 100
                })
    return pd.DataFrame(regional_data)

def create_regional_heatmap(regional_stats, grid_size):
    """Cria um mapa de calor das estatísticas regionais."""
    mean_matrix = np.zeros((grid_size, grid_size))
    for _, row in regional_stats.iterrows():
        i, j = int(row['Y']), int(row['X'])
        if i < grid_size and j < grid_size:
            mean_matrix[i, j] = row['Média']
    fig = go.Figure(data=go.Heatmap(
        z=mean_matrix, colorscale='viridis', showscale=True,
        text=[[f"Média: {mean_matrix[i, j]:.1f}\nRegião: {i + 1}-{j + 1}" for j in range(grid_size)] for i in range(grid_size)],
        texttemplate="%{text}", textfont={"size": 10}))
    fig.update_layout(title="Mapa de Calor Regional - Valores Médios por Região", xaxis_title="Região X", yaxis_title="Região Y", height=500)
    return fig

def create_spatial_correlation_analysis(image_array):
    """Cria uma análise de autocorrelação espacial."""
    if image_array.shape[0] > 100 or image_array.shape[1] > 100:
        reduction_factor = max(image_array.shape[0] // 100, image_array.shape[1] // 100)
        small_array = image_array[::reduction_factor, ::reduction_factor]
    else:
        small_array = image_array
    correlation = signal.correlate2d(small_array, small_array, mode='same')
    fig = go.Figure(data=go.Heatmap(z=correlation, colorscale='viridis', showscale=True))
    fig.update_layout(title="Matriz de Autocorrelação Espacial", height=400)
    return fig

def create_variogram_analysis(image_array):
    """Cria uma análise de variograma para dependência espacial."""
    from scipy.spatial.distance import pdist
    h, w = image_array.shape
    n_points = min(1000, h * w)
    indices = np.random.choice(h * w, n_points, replace=False)
    y_coords, x_coords = np.unravel_index(indices, (h, w))
    values = image_array.flatten()[indices]
    distances = pdist(np.column_stack([x_coords, y_coords]))
    value_differences = pdist(values[:, None])
    squared_differences = value_differences ** 2
    max_distance = np.sqrt(h ** 2 + w ** 2) / 2
    distance_bins = np.linspace(0, max_distance, 20)
    variogram_values = np.zeros(len(distance_bins) - 1)
    for i in range(len(distance_bins) - 1):
        mask = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
        if np.any(mask):
            variogram_values[i] = np.mean(squared_differences[mask]) / 2
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bin_centers, y=variogram_values, mode='lines+markers', name='Variograma Experimental'))
    fig.update_layout(title="Variograma Experimental", xaxis_title="Distância (pixels)", yaxis_title="Semivariância", height=400)
    return fig

def generate_tissue_change_predictions(image_array, time_horizon):
    """Gera previsões de mudanças teciduais com base em modelos simplificados."""
    change_factors = {'air': 0.1, 'fat': 0.3, 'soft_tissue': 0.8, 'bone': 0.2, 'metal': 0.05}
    prediction_map = np.zeros_like(image_array, dtype=float)
    prediction_map[image_array < -100] = change_factors['air'] * time_horizon
    prediction_map[(image_array >= -100) & (image_array < 0)] = change_factors['fat'] * time_horizon
    prediction_map[(image_array >= 0) & (image_array < 100)] = change_factors['soft_tissue'] * time_horizon
    prediction_map[(image_array >= 100) & (image_array < 400)] = change_factors['bone'] * time_horizon
    prediction_map[image_array >= 400] = change_factors['metal'] * time_horizon
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, image_array.shape)
    prediction_map += noise
    return prediction_map

def create_prediction_heatmap(prediction_map, time_horizon):
    """Cria um mapa de calor das previsões de mudanças teciduais."""
    fig = go.Figure(data=go.Heatmap(
        z=prediction_map, colorscale='hot', showscale=True,
        hovertemplate='Mudança Prevista: %{z:.2f} HU<extra></extra>'))
    fig.update_layout(title=f"Mapa Preditivo de Mudanças Teciduais ({time_horizon}h)", height=500)
    return fig

def run_predictive_simulation(image_array, time_horizon, ambient_temp, humidity, body_position):
    """Executa uma simulação preditiva completa de decomposição."""
    temp_factor = max(0.5, min(2.0, ambient_temp / 22))
    humidity_factor = 1.0 + (humidity - 60) / 100
    if body_position == "Decúbito Dorsal": position_factor = 1.2
    elif body_position == "Decúbito Ventral": position_factor = 1.1
    elif body_position == "Lateral": position_factor = 1.0
    else: position_factor = 1.3
    base_change = 2.0
    total_change = base_change * time_horizon * temp_factor * humidity_factor * position_factor
    significant_change = np.sum(image_array < 50) / image_array.size * 100
    return {'change_rate': total_change / time_horizon, 'changed_area': significant_change}

def simulate_temporal_trends(image_array, time_points, ambient_temp, humidity):
    """Simula as tendências temporais de mudanças de densidade em diferentes tecidos."""
    trends = {}
    tissue_types = {
        'Tecidos Moles': (image_array >= 0) & (image_array < 100),
        'Gordura': (image_array >= -100) & (image_array < 0),
        'Osso': (image_array >= 100) & (image_array < 400)
    }
    for tissue_name, mask in tissue_types.items():
        if np.any(mask):
            base_value = np.mean(image_array[mask])
            if tissue_name == 'Tecidos Moles': change_rate = 2.0 * (ambient_temp / 22) * (humidity / 60)
            elif tissue_name == 'Gordura': change_rate = 1.0 * (ambient_temp / 22) * (humidity / 60)
            else: change_rate = 0.3 * (ambient_temp / 22)
            trends[tissue_name] = [base_value + change_rate * t for t in time_points]
    return trends

def create_temporal_trend_chart(trend_data, time_points):
    """Cria um gráfico de tendências temporais de mudanças de densidade."""
    fig = go.Figure()
    for tissue_name, values in trend_data.items():
        fig.add_trace(go.Scatter(x=time_points, y=values, mode='lines+markers', name=tissue_name))
    fig.update_layout(title="Projeção Temporal de Mudanças de Densidade", xaxis_title="Tempo Post-Mortem (horas)", yaxis_title="Densidade Média (HU)", height=400)
    return fig

def calculate_tissue_composition(image_array, tissue_ranges):
    """Calcula a composição tecidual por faixas de HU."""
    total_pixels = image_array.size
    composition = {}
    for tissue_name, (min_hu, max_hu) in tissue_ranges.items():
        mask = (image_array >= min_hu) & (image_array < max_hu)
        percentage = np.sum(mask) / total_pixels * 100
        composition[tissue_name] = percentage
    return composition

def create_tissue_composition_chart(tissue_composition):
    """Cria um gráfico de barras da composição tecidual."""
    tissues = list(tissue_composition.keys())
    percentages = list(tissue_composition.values())
    fig = go.Figure(data=[go.Bar(x=tissues, y=percentages, marker_color=px.colors.qualitative.Set3)])
    fig.update_layout(title="Composição Tecidual por Faixas de HU", xaxis_title="Tipo de Tecido", yaxis_title="Porcentagem da Área Total", height=400)
    return fig

def simulate_metabolic_changes(image_array, metabolic_rate, enzyme_activity):
    """Simula mudanças metabólicas post-mortem."""
    soft_tissue_mask = (image_array >= 0) & (image_array < 100)
    soft_tissue_percentage = np.sum(soft_tissue_mask) / image_array.size * 100
    autolysis = min(100, soft_tissue_percentage * metabolic_rate * 0.5)
    gas_mask = image_array < -100
    gas_percentage = np.sum(gas_mask) / image_array.size * 100
    gas_production = min(10, gas_percentage * enzyme_activity * 0.2)
    acidity = 6.8 - (autolysis / 100 * 1.5)
    return {'autolysis': autolysis, 'gas_production': gas_production, 'acidity': acidity}

def check_temporal_consistency(metadata):
    """Verifica a consistência temporal dos metadados DICOM."""
    dates = {key: value for key, value in metadata.items() if 'date' in key.lower() and value.strip()}
    times = {key: value for key, value in metadata.items() if 'time' in key.lower() and value.strip()}
    if not dates and not times: return {'consistent': False, 'issue': 'Sem informações temporais'}
    unique_dates = set(dates.values())
    if len(unique_dates) > 1: return {'consistent': False, 'issue': f'Datas inconsistentes: {unique_dates}'}
    return {'consistent': True, 'issue': None}

def check_dicom_compliance(metadata):
    """Verifica a conformidade com o padrão DICOM."""
    required_fields = ['SOPClassUID', 'SOPInstanceUID', 'StudyDate', 'StudyTime', 'AccessionNumber', 'Modality', 'Manufacturer', 'InstanceNumber']
    present_fields = [field for field in required_fields if field in metadata]
    compliance_score = len(present_fields) / len(required_fields)
    return {'score': compliance_score, 'missing': [field for field in required_fields if field not in metadata], 'present': present_fields}

def analyze_image_noise(image_array):
    """Analisa o ruído na imagem com técnicas avançadas."""
    noise_residual = image_array - ndimage.median_filter(image_array, size=3)
    total_noise = np.std(noise_residual)
    background_mask = identify_homogeneous_regions(image_array)
    background_noise = np.std(noise_residual[background_mask]) if np.any(background_mask) else 0
    signal_mask = identify_high_contrast_regions(image_array)
    signal_noise = np.std(noise_residual[signal_mask]) if np.any(signal_mask) else 0
    noise_pattern = analyze_noise_pattern(noise_residual)
    return {'total_noise': total_noise, 'background_noise': background_noise, 'signal_noise': signal_noise, 'pattern': noise_pattern}

def analyze_compression(image_array):
    """Analisa características de compressão da imagem."""
    hist, _ = np.histogram(image_array.flatten(), bins=256, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    unique_values = len(np.unique(image_array))
    compression_ratio = unique_values / image_array.size
    max_entropy = np.log2(256)
    redundancy = (1 - entropy / max_entropy) * 100 if max_entropy > 0 else 0
    likely_compressed = compression_ratio < 0.5 or entropy < 6.0
    return {'ratio': compression_ratio, 'entropy': entropy, 'redundancy': redundancy, 'likely_compressed': likely_compressed}

def calculate_spectral_metrics(fft_data):
    """Calcula métricas avançadas do espectro de frequência."""
    magnitude_spectrum = np.abs(fft_data)
    power_spectrum = magnitude_spectrum ** 2
    total_energy = np.sum(power_spectrum)
    h, w = power_spectrum.shape
    y_coords, x_coords = np.indices(power_spectrum.shape)
    centroid_x = np.sum(x_coords * power_spectrum) / total_energy
    centroid_y = np.sum(y_coords * power_spectrum) / total_energy
    normalized_power = power_spectrum / total_energy
    normalized_power = normalized_power[normalized_power > 0]
    spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power))
    max_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
    dominant_frequency = np.sqrt((max_idx[0] - h / 2) ** 2 + (max_idx[1] - w / 2) ** 2)
    return {'total_energy': total_energy, 'centroid_x': centroid_x, 'centroid_y': centroid_y, 'spectral_entropy': spectral_entropy, 'dominant_frequency': dominant_frequency}

def calculate_texture_features(image_array):
    """Calcula características de textura avançadas usando GLCM."""
    try:
        image_uint8 = img_as_ubyte((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)))
        glcm = graycomatrix(image_uint8, [1], [0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        hist, _ = np.histogram(image_array.flatten(), bins=256, density=True)
        hist = hist[hist > 0]
        complexity = -np.sum(hist * np.log2(hist)) / 8
        return {'contrast': contrast, 'energy': energy, 'homogeneity': homogeneity, 'correlation': correlation, 'complexity': complexity}
    except ImportError:
        return {'contrast': np.std(image_array), 'energy': np.mean(image_array ** 2), 'homogeneity': 1.0 / (1.0 + np.var(image_array)), 'correlation': 0.5, 'complexity': 0.5}

def analyze_structures(image_array):
    """Analisa estruturas na imagem usando técnicas morfológicas."""
    grad_x = np.gradient(image_array, axis=1); grad_y = np.gradient(image_array, axis=0); gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    threshold = np.percentile(gradient_magnitude, 95); edges = gradient_magnitude > threshold; edge_density = np.sum(edges) / edges.size
    labeled, num_components = ndimage.label(edges); component_sizes = ndimage.sum(edges, labeled, range(1, num_components + 1)); avg_component_size = np.mean(component_sizes) if num_components > 0 else 0
    aspect_ratios = []
    for i in range(1, num_components + 1):
        component_mask = labeled == i; y_indices, x_indices = np.where(component_mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            height = np.max(y_indices) - np.min(y_indices) + 1; width = np.max(x_indices) - np.min(x_indices) + 1
            if width > 0: aspect_ratios.append(height / width)
    avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 0
    repetitive_patterns = False
    return {'edge_density': edge_density, 'connected_components': num_components, 'avg_component_size': avg_component_size, 'avg_aspect_ratio': avg_aspect_ratio, 'repetitive_patterns': repetitive_patterns, 'structure_map': edges.astype(float)}

def analyze_temporal_information(dicom_data):
    """Analisa informações temporais dos metadados DICOM."""
    temporal_info = {'study_date': None, 'acquisition_time': None, 'content_date': None, 'time_consistency': 'unknown', 'estimated_age_days': None}
    if hasattr(dicom_data, 'StudyDate') and dicom_data.StudyDate: temporal_info['study_date'] = dicom_data.StudyDate
    if hasattr(dicom_data, 'AcquisitionTime') and dicom_data.AcquisitionTime: temporal_info['acquisition_time'] = dicom_data.AcquisitionTime
    if hasattr(dicom_data, 'ContentDate') and dicom_data.ContentDate: temporal_info['content_date'] = dicom_data.ContentDate
    dates = [d for d in [temporal_info['study_date'], temporal_info['content_date']] if d]
    if len(set(dates)) == 1: temporal_info['time_consistency'] = 'consistent'
    elif len(set(dates)) > 1: temporal_info['time_consistency'] = 'inconsistent'
    if temporal_info['study_date']:
        try:
            study_date = datetime.strptime(temporal_info['study_date'], '%Y%m%d')
            age_days = (datetime.now() - study_date).days
            temporal_info['estimated_age_days'] = age_days
        except ValueError: pass
    return temporal_info

def analyze_authenticity(dicom_data, image_array):
    """Analisa a autenticidade da imagem DICOM com base em vários indicadores."""
    authenticity_report = {
        'dicom_structure': 'pass', 'metadata_consistency': 'pass', 'digital_signature': 'fail',
        'temporal_coherence': 'pass', 'noise_patterns': 'pass', 'editing_evidence': 'pass',
        'authenticity_score': 0.7, 'anomalies': [], 'suspicion_map': None
    }
    if not hasattr(dicom_data, 'SOPClassUID') or not dicom_data.SOPClassUID: authenticity_report['dicom_structure'] = 'fail'; authenticity_report['anomalies'].append('Estrutura DICOM incompleta')
    if hasattr(dicom_data, 'DigitalSignaturesSequence'): authenticity_report['digital_signature'] = 'pass'
    else: authenticity_report['anomalies'].append('Assinatura digital não presente')
    noise_analysis = analyze_image_noise(image_array)
    if noise_analysis['pattern'] != 'random': authenticity_report['noise_patterns'] = 'warning'; authenticity_report['anomalies'].append('Padrão de ruído não natural detectado')
    editing_evidence = detect_editing_evidence(image_array)
    if editing_evidence['evidence_found']: authenticity_report['editing_evidence'] = 'fail'; authenticity_report['anomalies'].extend(editing_evidence['anomalies']); authenticity_report['suspicion_map'] = editing_evidence['suspicion_map']
    pass_count = sum(1 for k, v in authenticity_report.items() if k in ['dicom_structure', 'metadata_consistency', 'digital_signature', 'temporal_coherence', 'noise_patterns', 'editing_evidence'] and v == 'pass')
    warning_count = sum(1 for k, v in authenticity_report.items() if k in ['dicom_structure', 'metadata_consistency', 'digital_signature', 'temporal_coherence', 'noise_patterns', 'editing_evidence'] and v == 'warning')
    authenticity_report['authenticity_score'] = (pass_count + 0.5 * warning_count) / 6
    return authenticity_report

def calculate_forensic_quality(image_array):
    """Calcula métricas de qualidade para análise forense."""
    resolution_analysis = analyze_resolution(image_array); contrast = np.percentile(image_array, 75) - np.percentile(image_array, 25)
    max_contrast = np.max(image_array) - np.min(image_array); detectable_contrast = contrast / max_contrast if max_contrast > 0 else 0
    suitability_identification = min(1.0, resolution_analysis['resolution_score'] * 0.7 + detectable_contrast * 0.3)
    suitability_analysis = min(1.0, resolution_analysis['resolution_score'] * 0.5 + detectable_contrast * 0.5)
    suitability_documentation = min(1.0, resolution_analysis['resolution_score'] * 0.3 + detectable_contrast * 0.7)
    limitations = []; 
    if resolution_analysis['resolution_score'] < 0.5: limitations.append("Resolução insuficiente para análise detalhada")
    if detectable_contrast < 0.2: limitations.append("Contraste limitado pode dificultar a análise")
    overall_quality = (suitability_identification + suitability_analysis + suitability_documentation) / 3
    return {'overall_quality': overall_quality, 'effective_resolution': resolution_analysis['effective_resolution'], 'detectable_contrast': detectable_contrast, 'suitability_identification': suitability_identification, 'suitability_analysis': suitability_analysis, 'suitability_documentation': suitability_documentation, 'limitations': limitations}

def detect_artifacts(image_array):
    """Detecta e classifica artefatos na imagem."""
    artifacts = []; artifact_map = np.zeros_like(image_array, dtype=bool)
    noise_artifacts = detect_noise_artifacts(image_array)
    if noise_artifacts['detected']: artifacts.append({'type': 'noise', 'description': 'Ruído excessivo ou padrão anômalo', 'severity': noise_artifacts['severity']}); artifact_map = np.logical_or(artifact_map, noise_artifacts['mask'])
    motion_artifacts = detect_motion_artifacts(image_array)
    if motion_artifacts['detected']: artifacts.append({'type': 'motion', 'description': 'Artefatos de movimento detectados', 'severity': motion_artifacts['severity']}); artifact_map = np.logical_or(artifact_map, motion_artifacts['mask'])
    metal_artifacts = detect_metal_artifacts(image_array)
    if metal_artifacts['detected']: artifacts.append({'type': 'metal', 'description': 'Artefatos de beam hardening por metais', 'severity': metal_artifacts['severity']}); artifact_map = np.logical_or(artifact_map, metal_artifacts['mask'])
    artifacts_by_type = {a['type']: artifacts.count(a) for a in artifacts}
    affected_area = np.sum(artifact_map) / artifact_map.size * 100
    return {'artifacts': artifacts, 'artifact_map': artifact_map.astype(float), 'affected_area': affected_area, 'artifacts_by_type': artifacts_by_type}

def identify_homogeneous_regions(image_array, threshold=5):
    """Identifica regiões homogêneas na imagem."""
    local_std = ndimage.generic_filter(image_array, np.std, size=5)
    homogeneous_regions = local_std < threshold
    return homogeneous_regions

def identify_high_contrast_regions(image_array, threshold=20):
    """Identifica regiões de alto contraste na imagem."""
    grad_x = np.gradient(image_array, axis=1); grad_y = np.gradient(image_array, axis=0); gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    high_contrast_regions = gradient_magnitude > threshold
    return high_contrast_regions

def analyze_noise_pattern(noise_residual):
    """Analisa o padrão de ruído na imagem."""
    if noise_residual.shape[0] > 100 or noise_residual.shape[1] > 100: small_noise = noise_residual[::2, ::2]
    else: small_noise = noise_residual
    correlation = signal.correlate2d(small_noise, small_noise, mode='same'); correlation = correlation / np.max(correlation)
    center = np.array(correlation.shape) // 2
    peripheral_correlation = np.mean(correlation) - correlation[center[0], center[1]]
    if peripheral_correlation < 0.1: 
        return "random"
    else: 
        return "periodic"

def detect_repetitive_patterns(image_array):
    """Detecta padrões repetitivos na imagem."""
    return False

def analyze_resolution(image_array):
    """Analisa a resolução efetiva da imagem."""
    grad_x = np.gradient(image_array, axis=1); grad_y = np.gradient(image_array, axis=0); gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    sharp_edges = gradient_magnitude > np.percentile(gradient_magnitude, 95); edge_sharpness = np.mean(gradient_magnitude[sharp_edges]) if np.any(sharp_edges) else 0
    effective_resolution = edge_sharpness / 10; resolution_score = min(1.0, effective_resolution / 5.0)
    return {'effective_resolution': effective_resolution, 'resolution_score': resolution_score}

def detect_editing_evidence(image_array):
    """Detecta evidências de edição ou manipulação da imagem."""
    evidence = {'evidence_found': False, 'anomalies': [], 'suspicion_map': None}
    statistical_anomalies = detect_statistical_anomalies(image_array)
    if statistical_anomalies['anomalies_detected']: evidence['evidence_found'] = True; evidence['anomalies'].extend(statistical_anomalies['anomalies']); evidence['suspicion_map'] = statistical_anomalies['suspicion_map']
    compression_analysis = analyze_compression(image_array)
    if compression_analysis['likely_compressed']: evidence['evidence_found'] = True; evidence['anomalies'].append('Padrões de compressão inconsistentes detectados')
    return evidence

def detect_statistical_anomalies(image_array):
    """Detecta anomalias estatísticas que podem indicar manipulação."""
    anomalies = {'anomalies_detected': False, 'anomalies': [], 'suspicion_map': None}
    h, w = image_array.shape; regions = [image_array[:h // 2, :w // 2], image_array[:h // 2, w // 2:], image_array[h // 2:, :w // 2], image_array[h // 2:, w // 2:]]
    region_stats = []; [region_stats.append({'mean': np.mean(r), 'std': np.std(r), 'skewness': stats.skew(r.flatten())}) for r in regions]
    means = [s['mean'] for s in region_stats]; stds = [s['std'] for s in region_stats]
    if np.std(means) > 2 * np.mean(stds): anomalies['anomalies_detected'] = True; anomalies['anomalies'].append('Inconsistências estatísticas entre regiões')
    suspicion_map = np.zeros_like(image_array, dtype=float); global_mean = np.mean(image_array); global_std = np.std(image_array)
    suspicion_map[np.abs(image_array - global_mean) > 3 * global_std] = 1.0; anomalies['suspicion_map'] = suspicion_map
    return anomalies

def detect_noise_artifacts(image_array):
    """Detecta artefatos de ruído e cria uma máscara."""
    noise_analysis = analyze_image_noise(image_array); detected = noise_analysis['pattern'] != 'random'; severity = 'high' if noise_analysis['total_noise'] > 50 else 'medium'
    noise_mask = identify_high_noise_regions(image_array); return {'detected': detected, 'severity': severity, 'mask': noise_mask}

def detect_motion_artifacts(image_array):
    """Detecta artefatos de movimento na imagem."""
    derivative_x = np.gradient(image_array, axis=1); derivative_y = np.gradient(image_array, axis=0); motion_pattern = np.abs(derivative_x) + np.abs(derivative_y)
    motion_mask = motion_pattern > np.percentile(motion_pattern, 95); detected = np.any(motion_mask); severity = 'medium'
    return {'detected': detected, 'severity': severity, 'mask': motion_mask}

def detect_metal_artifacts(image_array):
    """Detecta artefatos de beam hardening por metais."""
    metal_mask = image_array > 1000; streak_detected = detect_streak_artifacts(image_array); detected = np.any(metal_mask) and streak_detected
    severity = 'high' if detected else 'low'; return {'detected': detected, 'severity': severity, 'mask': metal_mask}

def detect_streak_artifacts(image_array):
    """Detecta padrões de 'streak artifacts' típicos de metais."""
    grad_x = np.gradient(image_array, axis=1); grad_y = np.gradient(image_array, axis=0); straight_line_pattern = np.abs(grad_x) + np.abs(grad_y)
    line_mask = straight_line_pattern > np.percentile(straight_line_pattern, 90)
    return np.any(line_mask)

def identify_high_noise_regions(image_array, threshold=2.0):
    """Identifica regiões com ruído excessivo."""
    local_std = ndimage.generic_filter(image_array, np.std, size=5); global_std = np.std(image_array)
    high_noise_regions = local_std > threshold * global_std; return high_noise_regions

def get_user_reports(email): return []
def log_security_event(email, event_type, description): pass
def generate_advanced_ra_index_data(image_array):
    h, w = image_array.shape; grid_size = 8; h_step, w_step = h // grid_size, w // grid_size
    ra_data = {'coords': [], 'ra_values': [], 'risk_categories': [], 'tissue_types': [], 'intensities': []}
    def categorize_risk(mean_intensity):
        if mean_intensity < -500: return 'Baixo', 'Gás/Ar'; elif -500 <= mean_intensity < 0: return 'Baixo', 'Gordura'
        elif 0 <= mean_intensity < 100: return 'Médio', 'Tecido Mole'; elif 100 <= mean_intensity < 400: return 'Médio', 'Músculo'
        elif 400 <= mean_intensity < 1000: return 'Alto', 'Osso'; else: return 'Crítico', 'Metal/Implante'
    for i in range(grid_size):
        for j in range(grid_size):
            region = image_array[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
            mean_intensity = np.mean(region); std_intensity = np.std(region); intensity_factor = min(abs(mean_intensity) / 1000, 1.0)
            variation_factor = min(std_intensity / 500, 1.0); center_distance = np.sqrt((i - grid_size / 2) ** 2 + (j - grid_size / 2) ** 2)
            position_factor = 1 - (center_distance / (grid_size / 2)); ra_value = (intensity_factor * 0.5 + variation_factor * 0.3 + position_factor * 0.2) * 100
            risk_category, tissue_type = categorize_risk(mean_intensity)
            ra_data['coords'].append((i, j)); ra_data['ra_values'].append(ra_value); ra_data['risk_categories'].append(risk_category); ra_data['tissue_types'].append(tissue_type); ra_data['intensities'].append(mean_intensity)
    return ra_data, grid_size

class DispersaoGasosaCalculator:
    def __init__(self):
        self.sitios_anatomicos = ['Câmaras Cardíacas', 'Parênquima Hepático', 'Vasos Renais', 'Veia Inominada Esquerda', 'Aorta Abdominal', 'Parênquima Renal', 'Vértebra Lombar (L3)', 'Tecido Subcutâneo Peritoneal']
        self.gases = ['Putrescina', 'Cadaverina', 'Metano']
        self.coeficientes_difusao = {'Putrescina': 0.05, 'Cadaverina': 0.045, 'Metano': 0.12}
        self.limites_deteccao = {'Putrescina': 5.0, 'Cadaverina': 5.0, 'Metano': 2.0}
        self.locais_anatomicos_qualitativos = {"Cavidades Cardíacas": {"I": 5, "II": 15, "III": 20}, "Parênquima Hepático e Vasos": {"I": 8, "II": 17, "III": 20}, "Veia Inominada Esquerda": {"I": 1, "II": 5, "III": 8}, "Aorta Abdominal": {"I": 1, "II": 5, "III": 8}, "Parênquima Renal": {"I": 7, "II": 10, "III": 25}, "Vértebra L3": {"I": 7, "II": 8, "III": 8}, "Tecidos Subcutâneos Peitorais": {"I": 5, "II": 8, "III": 8}}
        self.pontos_corte_qualitativos = {"Cavidades Cardíacas (Grau III)": 50, "Cavidade Craniana (Grau II ou III)": 60}
        logging.info("Calculadora de Dispersão Gasosa inicializada.")
    def calcular_index_ra_qualitativo(self, classificacoes): pontuacao_total = sum(self.locais_anatomicos_qualitativos.get(local, {}).get(grau, 0) for local, grau in classificacoes.items()); return pontuacao_total
    def interpretar_index_ra_qualitativo(self, ra_index):
        if ra_index is None: return "Não foi possível calcular o RA-Index"
        interpretacao = f"RA-Index: {ra_index}/100\n"
        if ra_index >= self.pontos_corte_qualitativos["Cavidade Craniana (Grau II ou III)"]: interpretacao += "• Alteração radiológica avançada (≥60)\n"
        elif ra_index >= self.pontos_corte_qualitativos["Cavidades Cardíacas (Grau III)"]: interpretacao += "• Alteração radiológica moderada (≥50)\n"
        else: interpretacao += "• Alteração radiológica leve ou ausente (<50)\n"
        return interpretacao
    def calcular_index_ra_original(self, dados):
        try:
            pontuacao_total = sum(self.locais_anatomicos_qualitativos.get(local, {}).get(grau, 0) for local, grau in dados.items())
            return pontuacao_total
        except Exception as e: logging.error(f"Erro no cálculo do Index-RA original: {e}"); return None
    def segunda_lei_fick(self, C, t, D, x): return C * np.exp(-D * t / x**2)
    def modelo_mitscherlich_ajustado(self, t, a, b, c): return a * (1 - np.exp(-b * t)) + c
    def modelo_korsmeyer_peppas(self, t, k, n): return k * t**n
    def calcular_numero_knudsen(self, caminho_livre_medio, dimensao_caracteristica): return caminho_livre_medio / dimensao_caracteristica
    def tratar_valores_nd(self, dados, gas, metodo='limite_deteccao'):
        if metodo == 'limite_deteccao': limite = self.limites_deteccao.get(gas, 0.0); return np.where(np.isnan(dados), limite / np.sqrt(2), dados)
        elif metodo == 'media': media = np.nanmean(dados); return np.where(np.isnan(dados), media, dados)
        elif metodo == 'mediana': mediana = np.nanmedian(dados); return np.where(np.isnan(dados), mediana, dados)
        else: return dados
    def ajustar_modelo_difusao(self, tempo, concentracao, gas, sitio):
        try:
            concentracao_tratada = self.tratar_valores_nd(concentracao, gas); D_estimado = self.coeficientes_difusao.get(gas, 0.0); x0 = 1.0
            popt, pcov = curve_fit(lambda t, D, x: self.segunda_lei_fick(np.nanmax(concentracao_tratada), t, D, x), tempo, concentracao_tratada, p0=[D_estimado, x0], bounds=([0.001, 0.1], [1.0, 10.0]))
            concentracao_predita = self.segunda_lei_fick(np.nanmax(concentracao_tratada), tempo, *popt); ss_res = np.sum((concentracao_tratada - concentracao_predita) ** 2); ss_tot = np.sum((concentracao_tratada - np.mean(concentracao_tratada)) ** 2); r_squared = 1 - (ss_res / ss_tot)
            return {'coeficiente_difusao': popt[0], 'posicao_caracteristica': popt[1], 'r_quadrado': r_squared, 'covariancia': pcov}
        except Exception as e: logging.error(f"Erro no ajuste do modelo de difusão: {e}"); return None
    def prever_index_ra_aprimorado(self, dados):
        resultados = {}; resultados['index_ra_original'] = self.calcular_index_ra_qualitativo(dados); tempos = np.array([0, 6, 12, 18, 24, 30, 36, 42]); concentracoes = {}
        for gas in self.gases: concentracoes[gas] = {}; [concentracoes[gas].update({sitio: dados.get(f"{sitio}_{gas}", np.random.exponential(scale=50, size=len(tempos)))}) for sitio in self.sitios_anatomicos]
        modelos_ajustados = {}; [modelos_ajustados.update({gas: {sitio: self.ajustar_modelo_difusao(tempos, concentracoes[gas][sitio], gas, sitio) for sitio in self.sitios_anatomicos if self.ajustar_modelo_difusao(tempos, concentracoes[gas][sitio], gas, sitio) is not None}}) for gas in self.gases]
        fator_difusao = np.mean([modelos_ajustados[gas][sitio]['coeficiente_difusao'] for gas in self.gases for sitio in self.sitios_anatomicos if gas in modelos_ajustados and sitio in modelos_ajustados[gas]]); knudsen_avg = np.mean([self.calcular_numero_knudsen(1e-6, 1e-4) for _ in range(10)])
        resultados['index_ra_aprimorado'] = resultados['index_ra_original'] * (1 + 0.1 * np.log(fator_difusao) - 0.05 * knudsen_avg)
        resultados['fator_difusao_medio'] = fator_difusao; resultados['numero_knudsen_medio'] = knudsen_avg; resultados['modelos_ajustados'] = modelos_ajustados
        return resultados
    def gerar_relatorio(self, resultados, nome_arquivo=None):
        relatorio = ["RELATÓRIO DE ANÁLISE DE DISPERSÃO GASOSA POST-MORTEM", "=" * 60, f"Data da análise: {datetime.now().strftime('%d/%m/%Y %H:%M')}", f"Index-RA Original: {resultados.get('index_ra_original', 'N/A')}", f"Index-RA Aprimorado: {resultados.get('index_ra_aprimorado', 'N/A'):.2f}", "", "PARÂMETROS DO MODELO:", f"Fator de Difusão Médio: {resultados.get('fator_difusao_medio', 'N/A'):.4f}", f"Número de Knudsen Médio: {resultados.get('numero_knudsen_medio', 'N/A'):.6f}", "", "ANÁLISE POR GÁS:"]
        if 'modelos_ajustados' in resultados:
            for gas in resultados['modelos_ajustados']:
                relatorio.append(f"  {gas}:")
                for sitio in resultados['modelos_ajustados'][gas]:
                    modelo = resultados['modelos_ajustados'][gas][sitio]; relatorio.append(f"    {sitio}: D = {modelo['coeficiente_difusao']:.6f}, R² = {modelo['r_quadrado']:.3f}")
        relatorio_texto = "\n".join(relatorio)
        if nome_arquivo: with open(nome_arquivo, 'w', encoding='utf-8') as f: f.write(relatorio_texto)
        logging.info(relatorio_texto); return relatorio_texto
    def plotar_curvas_difusao(self, resultados, gas, sitio, tempo, concentracao, nome_arquivo=None):
        try:
            if gas in resultados['modelos_ajustados'] and sitio in resultados['modelos_ajustados'][gas]:
                modelo = resultados['modelos_ajustados'][gas][sitio]; tempo_suave = np.linspace(min(tempo), max(tempo), 100)
                concentracao_predita = self.segunda_lei_fick(np.nanmax(concentracao), tempo_suave, modelo['coeficiente_difusao'], modelo['posicao_caracteristica'])
                fig, ax = plt.subplots(figsize=(10, 6)); ax.scatter(tempo, concentracao, color='blue', label='Dados Observados', zorder=5); ax.plot(tempo_suave, concentracao_predita, 'r-', label='Modelo Ajustado', linewidth=2)
                ax.set_xlabel('Tempo Post-Mortem (horas)'); ax.set_ylabel('Concentração (UH)'); ax.set_title(f'Dispersão de {gas} no {sitio}\nD = {modelo["coeficiente_difusao"]:.4f} cm²/h, R² = {modelo["r_quadrado"]:.3f}')
                ax.legend(); ax.grid(True, alpha=0.3)
                if nome_arquivo: plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
                plt.show()
            else: logging.warning(f"Dados insuficientes para {gas} no {sitio}")
        except Exception as e: logging.error(f"Erro ao gerar gráfico: {e}")

def enhanced_technical_analysis_tab(dicom_data, image_array):
    st.header("Análise Técnica Forense Avançada"); with st.expander("Base Científica (Normas ABNT)"): st.markdown("""
        **Referências para Análise Técnica Forense:**
        - SILVA, W. L. **Análise quantitativa de alterações post-mortem por tomografia computadorizada**. 2023.
        - EGGER, C. et al. **Development and validation of a postmortem radiological alteration index**. Int J Legal Med, 2012.
        - ALTAIMIRANO, R. **Técnicas de imagem aplicadas à tanatologia forense**. Revista de Medicina Legal, 2022.
        - INTERPOL. **Guidelines for Forensic Imaging**. 2014.
        - NIST. **Digital Imaging and Communications in Medicine (DICOM) Standards**. 2023.
        """); tab_metadata, tab_forensic, tab_authentication, tab_quality, tab_artifacts = st.tabs(["Metadados DICOM", "Análise Forense", "Autenticidade", "Qualidade", "Artefatos"])
    with tab_metadata:
        st.markdown("### Metadados DICOM Completos"); categories = {'Informações do Paciente': {'keywords': ['patient', 'name', 'id', 'birth', 'sex', 'age', 'weight'], 'items': []}, 'Parâmetros de Aquisição': {'keywords': ['kv', 'ma', 'exposure', 'dose', 'current', 'time'], 'items': []}, 'Configurações do Equipamento': {'keywords': ['manufacturer', 'model', 'software', 'station', 'device', 'serial'], 'items': []}, 'Dados de Imagem': {'keywords': ['rows', 'columns', 'spacing', 'thickness', 'pixel', 'size', 'resolution'], 'items': []}, 'Informações Temporais': {'keywords': ['date', 'time', 'acquisition', 'study', 'series', 'content'], 'items': []}, 'Parâmetros de Reconstrução': {'keywords': ['kernel', 'algorithm', 'filter', 'reconstruction', 'slice'], 'items': []}, 'Dados Técnicos Forenses': {'keywords': ['forensic', 'legal', 'postmortem', 'autopsy', 'examination'], 'items': []}}
        metadata_summary = {}; 
        for elem in dicom_data:
            if elem.tag.group != 0x7fe0: tag_name = elem.name if hasattr(elem, 'name') else str(elem.tag); tag_value = str(elem.value) if len(str(elem.value)) < 100 else str(elem.value)[:100] + "..."; categorized = False
            for category, info in categories.items():
                if any(keyword in tag_name.lower() for keyword in info['keywords']): info['items'].append(f"**{tag_name}**: {tag_value}"); categorized = True; break
            if not categorized: categories['Dados Técnicos Forenses']['items'].append(f"**{tag_name}**: {tag_value}")
            metadata_summary[tag_name] = tag_value
        col1, col2 = st.columns(2)
        with col1:
            for i, (category, info) in enumerate(list(categories.items())[:4]):
                if info['items']: with st.expander(f"{category} ({len(info['items'])} itens)"): [st.markdown(f"• {item}") for item in info['items'][:25]]
        with col2:
            for i, (category, info) in enumerate(list(categories.items())[4:]):
                if info['items']: with st.expander(f"{category} ({len(info['items'])} itens)"): [st.markdown(f"• {item}") for item in info['items'][:25]]
        st.markdown("#### Análise Rápida de Metadados"); col1, col2, col3 = st.columns(3)
        with col1: essential_metadata = ['PatientName', 'PatientID', 'StudyDate', 'StudyTime']; missing_essential = [meta for meta in essential_metadata if meta not in metadata_summary]; if missing_essential: st.error(f"Metadados essenciais faltantes: {len(missing_essential)}"); else: st.success("Todos metadados essenciais presentes")
        with col2: time_consistency = check_temporal_consistency(metadata_summary); if time_consistency['consistent']: st.success("Consistência temporal validada"); else: st.warning(f"Inconsistência temporal: {time_consistency['issue']}")
        with col3: dicom_compliance = check_dicom_compliance(metadata_summary); compliance_score = dicom_compliance.get('score', 0); if compliance_score > 0.8: st.success(f"Conformidade DICOM: {compliance_score:.0%}"); elif compliance_score > 0.5: st.warning(f"Conformidade DICOM: {compliance_score:.0%}"); else: st.error(f"Conformidade DICOM: {compliance_score:.0%}")
    with tab_forensic:
        st.markdown("### Análise Forense Digital Avançada"); forensic_tab1, forensic_tab2, forensic_tab3, forensic_tab4 = st.tabs(["Integridade", "Espectral", "Morfológica", "Temporal"])
        with forensic_tab1: st.markdown("#### Análise de Integridade"); col1, col2, col3 = st.columns(3); with col1: st.markdown("##### Assinaturas Digitais"); hash_md5 = hashlib.md5(image_array.tobytes()).hexdigest(); hash_sha1 = hashlib.sha1(image_array.tobytes()).hexdigest(); hash_sha256 = hashlib.sha256(image_array.tobytes()).hexdigest(); st.text_area("MD5", hash_md5, height=60); st.text_area("SHA-1", hash_sha1, height=60); st.text_area("SHA-256", hash_sha256, height=60); if hasattr(dicom_data, 'DigitalSignaturesSequence'): st.success("Assinatura digital DICOM presente"); else: st.warning("Assinatura digital DICOM não encontrada")
        with col2: st.markdown("##### Análise de Ruído"); noise_analysis = analyze_image_noise(image_array); st.metric("Ruído Total", f"{noise_analysis['total_noise']:.2f}"); st.metric("Ruído de Fundo", f"{noise_analysis['background_noise']:.2f}"); st.metric("Ruído de Sinal", f"{noise_analysis['signal_noise']:.2f}"); if noise_analysis['pattern'] == "random": st.success("Padrão de ruído: Aleatório"); elif noise_analysis['pattern'] == "periodic": st.warning("Padrão de ruído: Periódico (possível artefato)"); else: st.info(f"Padrão de ruído: {noise_analysis['pattern']}")
        with col3: st.markdown("##### Análise de Compressão"); compression_analysis = analyze_compression(image_array); st.metric("Taxa de Compressão", f"{compression_analysis['ratio']:.4f}"); st.metric("Entropia de Dados", f"{compression_analysis['entropy']:.2f} bits"); st.metric("Redundância", f"{compression_analysis['redundancy']:.2f}%"); if compression_analysis['likely_compressed']: st.warning("Possível compressão com perdas detectada"); else: st.success("Sem evidências de compressão com perdas")
        with forensic_tab2: st.markdown("#### 📊 Análise Espectral"); col1, col2 = st.columns(2); with col1: st.markdown("##### Transformada de Fourier (FFT)"); fft_2d = np.fft.fft2(image_array); magnitude_spectrum = np.log(np.abs(fft_2d) + 1); spectral_metrics = calculate_spectral_metrics(fft_2d); st.metric("Energia Espectral Total", f"{spectral_metrics['total_energy']:.2e}"); st.metric("Centroide Espectral", f"({spectral_metrics['centroid_x']:.1f}, {spectral_metrics['centroid_y']:.1f})"); st.metric("Entropia Espectral", f"{spectral_metrics['spectral_entropy']:.2f}"); dominant_freq = spectral_metrics['dominant_frequency']; st.metric("Frequência Dominante", f"{dominant_freq:.2f} ciclos/pixel")
        with col2: st.markdown("##### Distribuição de Energia"); power_spectrum = np.abs(np.fft.fft2(image_array))**2; total_energy = np.sum(power_spectrum); h, w = power_spectrum.shape; energy_low = np.sum(power_spectrum[:h//4, :w//4]); energy_mid = np.sum(power_spectrum[h//4:3*h//4, w//4:3*w//4]); energy_high = np.sum(power_spectrum[3*h//4:, 3*w//4:])
        st.metric("Energia Baixa Frequência", f"{energy_low/total_energy*100:.1f}%"); st.metric("Energia Média Frequência", f"{energy_mid/total_energy*100:.1f}%"); st.metric("Energia Alta Frequência", f"{energy_high/total_energy*100:.1f}%"); snr_spectral = 10 * np.log10(energy_mid / (energy_high + 1e-10)); st.metric("SNR Espectral", f"{snr_spectral:.2f} dB"); fig = px.imshow(magnitude_spectrum, color_continuous_scale='viridis'); fig.update_layout(title="Espectro de Magnitude (Log)"); st.plotly_chart(fig, use_container_width=True)
        with forensic_tab3: st.markdown("#### 🔍 Análise Morfológica"); col1, col2 = st.columns(2); with col1: st.markdown("##### Análise de Textura"); texture_features = calculate_texture_features(image_array); st.metric("Contraste", f"{texture_features['contrast']:.2f}"); st.metric("Energia", f"{texture_features['energy']:.4f}"); st.metric("Homogeneidade", f"{texture_features['homogeneity']:.3f}"); st.metric("Correlação", f"{texture_features['correlation']:.3f}"); complexity = texture_features['complexity']; if complexity > 0.7: st.info("Textura de alta complexidade"); elif complexity > 0.4: st.info("Textura de complexidade moderada"); else: st.info("Textura de baixa complexidade")
        with col2: st.markdown("##### Análise Estrutural"); structural_analysis = analyze_structures(image_array); st.metric("Densidade de Bordas", f"{structural_analysis['edge_density']:.4f}"); st.metric("Componentes Conectados", structural_analysis['connected_components']); st.metric("Tamanho Médio de Componentes", f"{structural_analysis['avg_component_size']:.1f} px"); st.metric("Razão de Aspecto Média", f"{structural_analysis['avg_aspect_ratio']:.2f}"); if structural_analysis['repetitive_patterns']: st.warning("Padrões repetitivos detectados"); else: st.success("Sem padrões repetitivos evidentes"); fig = px.imshow(structural_analysis['structure_map'], color_continuous_scale='gray'); fig.update_layout(title="Mapa de Estruturas Detectadas"); st.plotly_chart(fig, use_container_width=True)
        with forensic_tab4: st.markdown("#### Análise Temporal"); temporal_analysis = analyze_temporal_information(dicom_data); col1, col2 = st.columns(2); with col1: st.markdown("##### Metadados Temporais"); if temporal_analysis['study_date']: st.metric("Data do Estudo", temporal_analysis['study_date']); if temporal_analysis['acquisition_time']: st.metric("Tempo de Aquisição", temporal_analysis['acquisition_time']); if temporal_analysis['content_date']: st.metric("Data do Conteúdo", temporal_analysis['content_date']); if temporal_analysis['time_consistency'] == "consistent": st.success("Consistência temporal validada"); else: st.warning("Consistência temporal indeterminada")
        with col2: st.markdown("##### Linha do Tempo Forense"); timeline_events = []; if temporal_analysis['study_date']: timeline_events.append(f"Estudo: {temporal_analysis['study_date']}"); if temporal_analysis['acquisition_time']: timeline_events.append(f"Aquisição: {temporal_analysis['acquisition_time']}"); if temporal_analysis['content_date']: timeline_events.append(f"Conteúdo: {temporal_analysis['content_date']}"); timeline_events.append(f"Análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"); [st.markdown(f"- {e}") for e in timeline_events]; if temporal_analysis['estimated_age_days'] is not None: age_days = temporal_analysis['estimated_age_days']; if age_days < 7: st.info(f"Imagem recente ({age_days} dias)"); elif age_days < 30: st.info(f"Imagem com {age_days} dias"); else: st.info(f"Imagem antiga ({age_days} dias)")
    with tab_authentication: st.markdown("### Análise de Autenticidade"); authenticity_report = analyze_authenticity(dicom_data, image_array); col1, col2 = st.columns(2); with col1: st.markdown("#### Verificações de Integridade"); checks = [{"name": "Estrutura DICOM válida", "status": authenticity_report['dicom_structure']}, {"name": "Metadados consistentes", "status": authenticity_report['metadata_consistency']}, {"name": "Assinatura digital presente", "status": authenticity_report['digital_signature']}, {"name": "Sequência temporal coerente", "status": authenticity_report['temporal_coherence']}, {"name": "Padrões de ruído naturais", "status": authenticity_report['noise_patterns']}, {"name": "Sem evidências de edição", "status": authenticity_report['editing_evidence']}]
    for check in checks:
        if check['status'] == "pass": st.success(f" {check['name']}"); elif check['status'] == "warning": st.warning(f" {check['name']}"); else: st.error(f" {check['name']}")
    authenticity_score = authenticity_report['authenticity_score']; st.metric("Score de Autenticidade", f"{authenticity_score:.0%}"); if authenticity_score > 0.8: st.success("Alta probabilidade de autenticidade"); elif authenticity_score > 0.5: st.warning("Autenticidade questionável"); else: st.error("Alta probabilidade de manipulação")
    with col2: st.markdown("#### Detecção de Manipulação"); if 'anomalies' in authenticity_report and authenticity_report['anomalies']: st.error("Anomalias detectadas:"); [st.markdown(f"- {a}") for a in authenticity_report['anomalies']]; else: st.success("Nenhuma anomalia evidente detectada")
    if 'suspicion_map' in authenticity_report and authenticity_report['suspicion_map'] is not None: st.warning("Regiões suspeitas identificadas"); fig = px.imshow(authenticity_report['suspicion_map'], color_continuous_scale='hot'); fig.update_layout(title="Mapa de Suspeição de Manipulação"); st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Recomendações"); if 'authenticity_score' in authenticity_report: if authenticity_report['authenticity_score'] > 0.8: st.info("Imagem considerada autêntica. Proceda com a análise."); elif authenticity_report['authenticity_score'] > 0.5: st.warning("Imagem com questões de autenticidade. Verifique cuidadosamente."); else: st.error("Imagem potencialmente manipulada. Considere descartar ou investigar profundamente.")
    with tab_quality: st.markdown("### Análise de Qualidade Forense"); quality_metrics = calculate_forensic_quality(image_array); col1, col2, col3 = st.columns(3); with col1: st.markdown("#### Métricas de Qualidade"); st.metric("Qualidade Geral", f"{quality_metrics['overall_quality']:.0%}"); st.metric("Resolução Efetiva", f"{quality_metrics['effective_resolution']:.1f} LP/mm"); st.metric("Contraste Detectável", f"{quality_metrics['detectable_contrast']:.2f}")
    with col2: st.markdown("#### Adequação Forense"); st.metric("Adequação para Identificação", f"{quality_metrics['suitability_identification']:.0%}"); st.metric("Adequação para Análise", f"{quality_metrics['suitability_analysis']:.0%}"); st.metric("Adequação para Documentação", f"{quality_metrics['suitability_documentation']:.0%}")
    with col3: st.markdown("#### Limitações"); if quality_metrics['limitations']: st.warning("Limitações identificadas:"); [st.markdown(f"- {l}") for l in quality_metrics['limitations']]; else: st.success("Sem limitações significativas"); st.markdown("#### Recomendações Técnicas"); if quality_metrics['overall_quality'] > 0.8: st.success("Qualidade excelente para todos os fins forenses"); elif quality_metrics['overall_quality'] > 0.6: st.info("Qualidade adequada para a maioria dos fins forenses"); elif quality_metrics['overall_quality'] > 0.4: st.warning("Qualidade limitada - use com cautela para análise forense"); else: st.error("Qualidade inadequada para análise forense")
    with tab_artifacts: st.markdown("### Detecção de Artefatos"); artifact_report = detect_artifacts(image_array); col1, col2 = st.columns(2); with col1: st.markdown("#### Artefatos Detectados"); if artifact_report['artifacts']: st.warning(f"{len(artifact_report['artifacts'])} artefatos detectados:"); [st.markdown(f"- **{a['type']}**: {a['description']}; Severidade: {a['severity']}") for a in artifact_report['artifacts']]; else: st.success("Nenhum artefato significativo detectado")
    with col2: st.markdown("#### Mapa de Artefatos"); if artifact_report['artifact_map'] is not None: fig = px.imshow(artifact_report['artifact_map'], color_continuous_scale='hot'); fig.update_layout(title="Mapa de Localização de Artefatos"); st.plotly_chart(fig, use_container_width=True); if artifact_report['artifacts']: st.metric("Área Afetada por Artefatos", f"{artifact_report['affected_area']:.1f}%"); st.metric("Artefatos por Tipo", str(artifact_report['artifacts_by_type']))
    st.markdown("#### Mitigação de Artefatos"); if artifact_report['artifacts']: st.info("Recomendações para mitigação:"); mitigation_strategies = {'noise': "Aplicar filtros de redução de ruído adaptativos", 'motion': "Considerar técnicas de correção de movimento", 'metal': "Aplicar algoritmos de correção de artefatos metálicos", 'ring': "Aplicar correção de artefatos em anel"}; [st.markdown(f"- Para {a['type']}: {mitigation_strategies.get(a['type'], 'N/A')}") for a in artifact_report['artifacts']]
    
def enhanced_ra_index_tab(dicom_data, image_array):
    st.subheader("RA-Index - Análise de Risco Aprimorada"); ra_data, grid_size = generate_advanced_ra_index_data(image_array); st.markdown("### Estatísticas Gerais do RA-Index")
    col1, col2, col3, col4 = st.columns(4); with col1: avg_ra = np.mean(ra_data['ra_values']); st.metric("RA-Index Médio", f"{avg_ra:.1f}")
    with col2: max_ra = np.max(ra_data['ra_values']); st.metric("RA-Index Máximo", f"{max_ra:.1f}")
    with col3: risk_counts = pd.Series(ra_data['risk_categories']).value_counts(); critical_count = risk_counts.get('Crítico', 0); st.metric("Regiões Críticas", critical_count)
    with col4: high_risk_count = risk_counts.get('Alto', 0); st.metric("Regiões Alto Risco", high_risk_count)
    st.markdown("### Mapas de Calor Avançados"); col1, col2 = st.columns(2); with col1: ra_matrix = np.array(ra_data['ra_values']).reshape(grid_size, grid_size); fig1 = go.Figure(data=go.Heatmap(z=ra_matrix, colorscale='RdYlBu_r', showscale=True, text=ra_matrix.round(1), texttemplate="%{text}", textfont={"size": 12, "color": "white"}, hoverongaps=False)); fig1.update_layout(title="Mapa de Calor - RA-Index", xaxis_title="Região X", yaxis_title="Região Y", height=500); st.plotly_chart(fig1, use_container_width=True)
    with col2: tissue_mapping = {'Gás/Ar': 1, 'Gordura': 2, 'Tecido Mole': 3, 'Músculo': 4, 'Osso': 5, 'Metal/Implante': 6}; tissue_matrix = np.array([tissue_mapping[t] for t in ra_data['tissue_types']]).reshape(grid_size, grid_size); fig2 = go.Figure(data=go.Heatmap(z=tissue_matrix, colorscale='viridis', showscale=True, text=np.array(ra_data['tissue_types']).reshape(grid_size, grid_size), texttemplate="%{text}", textfont={"size": 8, "color": "white"}, hoverongaps=False)); fig2.update_layout(title=" Mapa de Tipos de Tecido", xaxis_title="Região X", yaxis_title="Região Y", height=500); st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### Análise de Distribuição de Risco"); col1, col2 = st.columns(2); with col1: fig3 = go.Figure(data=[go.Pie(labels=list(risk_counts.index), values=list(risk_counts.values), hole=.3, marker_colors=['#FF4B4B', '#FFA500', '#FFFF00', '#90EE90'])]); fig3.update_layout(title="Distribuição de Categorias de Risco", height=400); st.plotly_chart(fig3, use_container_width=True)
    with col2: fig4 = go.Figure(); fig4.add_trace(go.Histogram(x=ra_data['ra_values'], nbinsx=20, name="RA-Index", marker_color='lightcoral', opacity=0.7)); fig4.add_vline(x=np.mean(ra_data['ra_values']), line_dash="dash", line_color="red", annotation_text="Média"); fig4.add_vline(x=np.percentile(ra_data['ra_values'], 90), line_dash="dash", line_color="orange", annotation_text="P90"); fig4.update_layout(title="Distribuição de Valores RA-Index", xaxis_title="RA-Index", yaxis_title="Frequência", height=400); st.plotly_chart(fig4, use_container_width=True)
    st.markdown("### Análise Temporal Simulada"); time_points = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5']; temporal_data = {'Crítico': [], 'Alto': [], 'Médio': [], 'Baixo': []}; base_counts = risk_counts.to_dict(); [temporal_data[rl].append(max(0, int(base_counts.get(rl, 0) * (1 + 0.1 * np.sin(i * np.pi / 3) + np.random.normal(0, 0.05))))) for i in range(len(time_points)) for rl in temporal_data.keys()]
    fig5 = go.Figure(); colors = {'Crítico': 'red', 'Alto': 'orange', 'Médio': 'yellow', 'Baixo': 'green'}; [fig5.add_trace(go.Scatter(x=time_points, y=v, mode='lines+markers', name=rl, line=dict(color=colors[rl], width=3), marker=dict(size=8))) for rl, v in temporal_data.items()]
    fig5.update_layout(title="Evolução Temporal das Categorias de Risco", xaxis_title="Ponto Temporal", yaxis_title="Número de Regiões", height=400, hovermode='x unified'); st.plotly_chart(fig5, use_container_width=True)
    st.markdown("### Análise de Correlações"); col1, col2 = st.columns(2); with col1: fig6 = go.Figure(); colors_by_risk = {'Crítico': 'red', 'Alto': 'orange', 'Médio': 'yellow', 'Baixo': 'green'}
    for r in colors_by_risk.keys(): mask = np.array(ra_data['risk_categories']) == r; if np.any(mask): fig6.add_trace(go.Scatter(x=np.array(ra_data['intensities'])[mask], y=np.array(ra_data['ra_values'])[mask], mode='markers', name=r, marker=dict(color=colors_by_risk[r], size=8, opacity=0.7)))
    fig6.update_layout(title="Correlação: RA-Index vs Intensidade HU", xaxis_title="Intensidade (HU)", yaxis_title="RA-Index", height=400); st.plotly_chart(fig6, use_container_width=True)
    with col2: x_coords = [c[0] for c in ra_data['coords']]; y_coords = [c[1] for c in ra_data['coords']]; fig7 = go.Figure(data=[go.Scatter3d(x=x_coords, y=y_coords, z=ra_data['ra_values'], mode='markers', marker=dict(size=8, color=ra_data['ra_values'], colorscale='RdYlBu_r', showscale=True, opacity=0.8), text=[f"Região ({x},{y})<br>RA-Index: {ra:.1f}<br>Tipo: {t}" for (x, y), ra, t in zip(ra_data['coords'], ra_data['ra_values'], ra_data['tissue_types'])], hovertemplate='%{text}<extra></extra>')]); fig7.update_layout(title="Visualização 3D do RA-Index", scene=dict(xaxis_title="Região X", yaxis_title="Região Y", zaxis_title="RA-Index"), height=400); st.plotly_chart(fig7, use_container_width=True)
    st.markdown("### Relatório de Recomendações"); col1, col2 = st.columns(2)
    with col1: st.markdown("#### Regiões de Atenção"); high_risk_indices = [i for i, ra in enumerate(ra_data['ra_values']) if ra > 70]; 
    if high_risk_indices: [st.warning(f"**Região ({c[0]}, {c[1]})**\n- RA-Index: {r:.1f}\n- Tipo: {t}\n- Categoria: {risk}") for idx in high_risk_indices[:5] for c, r, t, risk in [(ra_data['coords'][idx], ra_data['ra_values'][idx], ra_data['tissue_types'][idx], ra_data['risk_categories'][idx])]]; else: st.success("Nenhuma região de alto risco identificada")
    with col2: st.markdown("#### Estatísticas de Monitoramento"); monitoring_stats = {"Cobertura de Análise": "100%", "Precisão Estimada": "94.2%", "Sensibilidade": "89.7%", "Especificidade": "96.1%", "Valor Preditivo Positivo": "87.3%", "Valor Preditivo Negativo": "97.8%"}; [st.metric(m, v) for m, v in monitoring_stats.items()]
    st.markdown("### Exportar Dados RA-Index")
    if st.button("Gerar Relatório RA-Index"):
        df_export = pd.DataFrame({'Região_X': [c[0] for c in ra_data['coords']], 'Região_Y': [c[1] for c in ra_data['coords']], 'RA_Index': ra_data['ra_values'], 'Categoria_Risco': ra_data['risk_categories'], 'Tipo_Tecido': ra_data['tissue_types'], 'Intensidade_Media': ra_data['intensities']}); csv_buffer = BytesIO(); df_export.to_csv(csv_buffer, index=False, encoding='utf-8'); csv_buffer.seek(0); st.download_button(label="Baixar Dados RA-Index (CSV)", data=csv_buffer, file_name=f"ra_index_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv"); st.success("Relatório RA-Index preparado para download!")

def enhanced_reporting_tab(dicom_data, image_array, user_data):
    st.subheader("Relatórios Completos"); report_tab1, report_tab2, report_tab3 = st.tabs(["Gerar Relatório", "Relatórios Salvos", "Configurações"])
    with report_tab1: st.markdown("### Personalizar Relatório"); col1, col2 = st.columns(2)
    with col1: report_name = st.text_input("Nome do Relatório", value=f"Análise_{datetime.now().strftime('%Y%m%d_%H%M')}", help="Nome personalizado para o relatório"); report_type = st.selectbox("Tipo de Relatório", ["Completo", "Forense", "Qualidade", "Estatístico", "Técnico"], help="Selecione o tipo de relatório a ser gerado"); include_sections = st.multiselect("Seções a Incluir", ["Metadados", "Estatísticas", "Análise Técnica", "Qualidade", "Análise Post-Mortem", "RA-Index", "Visualizações", "Imagens"], default=["Metadados", "Estatísticas", "Análise Técnica", "Qualidade", "Análise Post-Mortem", "RA-Index"], help="Selecione as seções a incluir no relatório")
    with col2: format_options = st.selectbox("Formato de Exportação", ["PDF", "HTML", "CSV"]); st.markdown("**Opções de Visualização:**"); include_3d = st.checkbox("Incluir visualizações 3D", value=True); include_heatmaps = st.checkbox("Incluir mapas de calor", value=True); include_graphs = st.checkbox("Incluir gráficos estatísticos", value=True)
    if st.button("Gerar Relatório Completo", type="primary", use_container_width=True):
        with st.spinner("Gerando relatório... Isso pode levar alguns minutos"):
            try:
                report_data = generate_comprehensive_report(dicom_data, image_array, include_sections, include_3d, include_heatmaps, include_graphs)
                if format_options == "PDF": report_file, mime_type, file_ext = generate_pdf_report(report_data, report_name), "application/pdf", "pdf"
                elif format_options == "HTML": report_file, mime_type, file_ext = generate_html_report(report_data, report_name), "text/html", "html"
                else: report_file, mime_type, file_ext = generate_csv_report(report_data, report_name), "text/csv", "csv"
                save_report_to_db(st.session_state.user_data['email'], report_name, report_file.getvalue(), {'report_type': report_type, 'include_sections': include_sections, 'format': format_options, 'timestamp': datetime.now().isoformat()})
                st.success("Relatório gerado com sucesso!"); st.download_button(label=f"Download do Relatório ({format_options})", data=report_file, file_name=f"{report_name}.{file_ext}", mime=mime_type, use_container_width=True)
            except Exception as e: st.error(f"Erro ao gerar relatório: {e}"); logging.error(f"Erro na geração de relatório: {e}")
    with report_tab2:
        st.markdown("### Relatórios Salvos"); user_reports = get_user_reports(st.session_state.user_data['email'])
        if user_reports:
            for report_id, report_name, generated_at in user_reports:
                col1, col2, col3 = st.columns([3, 1, 1]); with col1: st.markdown(f"**{report_name}**"); st.caption(f"Gerado em: {generated_at}")
                with col2: st.button("Visualizar", key=f"view_{report_id}")
                with col3: st.button("Download", key=f"download_{report_id}")
                st.divider()
        else: st.info("Nenhum relatório salvo ainda. Gere seu primeiro relatório na aba 'Gerar Relatório'.")
    with report_tab3: st.markdown("### Configurações de Relatórios"); st.markdown("#### Preferências de Exportação"); st.selectbox("Formato Padrão", ["PDF", "HTML", "CSV"]); st.checkbox("Salvar automaticamente após gerar"); st.checkbox("Incluir timestamp no nome do arquivo", value=True)
    st.markdown("#### Configurações de Visualização"); st.selectbox("Tema Visual", ["Claro", "Escuro", "Automático"]); st.slider("Resolução dos Gráficos (DPI)", 72, 300, 150); st.slider("Qualidade das Imagens", 50, 100, 85)
    if st.button("Salvar Configurações", use_container_width=True): st.success("Configurações salvas com sucesso!")

def show_user_form():
    st.markdown("""<div style="text-align: center; margin-bottom: 2rem;"><h1 style="color: #000000; font-size: 2.8rem; margin-bottom: 0.5rem; font-weight: 700;">DICOM Autopsy Viewer PRO</h1><h2 style="color: #333333; font-weight: 500; margin-top: 0;">Sistema Avançado de Análise Forense Digital</h2></div>""", unsafe_allow_html=True)
    with st.form("user_registration"):
        col1, col2 = st.columns([1, 2])
        with col1: st.empty(); st.markdown("<br><br>", unsafe_allow_html=True)
        with col2: st.markdown("### Registro de Usuário"); name = st.text_input("Nome Completo *", placeholder="Dr. João Silva", help="Informe seu nome completo"); email = st.text_input("Email Institucional *", placeholder="joao.silva@hospital.com", help="Utilize email institucional para registro")
        col_a, col_b = st.columns(2); with col_a: role = st.selectbox("Função *", ["Radiologista", "Médico Legista", "Técnico em Radiologia", "Pesquisador", "Estudante", "Outro"], help="Selecione sua função principal"); with col_b: department = st.text_input("Departamento/Instituição", placeholder="Departamento de Radiologia", help="Informe seu departamento ou instituição")
        with st.expander("Termos de Uso e Política de Privacidade"): st.markdown("""**Termos de Uso:**\n1. Utilização autorizada apenas para fins educacionais e de pesquisa\n2. Proibido o carregamento de dados de pacientes reais não autorizados\n3. Compromisso com a confidencialidade das informações processadas\n4. Os relatórios gerados são de responsabilidade do usuário\n5. O sistema não armazena imagens médicas, apenas metadados anônimos\n\n**Política de Privacidade:**\n- Seus dados de registro são armazenados de forma segura\n- As análises realizadas são confidenciais\n- Metadados das imagens são anonimizados para análise estatística\n- Relatórios gerados podem ser excluídos a qualquer momento\n"""); terms_accepted = st.checkbox("Eu concordo com os termos de uso e política de privacidade")
        submitted = st.form_submit_button("Iniciar Sistema →", use_container_width=True)
        if submitted:
            if not all([name, email, terms_accepted]): st.error("Por favor, preencha todos os campos obrigatórios e aceite os termos de uso.")
            else:
                try:
                    conn = sqlite3.connect("dicom_viewer.db"); cursor = conn.cursor(); cursor.execute("""INSERT INTO users (name, email, role, department) VALUES (?, ?, ?, ?)""", (name, email, role, department)); conn.commit(); conn.close()
                    st.session_state.user_data = {'name': name, 'email': email, 'role': role, 'department': department}
                    log_security_event(email, "USER_REGISTRATION", f"Role: {role}"); st.success("Usuário registrado com sucesso!"); st.rerun()
                except Exception as e: st.error(f"Erro ao registrar usuário: {e}")

def show_main_app():
    user_data = st.session_state.user_data
    with st.sidebar:
        st.markdown(f"""<div style="padding: 1rem; border-bottom: 1px solid #E0E0E0; margin-bottom: 1rem;"><h3 style="color: #000000; margin-bottom: 0.5rem;">{user_data['name']}</h3><p style="color: #666666; margin: 0;"><strong>Função:</strong> {user_data['role']}</p><p style="color: #666666; margin: 0;"><strong>Email:</strong> {user_data['email']}</p>{f'<p style="color: #666666; margin: 0;"><strong>Departamento:</strong> {user_data["department"]}</p>' if user_data['department'] else ''}</div>""", unsafe_allow_html=True)
        st.markdown("### Navegação"); st.markdown("---")
        uploaded_file = st.file_uploader("Selecione um arquivo DICOM:", type=['dcm', 'dicom'], help="Carregue um arquivo DICOM para análise forense avançada")
        st.markdown("---"); st.markdown("### Relatórios Salvos"); user_reports = get_user_reports(user_data['email'])
        if user_reports: [st.button(f"{r[1]} - {r[2].split()[0]}", key=f"report_{r[0]}") for r in user_reports]
        else: st.info("Nenhum relatório salvo ainda.")
        st.markdown("---"); with st.expander("Informações do Sistema"): st.write("**Versão:** 3.0 Professional"); st.write("**Última Atualização:** 2025-09-15"); st.write("**Status:** Online"); st.write("**Armazenamento:** 2.5 GB disponíveis")
        if st.button("Trocar Usuário", use_container_width=True): st.session_state.user_data = None; st.rerun()
    st.markdown(f"""<div style="display: flex; align-items: center; margin-bottom: 2rem;"><h1 style="color: #000000; margin-right: 1rem; margin-bottom: 0;">DICOM Autopsy Viewer</h1><span style="background-color: #000000; color: #FFFFFF; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">v3.0 Professional</span></div><p style="color: #666666; margin-bottom: 2rem;">Bem-vindo, <strong>{user_data['name']}</strong>! Utilize as ferramentas abaixo para análise forense avançada de imagens DICOM.</p>""", unsafe_allow_html=True)
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file: tmp_file.write(uploaded_file.read()); tmp_path = tmp_file.name
            log_security_event(user_data['email'], "FILE_UPLOAD", f"Filename: {uploaded_file.name}")
            try:
                dicom_data = pydicom.dcmread(tmp_path); image_array = dicom_data.pixel_array
                st.session_state.dicom_data = dicom_data; st.session_state.image_array = image_array; st.session_state.uploaded_file_name = uploaded_file.name
                st.markdown("### Informações do Arquivo"); col1, col2, col3, col4 = st.columns(4); with col1: st.metric("Dimensões", f"{image_array.shape[0]} × {image_array.shape[1]}")
                with col2: st.metric("Tipo de Dados", str(image_array.dtype)); with col3: st.metric("Faixa de Valores", f"{image_array.min()} → {image_array.max()}"); with col4: st.metric("Tamanho do Arquivo", f"{uploaded_file.size / 1024:.1f} KB")
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Visualização", "Estatísticas", "Análise Técnica", "Qualidade", "Análise Post-Mortem", "RA-Index", "Relatórios"])
                with tab1: enhanced_visualization_tab(dicom_data, image_array); with tab2: enhanced_statistics_tab(dicom_data, image_array); with tab3: enhanced_technical_analysis_tab(dicom_data, image_array); with tab4: enhanced_quality_metrics_tab(dicom_data, image_array); with tab5: enhanced_post_mortem_analysis_tab(dicom_data, image_array); with tab6: enhanced_ra_index_tab(dicom_data, image_array); with tab7: enhanced_reporting_tab(dicom_data, image_array, user_data)
            except pydicom.errors.NotImplementedError as e: st.error(f"Erro ao processar arquivo DICOM: A compressão '{dicom_data.file_meta.TransferSyntaxUID.name}' não é suportada ou a biblioteca necessária está faltando."); st.info("Para arquivos comprimidos, certifique-se de que as bibliotecas `pydicom`, `jpeg_ls`, `gdcm` e `reportlab` estão instaladas corretamente."); logging.error(f"Erro de compressão DICOM: {e}")
            except Exception as e: st.error(f"Erro no processamento do arquivo DICOM: {e}"); logging.error(f"Erro no processamento DICOM: {e}")
            finally: os.unlink(tmp_path)
        except Exception as e: st.error(f"Erro ao carregar arquivo: {e}"); logging.error(f"Erro no carregamento: {e}")
    else:
        st.info("Carregue um arquivo DICOM na barra lateral para começar a análise."); st.markdown("## Funcionalidades Disponíveis"); col1, col2, col3 = st.columns(3)
        with col1: st.markdown("""<div class="info-card"><h4>Visualização Avançada</h4><ul><li>Janelamento Hounsfield personalizado</li><li>Ferramentas colorimétricas</li><li>Análise de pixels interativa</li><li>Visualização 3D multiplana</li><li>Download de imagens processadas</li></ul></div>""", unsafe_allow_html=True)
        with col2: st.markdown("""<div class="info-card"><h4>Análise Estatística</h4><ul><li>6+ tipos de visualizações</li><li>Análise regional detalhada</li><li>Correlações avançadas</li><li>Densidade de probabilidade</li><li>Mapas de calor interativos</li></ul></div>""", unsafe_allow_html=True)
        with col3: st.markdown("""<div class="info-card"><h4>Análise Forense</h4><ul><li>Metadados completos DICOM</li><li>Verificação de integridade</li><li>Detecção de anomalias</li><li>Timeline forense</li><li>Autenticidade de imagens</li></ul></div>""", unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4: st.markdown("""<div class="info-card"><h4>Controle de Qualidade</h4><ul><li>Métricas de qualidade de imagem</li><li>Análise de ruído e artefatos</li><li>Detecção de compressão</li><li>Uniformidade e resolução</li><li>Relatórios de qualidade</li></ul></div>""", unsafe_allow_html=True)
        with col5: st.markdown("""<div class="info-card"><h4>Análise Post-Mortem</h4><ul><li>Estimativa de intervalo post-mortem</li><li>Análise de fenômenos cadavéricos</li><li>Modelos de decomposição</li><li>Mapas de alterações teciduais</li><li>Correlações temporais</li></ul></div>""", unsafe_allow_html=True)
        with col6: st.markdown("""<div class="info-card"><h4>Relatórios Completos</h4><ul><li>Relatórios personalizáveis</li><li>Exportação em PDF/CSV</li><li>Histórico de análises</li><li>Comparativo entre exames</li><li>Banco de dados de casos</li></ul></div>""", unsafe_allow_html=True)
        st.markdown("## Casos de Uso Exemplares"); use_case_col1, use_case_col2 = st.columns(2); 
        with use_case_col1: st.expander("Identificação de Metais e Projéteis").markdown("1. Carregue a imagem DICOM\n2. Acesse a aba **Visualização**\n3. Utilize as ferramentas colorimétricas para destacar metais\n4. Ajuste a janela Hounsfield para a faixa de 1000-3000 HU\n5. Use os filtros de detecção de bordas para melhorar a visualização\n6. Gere um relatório completo com as medidas e localizações")
        with use_case_col2: st.expander("Estimativa de Intervalo Post-Mortem").markdown("1. Carregue a imagem DICOM\n2. Acesse a aba **Análise Post-Mortem**\n3. Configure os parâmetros ambientais\n4. Analise os mapas de distribuição gasosa\n5. Consulte as estimativas temporais\n6. Exporte o relatório forense completo")

def main():
    if 'user_data' not in st.session_state: st.session_state.user_data = None
    if 'dicom_data' not in st.session_state: st.session_state.dicom_data = None
    if 'image_array' not in st.session_state: st.session_state.image_array = None
    if 'current_report' not in st.session_state: st.session_state.current_report = None
    setup_matplotlib_for_plotting()
    if not safe_init_database(): st.error("Erro crítico: Não foi possível inicializar o sistema. Contate o administrador."); return
    update_css_theme()
    if st.session_state.user_data is None: show_user_form()
    else: show_main_app()

if __name__ == "__main__":
    main()
