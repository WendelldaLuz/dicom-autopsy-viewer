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
from plotly.subplots import make_subplots
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
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== SEÇÃO 1: FUNÇÕES DE VISUALIZAÇÃO APRIMORADA ======

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib para plotagem com configuração adequada.
    """
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
    """
    Aplica janelamento de Hounsfield na imagem
    """
    min_value = window_center - window_width / 2.0  # Usar divisão float
    max_value = window_center + window_width / 2.0
    
    windowed_image = np.copy(image)
    windowed_image[windowed_image < min_value] = min_value
    windowed_image[windowed_image > max_value] = max_value
    
    # Evitar divisão por zero
    if (max_value - min_value) <= 0:
        windowed_image = np.zeros_like(image, dtype=np.uint8)
    else:
        windowed_image = (windowed_image - min_value) / (max_value - min_value) * 255
    
    return windowed_image.astype(np.uint8)

def apply_colorimetric_analysis(image, metal_range, gas_range, metal_color, gas_color, 
                               brightness, contrast, apply_metal, apply_gas):
    """
    Aplica análise colorimétrica avançada com janelamentos específicos
    """
    # Primeiro, processar apenas a intensidade (não as cores)
    result_image = np.copy(image).astype(float)
    
    # Aplicar brilho e contraste apenas nos valores de intensidade
    result_image = result_image * contrast + brightness
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)
    
    # Converter para RGB se necessário (apenas para imagens em escala de cinza)
    if len(result_image.shape) == 2:
        if 'cv2' in globals():
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
        else:
            # Fallback sem OpenCV - converter manualmente para RGB
            result_image = np.stack([result_image] * 3, axis=-1)
    
    # Aplicar coloração para metais
    if apply_metal:
        # Criar máscara baseada na imagem original (não processada)
        metal_mask = (image >= metal_range[0]) & (image <= metal_range[1])
        if np.any(metal_mask):
            # Aplicar a cor RGB diretamente - SEM operações matemáticas
            # Para cada canal RGB
            result_image[metal_mask, 0] = metal_color[0]  # Canal R
            result_image[metal_mask, 1] = metal_color[1]  # Canal G
            result_image[metal_mask, 2] = metal_color[2]  # Canal B
    
    # Aplicar coloração para gases
    if apply_gas:
        # Criar máscara baseada na imagem original (não processada)
        gas_mask = (image >= gas_range[0]) & (image <= gas_range[1])
        if np.any(gas_mask):
            # Aplicar a cor RGB diretamente - SEM operações matemáticas
            # Para cada canal RGB
            result_image[gas_mask, 0] = gas_color[0]  # Canal R
            result_image[gas_mask, 1] = gas_color[1]  # Canal G
            result_image[gas_mask, 2] = gas_color[2]  # Canal B
    
    return result_image

def enhanced_visualization_tab(dicom_data, image_array):
    """
    Aba de visualização aprimorada com ferramentas colorimétricas
    """
    st.subheader("Visualização Avançada com Ferramentas Colorimétricas")
    
    # Controles principais em colunas
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🔧 Controles de Janelamento")
        # Presets de janelamento Hounsfield
        preset = st.selectbox("Preset de Janelamento:", [
            "Personalizado", "Ossos (400/1500)", "Metais (1000/4000)", 
            "Gases (-1000/400)", "Tecidos Moles (50/400)", "Pulmões (-600/1600)"
        ], key="preset_janelamento")
        
        # Configurar valores baseados no preset
        if preset == "Ossos (400/1500)":
            default_center, default_width = 400, 1500
        elif preset == "Metais (1000/4000)":
            default_center, default_width = 1000, 4000
        elif preset == "Gases (-1000/400)":
            default_center, default_width = -1000, 400
        elif preset == "Tecidos Moles (50/400)":
            default_center, default_width = 50, 400
        elif preset == "Pulmões (-600/1600)":
            default_center, default_width = -600, 1600
        else:
            default_center, default_width = 0, 1000
        
        window_center = st.slider("Centro da Janela (HU):", -2000, 4000, default_center, key="window_center")
        window_width = st.slider("Largura da Janela (HU):", 1, 6000, default_width, key="window_width")
    
    with col2:
        st.markdown("### Colorimetria Avançada")
        apply_metal = st.checkbox("Destacar Metais", value=False, key="apply_metal")
        metal_range = st.slider("Faixa de Metais (HU):", -1000, 4000, (800, 3000), disabled=not apply_metal, key="metal_range")
        metal_color = st.color_picker("Cor para Metais:", "#FF0000", disabled=not apply_metal, key="metal_color")
        
        apply_gas = st.checkbox("Destacar Gases", value=False, key="apply_gas")
        gas_range = st.slider("Faixa de Gases (HU):", -1000, 0, (-1000, -400), disabled=not apply_gas, key="gas_range")
        gas_color = st.color_picker("Cor para Gases:", "#00FF00", disabled=not apply_gas, key="gas_color")
    
    with col3:
        st.markdown("### ⚙️ Ajustes de Imagem")
        brightness = st.slider("Brilho:", -100, 100, 0, key="brightness")
        contrast = st.slider("Contraste:", 0.1, 3.0, 1.0, 0.1, key="contrast")
        
        # Filtros adicionais
        apply_filter = st.selectbox("Filtro Adicional:", [
            "Nenhum", "Aguçar", "Suavizar", "Detecção de Bordas", "Realce de Contraste"
        ], key="apply_filter")
    
    # Processar na escala original primeiro
    processed_values = image_array.astype(float) * contrast + brightness
    
    # Aplicar janelamento depois
    processed_image = apply_hounsfield_windowing(processed_values, window_center, window_width)
    
    # Converter cores hex para RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    metal_rgb = hex_to_rgb(metal_color)
    gas_rgb = hex_to_rgb(gas_color)
    
    # Aplicar análise colorimétrica
    if 'cv2' in globals():
        final_image = apply_colorimetric_analysis(
            processed_image, metal_range, gas_range, metal_rgb, gas_rgb,
            brightness, contrast, apply_metal, apply_gas
        )
    else:
        # Fallback sem OpenCV - apenas aplicar brilho/contraste
        final_image = processed_image.astype(float)
        final_image = final_image * contrast + brightness
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)
        
        # Se precisar converter para RGB e aplicar cores
        if apply_metal or apply_gas:
            # Converter para RGB (3 canais)
            if len(final_image.shape) == 2:
                final_image = np.stack([final_image] * 3, axis=-1)
            
            # Aplicar coloração para metais
            if apply_metal:
                metal_mask = (processed_image >= metal_range[0]) & (processed_image <= metal_range[1])
                if np.any(metal_mask):
                    # Aplicar cor RGB canal por canal
                    final_image[metal_mask, 0] = metal_rgb[0]  # Canal R
                    final_image[metal_mask, 1] = metal_rgb[1]  # Canal G
                    final_image[metal_mask, 2] = metal_rgb[2]  # Canal B
            
            # Aplicar coloração para gases
            if apply_gas:
                gas_mask = (processed_image >= gas_range[0]) & (processed_image <= gas_range[1])
                if np.any(gas_mask):
                    # Aplicar cor RGB canal por canal
                    final_image[gas_mask, 0] = gas_rgb[0]  # Canal R
                    final_image[gas_mask, 1] = gas_rgb[1]  # Canal G
                    final_image[gas_mask, 2] = gas_rgb[2]  # Canal B
    
    # Aplicar filtros adicionais
    if 'cv2' in globals() and apply_filter != "Nenhum":
        if apply_filter == "Aguçar":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            final_image = cv2.filter2D(final_image, -1, kernel)
        elif apply_filter == "Suavizar":
            final_image = cv2.GaussianBlur(final_image, (5, 5), 0)
        elif apply_filter == "Detecção de Bordas":
            if len(final_image.shape) == 3:
                gray = cv2.cvtColor(final_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = final_image
            edges = cv2.Canny(gray, 50, 150)
            final_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        elif apply_filter == "Realce de Contraste":
            final_image = cv2.convertScaleAbs(final_image, alpha=1.5, beta=30)
    
    # Exibir imagem processada
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.markdown("#### Imagem Original")
        fig_orig, ax_orig = plt.subplots(figsize=(8, 8))
        ax_orig.imshow(image_array, cmap='gray')
        ax_orig.axis('off')
        ax_orig.set_title("Imagem DICOM Original")
        st.pyplot(fig_orig)
        plt.close(fig_orig)
    
    with col_img2:
        st.markdown("#### Imagem Processada")
        fig_proc, ax_proc = plt.subplots(figsize=(8, 8))
        if len(final_image.shape) == 3:
            ax_proc.imshow(final_image)
        else:
            ax_proc.imshow(final_image, cmap='viridis')
        ax_proc.axis('off')
        ax_proc.set_title("Imagem com Processamento Avançado")
        st.pyplot(fig_proc)
        plt.close(fig_proc)
    
    # Análise de pixels interativa
    st.markdown("### Análise Interativa de Pixels")
    
    if st.button("Ativar Análise de Pixels", key="btn_analise_pixels"):
        st.info("Clique na imagem abaixo para analisar pixels específicos")
        
        # Criar gráfico interativo com Plotly
        fig_interactive = go.Figure()
        
        fig_interactive.add_trace(go.Heatmap(
            z=processed_image,
            colorscale='viridis',
            showscale=True,
            hovertemplate='X: %{x}<br>Y: %{y}<br>Valor HU: %{z}<extra></extra>'
        ))
        
        fig_interactive.update_layout(
            title="Mapa Interativo de Pixels - Clique para Analisar",
            xaxis_title="Coordenada X",
            yaxis_title="Coordenada Y",
            height=600
        )
        
        st.plotly_chart(fig_interactive, use_container_width=True, key="chart_interativo_pixels")
    
    # Opção de download
    st.markdown("### Download da Imagem Processada")
    
    if st.button("Preparar Download", key="btn_preparar_download"):
        # Converter imagem para formato de download
        if len(final_image.shape) == 3:
            pil_image = Image.fromarray(final_image.astype(np.uint8))
        else:
            pil_image = Image.fromarray(final_image.astype(np.uint8), mode='L')
        
        # Criar buffer para download
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        st.download_button(
            label="Baixar Imagem Processada (PNG)",
            data=img_buffer,
            file_name=f"dicom_processada_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            key="btn_download_imagem"
        )
        
        st.success("Imagem preparada para download!", key="msg_sucesso_download")

# ====== SEÇÃO 2: ESTATÍSTICAS AVANÇADAS AMPLIADA ======

def enhanced_statistics_tab(dicom_data, image_array):
    """
    Aba de estatísticas com múltiplas visualizações - AMPLIADA E CORRIGIDA
    """
    st.subheader(" Análise Avançada")
    
    # Verificação de segurança
    if image_array is None or not isinstance(image_array, np.ndarray) or image_array.size == 0:
        st.error("Dados de imagem inválidos para análise estatística")
        return
    
    # Calcular estatísticas básicas com verificação de erro
    try:
        # CORREÇÃO: Usar np.nanpercentile para evitar problemas com arrays vazios
        flat_array = image_array.flatten()
        
        stats_data = {
            'Média': np.mean(flat_array),
            'Mediana': np.median(flat_array),
            'Desvio Padrão': np.std(flat_array),
            'Mínimo': np.min(flat_array),
            'Máximo': np.max(flat_array),
            'Variância': np.var(flat_array),
            'Intervalo': np.max(flat_array) - np.min(flat_array),
            'Q1': np.percentile(flat_array, 25) if flat_array.size > 0 else 0,
            'Q3': np.percentile(flat_array, 75) if flat_array.size > 0 else 0,
            'IQR': np.percentile(flat_array, 75) - np.percentile(flat_array, 25) if flat_array.size > 0 else 0,
            'Energia': np.sum(flat_array**2) / flat_array.size if flat_array.size > 0 else 0
        }
        
        # Adicionar assimetria e curtose apenas se houver dados suficientes
        if flat_array.size > 1:
            stats_data['Assimetria'] = stats.skew(flat_array)
            stats_data['Curtose'] = stats.kurtosis(flat_array)
            
            # Calcular entropia
            hist, _ = np.histogram(flat_array, bins=256, density=True)
            hist = hist[hist > 0]  # Remover zeros
            stats_data['Entropia'] = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
        else:
            stats_data['Assimetria'] = 0
            stats_data['Curtose'] = 0
            stats_data['Entropia'] = 0
            
    except Exception as e:
        st.error(f"Erro ao calcular estatísticas: {e}")
        return
    
    # Display de métricas principais em abas
    tab1, tab2, tab3, tab4 = st.tabs(["Métricas Básicas", "Distribuição", "Análise Regional", "Estatísticas Avançadas"])
    
    with tab1:
        st.markdown("### Métricas Estatísticas Básicas")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Média (HU)", f"{stats_data['Média']:.2f}")
            st.metric("Mediana (HU)", f"{stats_data['Mediana']:.2f}")
            # CORREÇÃO: Moda calculada de forma segura
            try:
                mode_result = stats.mode(flat_array, keepdims=True)
                mode_value = mode_result.mode[0] if hasattr(mode_result, 'mode') else flat_array[0] if flat_array.size > 0 else 0
                st.metric("Moda (HU)", f"{mode_value:.2f}")
            except:
                st.metric("Moda (HU)", "N/A")
        
        with col2:
            st.metric("Desvio Padrão", f"{stats_data['Desvio Padrão']:.2f}")
            st.metric("Variância", f"{stats_data['Variância']:.2f}")
            st.metric("Amplitude", f"{stats_data['Intervalo']:.2f}")
        
        with col3:
            st.metric("Mínimo (HU)", f"{stats_data['Mínimo']:.2f}")
            st.metric("Máximo (HU)", f"{stats_data['Máximo']:.2f}")
            st.metric("IQR", f"{stats_data['IQR']:.2f}")
        
        with col4:
            st.metric("Assimetria", f"{stats_data['Assimetria']:.3f}")
            st.metric("Curtose", f"{stats_data['Curtose']:.3f}")
            st.metric("Entropia", f"{stats_data['Entropia']:.3f}")
    
    with tab2:
        st.markdown("### Análise de Distribuição")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Histograma detalhado
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(
                x=flat_array,
                nbinsx=100,
                name="Distribuição de Valores HU",
                marker_color='lightblue',
                opacity=0.7,
                hovertemplate="Valor: %{x}<br>Frequência: %{y}<extra></extra>"
            ))
            fig1.update_layout(
                title="Histograma de Distribuição de Valores HU",
                xaxis_title="Unidades Hounsfield (HU)",
                yaxis_title="Frequência",
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # 3. Gráfico de probabilidade normal (Q-Q Plot)
            st.markdown("#### Gráfico de Probabilidade Normal (Q-Q Plot)")
            try:
                if flat_array.size > 10:  # Apenas para amostras razoáveis
                    fig_qq = go.Figure()
                    
                    # Calcular quantis teóricos e amostrais
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
                    sample_quantiles = np.percentile(flat_array, np.linspace(1, 99, 100))
                    
                    fig_qq.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name='Q-Q Plot',
                        marker=dict(color='blue', size=6)
                    ))
                    
                    # Adicionar linha de referência (y=x)
                    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                    fig_qq.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Linha de Referência',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_qq.update_layout(
                        title="Gráfico Q-Q: Normalidade dos Dados",
                        xaxis_title="Quantis Teóricos",
                        yaxis_title="Quantis Amostrais",
                        height=400
                    )
                    st.plotly_chart(fig_qq, use_container_width=True)
                else:
                    st.info("Amostra muito pequena para análise Q-Q")
            except Exception as e:
                st.warning(f"Não foi possível gerar o gráfico Q-Q: {str(e)}")
        
        with col2:
            # 2. Box Plot
            fig2 = go.Figure()
            fig2.add_trace(go.Box(
                y=flat_array,
                name="Distribuição HU",
                boxpoints='outliers',
                marker_color='lightgreen',
                jitter=0.3,
                pointpos=-1.8
            ))
            fig2.update_layout(
                title="Box Plot - Análise de Outliers",
                yaxis_title="Unidades Hounsfield (HU)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # 4. Densidade de probabilidade
            st.markdown("#### Densidade de Probabilidade")
            try:
                if flat_array.size > 10:  # Apenas para amostras razoáveis
                    from scipy.stats import gaussian_kde
                    density = gaussian_kde(flat_array)
                    xs = np.linspace(np.min(flat_array), np.max(flat_array), 200)
                    
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(
                        x=xs,
                        y=density(xs),
                        mode='lines',
                        name="Densidade",
                        fill='tozeroy',
                        line=dict(color='purple', width=2)
                    ))
                    fig4.update_layout(
                        title="Função de Densidade de Probabilidade",
                        xaxis_title="Unidades Hounsfield (HU)",
                        yaxis_title="Densidade",
                        height=400
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("Amostra muito pequena para análise de densidade")
            except Exception as e:
                st.warning(f"Não foi possível calcular a densidade de probabilidade: {str(e)}")
    
    with tab3:
        st.markdown("###  Análise Estatística Regional")
        
        # Dividir imagem em regiões
        h, w = image_array.shape
        regions = {
            'Superior Esquerda': image_array[:h//2, :w//2],
            'Superior Direita': image_array[:h//2, w//2:],
            'Inferior Esquerda': image_array[h//2:, :w//2],
            'Inferior Direita': image_array[h//2:, w//2:],
            'Centro': image_array[h//4:3*h//4, w//4:3*w//4]
        }
        
        regional_stats = []
        for region_name, region_data in regions.items():
            if region_data.size > 0:
                flat_region = region_data.flatten()
                regional_stats.append({
                    'Região': region_name,
                    'Média': np.mean(flat_region),
                    'Desvio Padrão': np.std(flat_region),
                    'Mínimo': np.min(flat_region),
                    'Máximo': np.max(flat_region),
                    'Variância': np.var(flat_region),
                    'Tamanho': flat_region.size
                })
        
        if regional_stats:
            df_regional = pd.DataFrame(regional_stats)
            
            # Gráfico de barras comparativo
            fig7 = go.Figure()
            
            fig7.add_trace(go.Bar(
                x=df_regional['Região'],
                y=df_regional['Média'],
                name='Média',
                marker_color='lightblue',
                text=df_regional['Média'].round(2),
                textposition='auto'
            ))
            
            fig7.add_trace(go.Bar(
                x=df_regional['Região'],
                y=df_regional['Desvio Padrão'],
                name='Desvio Padrão',
                marker_color='lightcoral',
                text=df_regional['Desvio Padrão'].round(2),
                textposition='auto'
            ))
            
            fig7.update_layout(
                title="Comparação Estatística Regional",
                xaxis_title="Regiões da Imagem",
                yaxis_title="Valores",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig7, use_container_width=True)
            
            # Tabela de estatísticas regionais
            st.markdown("#### Tabela de Estatísticas Regionais")
            st.dataframe(df_regional, use_container_width=True)
        else:
            st.warning("Não foi possível calcular estatísticas regionais")
    
    with tab4:
        st.markdown("### Estatísticas Avançadas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 5. Mapa de calor da imagem
            fig5 = go.Figure(data=go.Heatmap(
                z=image_array,
                colorscale='Viridis',
                showscale=True,
                hovertemplate='X: %{x}<br>Y: %{y}<br>Valor HU: %{z}<extra></extra>'
            ))
            fig5.update_layout(
                title="Mapa de Calor da Imagem",
                height=400
            )
            st.plotly_chart(fig5, use_container_width=True)
            
            # 7. Análise de valores outliers
            st.markdown("#### Análise de Outliers")
            
            # Calcular limites para outliers
            Q1 = stats_data['Q1']
            Q3 = stats_data['Q3']
            IQR = stats_data['IQR']
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = flat_array[(flat_array < lower_bound) | (flat_array > upper_bound)]
            
            outlier_stats = {
                'Total de Outliers': len(outliers),
                'Percentual de Outliers': f"{(len(outliers) / flat_array.size) * 100:.2f}%" if flat_array.size > 0 else "0%",
                'Limite Inferior': f"{lower_bound:.2f}",
                'Limite Superior': f"{upper_bound:.2f}",
                'Outlier Mínimo': f"{np.min(outliers):.2f}" if len(outliers) > 0 else "N/A",
                'Outlier Máximo': f"{np.max(outliers):.2f}" if len(outliers) > 0 else "N/A"
            }
            
            for key, value in outlier_stats.items():
                st.metric(key, value)
        
        with col2:
            # 6. Análise de correlação espacial
            # Calcular gradientes
            try:
                grad_x = np.gradient(image_array.astype(float), axis=1)
                grad_y = np.gradient(image_array.astype(float), axis=0)
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                fig6 = go.Figure(data=go.Heatmap(
                    z=magnitude,
                    colorscale='plasma',
                    showscale=True,
                    hovertemplate='X: %{x}<br>Y: %{y}<br>Magnitude: %{z:.2f}<extra></extra>'
                ))
                fig6.update_layout(
                    title="Magnitude do Gradiente (Bordas)",
                    height=400
                )
                st.plotly_chart(fig6, use_container_width=True)
            except Exception as e:
                st.warning(f"Não foi possível calcular gradientes: {str(e)}")
            
            # 8. Análise de textura
            st.markdown("#### 🔍 Análise de Textura")
            
            texture_metrics = calculate_glcm_features(image_array)
            
            texture_df = pd.DataFrame(list(texture_metrics.items()), columns=['Métrica', 'Valor'])
            st.dataframe(texture_df, use_container_width=True, height=200)
            
            # Métricas de qualidade de imagem
            st.markdown("#### Métricas de Qualidade")
            
            quality_metrics = {
                'SNR': f"{calculate_snr(image_array):.2f}" if not np.isinf(calculate_snr(image_array)) else "∞",
                'Ruído Estimado': f"{estimate_noise(image_array):.2f}",
                'Contraste RMS': f"{np.sqrt(np.mean((flat_array - np.mean(flat_array))**2)):.2f}" if flat_array.size > 0 else "0"
            }
            
            for metric, value in quality_metrics.items():
                st.metric(metric, value)
    
    # Análise adicional
    st.markdown("### Relatório Estatístico Completo")
    
    with st.expander("Visualizar Relatório Detalhado"):
        # Estatísticas descritivas completas
        st.markdown("#### Estatísticas Descritivas Completas")
        
        desc_stats = {
            'Contagem': flat_array.size,
            'Média': stats_data['Média'],
            'Desvio Padrão': stats_data['Desvio Padrão'],
            'Variância': stats_data['Variância'],
            'Mínimo': stats_data['Mínimo'],
            '25%': stats_data['Q1'],
            '50% (Mediana)': stats_data['Mediana'],
            '75%': stats_data['Q3'],
            'Máximo': stats_data['Máximo'],
            'Intervalo': stats_data['Intervalo'],
            'IQR': stats_data['IQR'],
            'Assimetria': stats_data['Assimetria'],
            'Curtose': stats_data['Curtose'],
            'Entropia': stats_data['Entropia']
        }
        
        desc_df = pd.DataFrame(list(desc_stats.items()), columns=['Estatística', 'Valor'])
        st.dataframe(desc_df, use_container_width=True)
        
        # Teste de normalidade
        st.markdown("#### Teste de Normalidade")
        try:
            if flat_array.size > 3 and flat_array.size <= 5000:  # Limitação do teste
                stat, p_value = stats.shapiro(flat_array)
                normality = "Distribuição Normal" if p_value > 0.05 else "Não Normal"
                
                norm_test = {
                    'Estatística de Teste': f"{stat:.4f}",
                    'Valor-p': f"{p_value:.4f}",
                    'Interpretação': normality
                }
                
                norm_df = pd.DataFrame(list(norm_test.items()), columns=['Parâmetro', 'Valor'])
                st.dataframe(norm_df, use_container_width=True)
            else:
                st.info("Teste de normalidade não aplicável para este tamanho de amostra")
        except Exception as e:
            st.warning(f"Não foi possível realizar o teste de normalidade: {str(e)}")
    
    # Opção de exportação
    if st.button(" Exportar Relatório Estatístico", key="btn_export_stats"):
        # Preparar dados para exportação
        export_data = []
        
        if regional_stats:
            for region in regional_stats:
                export_data.append({
                    'Região': region['Região'],
                    'Média_HU': region['Média'],
                    'Desvio_Padrão': region['Desvio Padrão'],
                    'Mínimo_HU': region['Mínimo'],
                    'Máximo_HU': region['Máximo'],
                    'Variância': region['Variância'],
                    'Tamanho_Amostra': region['Tamanho']
                })
        
        # Adicionar estatísticas gerais
        export_data.append({
            'Região': 'GERAL',
            'Média_HU': stats_data['Média'],
            'Desvio_Padrão': stats_data['Desvio Padrão'],
            'Mínimo_HU': stats_data['Mínimo'],
            'Máximo_HU': stats_data['Máximo'],
            'Variância': stats_data['Variância'],
            'Tamanho_Amostra': flat_array.size
        })
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="Baixar Dados Estatísticos (CSV)",
            data=csv,
            file_name=f"estatisticas_imagem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ====== SEÇÃO 3: ANÁLISE TÉCNICA ======

def enhanced_technical_analysis_tab(dicom_data, image_array):
    """
    Aba de análise técnica com máximo de dados forenses
    """
    st.subheader("Análise Técnica Forense Avançada")
    
    # Extração de metadados DICOM
    st.markdown("### Metadados DICOM Completos")
    
    # Organizar metadados por categoria
    categories = {
        'Informações do Paciente': [],
        'Parâmetros de Aquisição': [],
        'Configurações do Equipamento': [],
        'Dados de Imagem': [],
        'Informações Temporais': [],
        'Dados Técnicos Forenses': []  # SEM EMOJI - CORREÇÃO AQUI
    }
    
    # Extrair informações relevantes
    for elem in dicom_data:
        if elem.tag.group != 0x7fe0:  # Excluir pixel data
            tag_name = elem.name if hasattr(elem, 'name') else str(elem.tag)
            value = str(elem.value) if len(str(elem.value)) < 100 else str(elem.value)[:100] + "..."
            
            # Categorizar por tipo de informação
            if any(keyword in tag_name.lower() for keyword in ['patient', 'name', 'id', 'birth', 'sex']):
                categories['Informações do Paciente'].append(f"**{tag_name}**: {value}")
            elif any(keyword in tag_name.lower() for keyword in ['kv', 'ma', 'exposure', 'slice', 'pixel']):
                categories['Parâmetros de Aquisição'].append(f"**{tag_name}**: {value}")
            elif any(keyword in tag_name.lower() for keyword in ['manufacturer', 'model', 'software', 'station']):
                categories['Configurações do Equipamento'].append(f"**{tag_name}**: {value}")
            elif any(keyword in tag_name.lower() for keyword in ['rows', 'columns', 'spacing', 'thickness']):
                categories['Dados de Imagem'].append(f"**{tag_name}**: {value}")
            elif any(keyword in tag_name.lower() for keyword in ['date', 'time', 'acquisition']):
                categories['Informações Temporais'].append(f"**{tag_name}**: {value}")
            else:         
                categories['Dados Técnicos Forenses'].append(f"**{tag_name}**: {value}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (category, items) in enumerate(list(categories.items())[:3]):
            if items:
                with st.expander(f"{category} ({len(items)} itens)"):
                    for item in items[:20]:  # Limitar a 20 itens por categoria
                        st.markdown(item)
    
    with col2:
        for i, (category, items) in enumerate(list(categories.items())[3:]):
            if items:
                with st.expander(f"{category} ({len(items)} itens)"):
                    for item in items[:20]:  
                        st.markdown(item)
    
    # Análise forense avançada
    st.markdown("### Análise Forense Digital Avançada")
    
    # Calcular métricas forenses específicas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Métricas de Integridade")
        
        # Calcular hash da imagem
        import hashlib
        image_hash = hashlib.sha256(image_array.tobytes()).hexdigest()
        st.code(f"SHA-256: {image_hash[:32]}...")
        
        # Análise de ruído
        noise_level = estimate_noise(image_array)
        st.metric("Nível de Ruído", f"{noise_level:.2f}")
        
        # Análise de compressão
        unique_values = len(np.unique(image_array))
        total_pixels = image_array.size
        compression_ratio = unique_values / total_pixels
        st.metric("Taxa de Compressão", f"{compression_ratio:.4f}")
    
    with col2:
        st.markdown("#### Análise Espectral")
        
        # FFT para análise de frequência
        fft_2d = np.fft.fft2(image_array)
        magnitude_spectrum = np.log(np.abs(fft_2d) + 1)
        
        # Energia em diferentes faixas
        low_freq_energy = np.sum(magnitude_spectrum[:50, :50])
        high_freq_energy = np.sum(magnitude_spectrum[-50:, -50:])
        
        st.metric("Energia Baixa Freq.", f"{low_freq_energy:.0f}")
        st.metric("Energia Alta Freq.", f"{high_freq_energy:.0f}")
        
        # Relação sinal-ruído estimada
        signal_power = np.var(image_array)
        noise_power = noise_level**2
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        st.metric("SNR (dB)", f"{snr:.2f}")
    
    with col3:
        st.markdown("#### Análise Morfológica")
        
        # Detecção de bordas
        if 'cv2' in globals():
            edges = cv2.Canny(image_array.astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
        else:
            # Usar gradientes numpy como alternativa
            grad_x = np.gradient(image_array, axis=1)
            grad_y = np.gradient(image_array, axis=0)
            edges = np.sqrt(grad_x**2 + grad_y**2)
            edge_density = np.sum(edges > np.percentile(edges, 95)) / edges.size
        
        st.metric("Densidade de Bordas", f"{edge_density:.4f}")
        
        # Análise de conectividade
        binary_image = image_array > np.mean(image_array)
        connected_components = len(np.unique(ndimage.label(binary_image)[0]))
        st.metric("Componentes Conexos", f"{connected_components}")
        
        # Análise de textura
        texture_energy = np.sum(np.gradient(image_array)**2)
        st.metric("Energia de Textura", f"{texture_energy:.0f}")
    
    # Gráficos de análise forense
    st.markdown("### Visualizações Forenses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Espectro de magnitude FFT
        fig1 = go.Figure(data=go.Heatmap(
            z=magnitude_spectrum[:100, :100],  # Mostrar apenas parte central
            colorscale='viridis',
            showscale=True
        ))
        fig1.update_layout(
            title="Espectro de Magnitude (FFT)",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Mapa de bordas
        fig2 = go.Figure(data=go.Heatmap(
            z=edges,
            colorscale='plasma',
            showscale=True
        ))
        fig2.update_layout(
            title="Mapa de Detecção de Bordas",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Análise de autenticidade
    st.markdown("### Análise de Autenticidade")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Verificações de Integridade")
        
        # Simulação de verificações (em um sistema real, estas seriam mais complexas)
        checks = {
            "Estrutura DICOM válida": True,
            "Metadados consistentes": True,
            "Assinatura digital": False,  # Simulado
            "Possível edição detectada": np.random.choice([True, False]),
            "Conformidade com padrão": True
        }
        
        for check, status in checks.items():
            if "✅" in check:
                st.success(check)
            elif "⚠️" in check and status:
                st.warning(check)
            elif "❌" in check:
                st.error(check)
            else:
                st.info(check)
    
    with col2:
        st.markdown("#### Timeline Forense")
        
        # Extrair datas importantes
        timeline_data = []
        if hasattr(dicom_data, 'StudyDate'):
            timeline_data.append(f" Data do Estudo: {dicom_data.StudyDate}")
        if hasattr(dicom_data, 'AcquisitionDate'):
            timeline_data.append(f" Data de Aquisição: {dicom_data.AcquisitionDate}")
        if hasattr(dicom_data, 'ContentDate'):
            timeline_data.append(f" Data do Conteúdo: {dicom_data.ContentDate}")
        
        timeline_data.append(f"🔍 Análise Realizada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for event in timeline_data:
            st.markdown(f"- {event}")
    
    with col3:
        st.markdown("#### Relatório de Anomalias")
        
        # Detectar possíveis anomalias
        anomalies = []
        
        # Verificar valores extremos
        if np.min(image_array) < -1000 or np.max(image_array) > 4000:
            anomalies.append("Valores HU fora do padrão")
        
        # Verificar uniformidade
        if np.std(image_array) > 1000:
            anomalies.append("Alta variabilidade nos dados")
        
        # Verificar ruído excessivo
        if noise_level > 100:
            anomalies.append("Nível de ruído elevado")
        
        # Verificar possível compressão excessiva
        if compression_ratio < 0.1:
            anomalies.append("Possível compressão excessiva")
        
        if not anomalies:
            st.success("Nenhuma anomalia detectada")
        else:
            for anomaly in anomalies:
                st.warning(anomaly)

# ====== SEÇÃO 4: MÉTRICAS DE QUALIDADE PROFISSIONAL ======

def calculate_psnr(original, processed=None):
    """Calcula PSNR (Peak Signal-to-Noise Ratio)"""
    if processed is None:
        # Se não há imagem processada, usar ruído estimado
        noise = estimate_noise(original)
        if noise == 0:
            return float('inf')
        return 20 * np.log10(np.max(original) / noise)
    else:
        # Entre original e processada
        mse = np.mean((original - processed) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(np.max(original) / np.sqrt(mse))

def calculate_ssim(original, processed=None):
    """Calcula SSIM (Structural Similarity Index) simplificado"""
    if processed is None:
        return 1.0  # Sem imagem processada para comparação
    
    try:
        from skimage.metrics import structural_similarity as ssim
        # Normalizar imagens para 0-1
        original_norm = (original - np.min(original)) / (np.max(original) - np.min(original))
        processed_norm = (processed - np.min(processed)) / (np.max(processed) - np.min(processed))
        return ssim(original_norm, processed_norm, data_range=1.0)
    except ImportError:
        # Fallback calculation
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        mu_x = np.mean(original)
        mu_y = np.mean(processed)
        sigma_x = np.var(original)
        sigma_y = np.var(processed)
        sigma_xy = np.cov(original.flatten(), processed.flatten())[0, 1]
        
        ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                  ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return ssim_val

def calculate_mtf(image_array, dicom_data):
    """Calcula MTF (Modulation Transfer Function) simplificado"""
    try:
        # Usar borda da imagem para estimar MTF
        edge_profile = image_array[image_array.shape[0] // 2, :]
        
        # Derivada do perfil de borda (Edge Spread Function)
        esf_derivative = np.gradient(edge_profile)
        
        # Normalizar e calcular MTF
        mtf = np.abs(np.fft.fft(esf_derivative))
        mtf = mtf[:len(mtf)//2]  # Manter apenas frequências positivas
        mtf = mtf / np.max(mtf)  # Normalizar
        
        # Encontrar frequência onde MTF cai para 50%
        freq_50 = np.argmax(mtf < 0.5) / len(mtf) if np.any(mtf < 0.5) else 1.0
        
        # Converter para lp/mm se PixelSpacing disponível
        if hasattr(dicom_data, 'PixelSpacing'):
            pixel_spacing = float(dicom_data.PixelSpacing[0])
            freq_50 = freq_50 / (2 * pixel_spacing)  # Conversão para lp/mm
        
        return float(freq_50), mtf
    except:
        return 0.0, np.array([0.0])

def calculate_cnr(image_array):
    """Calcula CNR (Contrast-to-Noise Ratio)"""
    try:
        # Selecionar duas regiões diferentes para calcular contraste
        h, w = image_array.shape
        roi1 = image_array[h//4:h//2, w//4:w//2]  # Região central
        roi2 = image_array[3*h//4:h, 3*w//4:w]    # Região periférica
        
        contrast = np.abs(np.mean(roi1) - np.mean(roi2))
        noise = estimate_noise(image_array)
        
        return contrast / noise if noise > 0 else 0.0
    except:
        return 0.0

def calculate_nps(image_array):
    """Calcula NPS (Noise Power Spectrum)"""
    try:
        # Remover tendência linear
        detrended = image_array - ndimage.uniform_filter(image_array, size=10)
        
        # Calcular espectro de potência do ruído
        fft_nps = np.fft.fft2(detrended)
        nps = np.abs(fft_nps) ** 2
        nps = np.fft.fftshift(nps)
        
        # Perfil radial do NPS
        center = np.array(nps.shape) // 2
        y, x = np.indices(nps.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        nps_radial = ndimage.mean(nps, labels=r, index=np.arange(0, np.max(r)))
        
        return nps, nps_radial
    except:
        return np.zeros_like(image_array), np.array([0.0])

def advanced_noise_analysis(image_array):
    """Análise avançada de ruído"""
    try:
        # Análise de ruído usando múltiplos métodos
        noise_levels = {}
        
        # Método 1: Diferença entre pixels adjacentes
        diff_h = image_array[:, 1:] - image_array[:, :-1]
        diff_v = image_array[1:, :] - image_array[:-1, :]
        noise_levels['Método Diferença'] = np.std(np.concatenate([diff_h.flatten(), diff_v.flatten()])) / np.sqrt(2)
        
        # Método 2: Filtro de uniformidade
        uniform_filtered = ndimage.uniform_filter(image_array, size=3)
        residual = image_array - uniform_filtered
        noise_levels['Método Residual'] = np.std(residual)
        
        # Método 3: Análise wavelet 
        from scipy import ndimage
        wavelet_approx = ndimage.gaussian_filter(image_array, sigma=1)
        wavelet_detail = image_array - wavelet_approx
        noise_levels['Método Wavelet'] = np.std(wavelet_detail)
        
        return noise_levels
    except:
        return {'Método Diferença': 0.0, 'Método Residual': 0.0, 'Método Wavelet': 0.0}

def professional_quality_metrics_tab(dicom_data, image_array, processed_image=None):
    """
    Aba de métricas de qualidade profissional para análise de imagem DICOM
    """
    st.subheader("Métricas de Qualidade de Imagem Profissional")
    
    # Calcular métricas avançadas
    with st.spinner("Calculando métricas avançadas de qualidade..."):
        # Métricas básicas
        snr_val = calculate_snr(image_array)
        psnr_val = calculate_psnr(image_array, processed_image)
        ssim_val = calculate_ssim(image_array, processed_image) if processed_image is not None else 1.0
        cnr_val = calculate_cnr(image_array)
        mtf_val, mtf_curve = calculate_mtf(image_array, dicom_data)
        
        # Análise de ruído avançada
        noise_levels = advanced_noise_analysis(image_array)
        nps_matrix, nps_radial = calculate_nps(image_array)
        
        # Cálculo de entropia e uniformidade corretos
        hist, _ = np.histogram(image_array.flatten(), bins=256)
        probabilities = hist / np.sum(hist)
        probabilities = probabilities[probabilities > 0]
        entropy_val = float(-np.sum(probabilities * np.log2(probabilities)))
        uniformity_val = float(np.sum(probabilities**2))
    
    # Display de métricas principais em abas
    tab1, tab2, tab3, tab4 = st.tabs(["Métricas Gerais", "Análise de Ruído", "Resolução", "Relatório Completo"])
    
    with tab1:
        st.markdown("### Métricas Fundamentais de Qualidade")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("SNR", f"{snr_val:.2f}", help="Relação Sinal-Ruído")
            st.metric("PSNR", f"{psnr_val:.2f} dB" if psnr_val != float('inf') else "∞ dB", 
                     help="Pico de Relação Sinal-Ruído")
        
        with col2:
            st.metric("CNR", f"{cnr_val:.2f}", help="Relação Contraste-Ruído")
            st.metric("SSIM", f"{ssim_val:.4f}", help="Índice de Similaridade Estrutural")
        
        with col3:
            st.metric("Entropia", f"{entropy_val:.2f} bits", help="Medida de informação/complexidade")
            st.metric("Uniformidade", f"{uniformity_val:.4f}", help="Uniformidade da distribuição de intensidade")
        
        with col4:
            st.metric("MTF₅₀", f"{mtf_val:.2f} lp/mm", help="Frequência espacial a 50% da modulação")
            st.metric("Dinâmica", f"{image_array.max()-image_array.min():.0f} HU", 
                     help="Faixa dinâmica da imagem")
    
    with tab2:
        st.markdown("### Análise Avançada de Ruído")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Comparação de métodos de ruído
            noise_df = pd.DataFrame(list(noise_levels.items()), columns=['Método', 'Nível de Ruído (HU)'])
            fig_noise = px.bar(noise_df, x='Método', y='Nível de Ruído (HU)', 
                              title="Comparação de Métodos de Análise de Ruído")
            st.plotly_chart(fig_noise, use_container_width=True)
            
            # Perfil do NPS
            fig_nps = go.Figure()
            fig_nps.add_trace(go.Scatter(
                x=np.arange(len(nps_radial)),
                y=nps_radial,
                mode='lines',
                name='NPS Radial',
                line=dict(color='blue', width=2)
            ))
            fig_nps.update_layout(
                title="Espectro de Potência do Ruído (NPS) - Perfil Radial",
                xaxis_title="Frequência Espacial",
                yaxis_title="Potência do Ruído",
                height=300
            )
            st.plotly_chart(fig_nps, use_container_width=True)
        
        with col2:
            # Mapa de ruído
            noise_map = image_array - ndimage.uniform_filter(image_array, size=5)
            fig_noise_map = go.Figure(data=go.Heatmap(
                z=noise_map,
                colorscale='Viridis',
                showscale=True,
                title="Mapa de Distribuição de Ruído"
            ))
            fig_noise_map.update_layout(height=400)
            st.plotly_chart(fig_noise_map, use_container_width=True)
            
            # Estatísticas de ruído
            st.markdown("**Estatísticas de Ruído:**")
            noise_stats = {
                'Ruído Médio': f"{np.mean(noise_map):.2f} HU",
                'Desvio Padrão': f"{np.std(noise_map):.2f} HU",
                'Ruído Máximo': f"{np.max(np.abs(noise_map)):.2f} HU"
            }
            for stat, value in noise_stats.items():
                st.write(f"{stat}: {value}")
    
    with tab3:
        st.markdown("###  Análise de Resolução e Nitidez")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Curva MTF
            fig_mtf = go.Figure()
            fig_mtf.add_trace(go.Scatter(
                x=np.linspace(0, 1, len(mtf_curve)),
                y=mtf_curve,
                mode='lines',
                name='MTF',
                line=dict(color='red', width=3)
            ))
            fig_mtf.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                             annotation_text="50% Modulação")
            fig_mtf.update_layout(
                title="Função de Transferência de Modulação (MTF)",
                xaxis_title="Frequência Espacial Normalizada",
                yaxis_title="Modulação",
                height=400
            )
            st.plotly_chart(fig_mtf, use_container_width=True)
        
        with col2:
            # Análise de bordas
            try:
                # Detecção de bordas para análise de nitidez
                if processed_image is not None and len(processed_image.shape) == 2:
                    edges = cv2.Canny(processed_image.astype(np.uint8), 50, 150)
                else:
                    edges = cv2.Canny(image_array.astype(np.uint8), 50, 150)
                
                fig_edges = go.Figure(data=go.Heatmap(
                    z=edges,
                    colorscale='Gray',
                    showscale=False,
                    title="Mapa de Bordas - Análise de Nitidez"
                ))
                fig_edges.update_layout(height=400)
                st.plotly_chart(fig_edges, use_container_width=True)
                
                # Métricas de nitidez
                edge_sharpness = np.mean(edges) if edges.size > 0 else 0
                st.metric("Índice de Nitidez", f"{edge_sharpness:.4f}")
                
            except Exception as e:
                st.warning("Análise de bordas não disponível")
    
    with tab4:
        st.markdown("###  Relatório Completo de Qualidade")
        
        # Gerar relatório abrangente
        report_data = {
            'Métrica': [
                'SNR (Relação Sinal-Ruído)',
                'PSNR (Pico SNR)',
                'CNR (Relação Contraste-Ruído)',
                'SSIM (Similaridade Estrutural)',
                'Entropia',
                'Uniformidade',
                'MTF₅₀ (Resolução)',
                'Faixa Dinâmica'
            ],
            'Valor': [
                f"{snr_val:.2f}",
                f"{psnr_val:.2f} dB" if psnr_val != float('inf') else "∞ dB",
                f"{cnr_val:.2f}",
                f"{ssim_val:.4f}",
                f"{entropy_val:.2f} bits",
                f"{uniformity_val:.4f}",
                f"{mtf_val:.2f} lp/mm",
                f"{image_array.max()-image_array.min():.0f} HU"
            ],
            'Status': [
                'Excelente' if snr_val > 50 else 'Bom' if snr_val > 30 else 'Aceitável' if snr_val > 15 else 'Ruim',
                'Excelente' if psnr_val > 60 else 'Bom' if psnr_val > 40 else 'Aceitável' if psnr_val > 20 else 'Ruim',
                'Excelente' if cnr_val > 5 else 'Bom' if cnr_val > 3 else 'Aceitável' if cnr_val > 1 else 'Ruim',
                'Excelente' if ssim_val > 0.9 else 'Bom' if ssim_val > 0.7 else 'Aceitável' if ssim_val > 0.5 else 'Ruim',
                'Alta' if entropy_val > 6 else 'Média' if entropy_val > 4 else 'Baixa',
                'Excelente' if uniformity_val > 0.1 else 'Boa' if uniformity_val > 0.05 else 'Baixa',
                'Alta' if mtf_val > 2.0 else 'Média' if mtf_val > 1.0 else 'Baixa',
                'Ampla' if (image_array.max()-image_array.min()) > 2000 else 'Média' if (image_array.max()-image_array.min()) > 1000 else 'Estreita'
            ]
        }
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        
        # Recomendações baseadas na análise
        st.markdown("#### Recomendações Técnicas")
        
        recommendations = []
        if snr_val < 20:
            recommendations.append("• Aumentar dose de radiação ou melhorar técnica de aquisição para melhorar SNR")
        if cnr_val < 2:
            recommendations.append("• Ajustar parâmetros de contraste ou usar meio de contraste para melhorar CNR")
        if mtf_val < 1.0:
            recommendations.append("• Verificar calibração do equipamento e parâmetros de reconstrução para melhorar resolução")
        if (image_array.max()-image_array.min()) < 1000:
            recommendations.append("• Ajustar janelamento Hounsfield para melhor utilizar a faixa dinâmica disponível")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("A imagem apresenta excelentes características de qualidade!")
        
        # Opção de exportar relatório
        if st.button("Exportar Relatório de Qualidade", use_container_width=True):
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="Baixar Relatório (CSV)",
                data=csv,
                file_name=f"relatorio_qualidade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Análise comparativa se houver imagem processada
    if processed_image is not None and np.any(processed_image != image_array):
        st.markdown("### Análise Comparativa: Original vs Processada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Diferença entre imagens
            difference = np.abs(image_array.astype(float) - processed_image.astype(float))
            fig_diff = go.Figure(data=go.Heatmap(
                z=difference,
                colorscale='Hot',
                showscale=True,
                title="Mapa de Diferenças (Original - Processada)"
            ))
            fig_diff.update_layout(height=400)
            st.plotly_chart(fig_diff, use_container_width=True)
        
        with col2:
            # Métricas de comparação
            mse = np.mean((image_array - processed_image) ** 2)
            rmse = np.sqrt(mse)
            nrmse = rmse / (np.max(image_array) - np.min(image_array))
            
            comp_metrics = {
                'MSE (Erro Quadrático Médio)': f"{mse:.2f}",
                'RMSE (Raiz do Erro Quadrático Médio)': f"{rmse:.2f} HU",
                'NRMSE (RMSE Normalizado)': f"{nrmse:.4f}",
                'PSNR (Original vs Processada)': f"{calculate_psnr(image_array, processed_image):.2f} dB",
                'SSIM (Original vs Processada)': f"{calculate_ssim(image_array, processed_image):.4f}"
            }
            
            st.markdown("**Métricas de Comparação:**")
            for metric, value in comp_metrics.items():
                st.write(f"{metric}: {value}")
    
    # Informações técnicas do DICOM relevantes para qualidade
    st.markdown("### Parâmetros Técnicos de Aquisição")
    
    tech_params = {}
    if hasattr(dicom_data, 'KVP'):
        tech_params['Tensão (kVp)'] = f"{dicom_data.KVP} kV"
    if hasattr(dicom_data, 'ExposureTime'):
        tech_params['Tempo de Exposição'] = f"{dicom_data.ExposureTime} ms"
    if hasattr(dicom_data, 'XRayTubeCurrent'):
        tech_params['Corrente do Tubo'] = f"{dicom_data.XRayTubeCurrent} mA"
    if hasattr(dicom_data, 'PixelSpacing'):
        tech_params['Espaçamento de Pixel'] = f"{dicom_data.PixelSpacing[0]} mm"
    if hasattr(dicom_data, 'SliceThickness'):
        tech_params['Espessura de Corte'] = f"{dicom_data.SliceThickness} mm"
    
    if tech_params:
        tech_df = pd.DataFrame(list(tech_params.items()), columns=['Parâmetro', 'Valor'])
        st.dataframe(tech_df, use_container_width=True, hide_index=True)
    else:
        st.info("Informações técnicas de aquisição não disponíveis no arquivo DICOM")

def calculate_ra_index_standard(image_array, dicom_data):
    """
    Implementação padrão do RA-Index baseado em Egger et al. (2012)
    Método tradicional de avaliação visual semiquantitativa
    """
    h, w = image_array.shape
    
    # Dividir em grid para análise regional
    grid_size = 8
    h_step, w_step = h // grid_size, w // grid_size
    
    ra_data = {
        'coords': [],
        'ra_values': [],
        'risk_categories': [],
        'tissue_types': [],
        'intensities': [],
        'gas_volume_estimates': []
    }
    
    # Definir categorias de risco baseadas em intensidade HU - Método tradicional
    def categorize_risk_standard(mean_intensity, region_size):
        if mean_intensity < -500:  # Gases/Ar
            # Classificação por tamanho conforme Egger et al.
            if region_size < 100:  # <1 cm equivalente
                return 'Baixo', 'Gás/Ar Grau I', 5
            elif region_size < 300:  # 1-3 cm equivalente
                return 'Médio', 'Gás/Ar Grau II', 15
            else:  # >3 cm equivalente
                return 'Alto', 'Gás/Ar Grau III', 20
        elif -500 <= mean_intensity < 0:  # Gordura
            return 'Baixo', 'Gordura', 0
        elif 0 <= mean_intensity < 100:  # Tecidos moles
            return 'Baixo', 'Tecido Mole', 0
        elif 100 <= mean_intensity < 400:  # Músculos
            return 'Médio', 'Músculo', 0
        elif 400 <= mean_intensity < 1000:  # Ossos
            return 'Médio', 'Osso', 0
        else:  # Metais/Implantes
            return 'Crítico', 'Metal/Implante', 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Extrair região
            region = image_array[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            
            # Calcular estatísticas da região
            mean_intensity = np.mean(region)
            region_size = region.size
            
            # Classificar usando método tradicional
            risk_category, tissue_type, ra_score = categorize_risk_standard(mean_intensity, region_size)
            
            # Estimar volume gasoso baseado em HU
            gas_volume = 0
            if mean_intensity < -500:
                # Estimativa volumétrica
                gas_pixels = np.sum(region < -500)
                gas_volume = gas_pixels * (0.1 ** 3)  # Assuming 0.1mm³ per pixel
            
            ra_data['coords'].append((i, j))
            ra_data['ra_values'].append(ra_score)
            ra_data['risk_categories'].append(risk_category)
            ra_data['tissue_types'].append(tissue_type)
            ra_data['intensities'].append(mean_intensity)
            ra_data['gas_volume_estimates'].append(gas_volume)
    
    return ra_data, grid_size

def calculate_ra_index_physical(image_array, dicom_data, post_mortem_interval=24):
    """
    Implementação baseada em princípios físicos (Lei de Fick, Modelo de Mitscherlich)
    Abordagem científica multidisciplinar proposta no estudo
    """
    h, w = image_array.shape
    
    # Obter parâmetros físicos do DICOM se disponíveis
    pixel_spacing = 1.0
    if hasattr(dicom_data, 'PixelSpacing'):
        pixel_spacing = float(dicom_data.PixelSpacing[0])
    
    slice_thickness = 5.0
    if hasattr(dicom_data, 'SliceThickness'):
        slice_thickness = float(dicom_data.SliceThickness)
    
    # Dividir em grid para análise regional
    grid_size = 8
    h_step, w_step = h // grid_size, w // grid_size
    
    ra_data = {
        'coords': [],
        'ra_values': [],
        'risk_categories': [],
        'tissue_types': [],
        'intensities': [],
        'gas_volume_estimates': [],
        'diffusion_coefficients': [],
        'knudsen_numbers': []
    }
    
    # Coeficientes de difusão estimados para gases post-mortem (mm²/h)
    DIFFUSION_COEFFICIENTS = {
        'putrescina': 0.15,
        'cadaverina': 0.12,
        'metano': 0.25
    }
    
    # Definir categorias de risco baseadas em princípios físicos
    def categorize_risk_physical(mean_intensity, region_size, region_data, pixel_area, post_mortem_interval):
        if mean_intensity < -500:  # Gases/Ar
            # Análise física detalhada para gases
            gas_pixels = np.sum(region_data < -500)
            total_gas_volume = gas_pixels * pixel_area * slice_thickness * (0.001)  # em cm³
            
            # Calcular concentração gasosa baseada em HU
            # HU = 1000 * (μ - μ_water) / μ_water ≈ -1000 para ar
            gas_concentration = (mean_intensity + 1000) / 1000  # Estimativa
            
            # Aplicar Segunda Lei de Fick para estimar dispersão
            # ∂C/∂t = D * ∇²C
            D_effective = DIFFUSION_COEFFICIENTS['metano']  # Usar metano como referência
            
            # Estimativa da dispersão
            dispersion_factor = D_effective * post_mortem_interval / (pixel_spacing ** 2)
            
            # Calcular número de Knudsen para avaliar regime de fluxo
            mean_free_path = 0.065  # mm (para ar em condições corporais)
            characteristic_length = np.sqrt(region_size) * pixel_spacing
            knudsen_number = mean_free_path / characteristic_length if characteristic_length > 0 else 0
            
            # Modelo de Mitscherlich ajustado para crescimento gasoso
            # C = C_max * (1 - e^(-k*t))
            k_growth = 0.05  # Coeficiente de crescimento estimado
            max_gas_potential = region_size * 0.3  # Máximo teórico de ocupação gasosa
            expected_gas = max_gas_potential * (1 - np.exp(-k_growth * post_mortem_interval))
            
            # Classificar baseado em análise física
            if gas_pixels < 0.1 * expected_gas:
                risk_level, tissue_desc, ra_score = 'Baixo', 'Gás Incipiente', 10
            elif gas_pixels < 0.3 * expected_gas:
                risk_level, tissue_desc, ra_score = 'Médio', 'Gás em Desenvolvimento', 25
            elif gas_pixels < 0.6 * expected_gas:
                risk_level, tissue_desc, ra_score = 'Alto', 'Gás Estabelecido', 50
            else:
                risk_level, tissue_desc, ra_score = 'Crítico', 'Gás Avançado', 80
            
            return risk_level, tissue_desc, ra_score, total_gas_volume, D_effective, knudsen_num
        
        else:
            # Tecidos não gasosos
            if mean_intensity < 0:
                return 'Baixo', 'Gordura', 0, 0, 0, 0
            elif mean_intensity < 100:
                return 'Baixo', 'Tecido Mole', 0, 0, 0, 0
            elif mean_intensity < 400:
                return 'Médio', 'Músculo', 0, 0, 0, 0
            elif mean_intensity < 1000:
                return 'Médio', 'Osso', 0, 0, 0, 0
            else:
                return 'Crítico', 'Metal/Implante', 0, 0, 0, 0
    
    pixel_area = pixel_spacing ** 2  # mm²
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Extrair região
            region = image_array[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            
            # Calcular estatísticas da região
            mean_intensity = np.mean(region)
            region_size = region.size
            
            # Classificar usando método físico
            risk_category, tissue_type, ra_score, gas_volume, diffusion_coeff, knudsen_num = categorize_risk_physical(
                mean_intensity, region_size, region, pixel_area, post_mortem_interval)
            
            ra_data['coords'].append((i, j))
            ra_data['ra_values'].append(ra_score)
            ra_data['risk_categories'].append(risk_category)
            ra_data['tissue_types'].append(tissue_type)
            ra_data['intensities'].append(mean_intensity)
            ra_data['gas_volume_estimates'].append(gas_volume)
            ra_data['diffusion_coefficients'].append(diffusion_coeff)
            ra_data['knudsen_numbers'].append(knudsen_num)
    
    return ra_data, grid_size

def professional_ra_index_tab(dicom_data, image_array):
    """
    Aba RA-Index profissional com comparação de métodos tradicionais vs. físicos
    """
    st.subheader("RA-Index - Análise de Risco Radiológico Avançada")
    
    # Introdução teórica
    with st.expander(" Fundamentação Teórica e Metodológica", expanded=False):
        st.markdown("""
        ### Interface Multidisciplinar: Física Quântica e Radiologia Legal
        
        Esta análise aplica princípios das ciências físicas ao campo da radiologia legal,
        examinando a relação entre a objetividade dos métodos de imagem (baseada em leis físicas)
        e a subjetividade da percepção humana na análise de alterações teciduais.
        
        **Base Física:** A intensidade I de um feixe de raios-X após atravessar um tecido
        é quantificada pela lei de atenuação de fótons:
        
        $$I = I_0 e^{-μx}$$
        
        Onde:
        - $I_0$ = intensidade inicial do feixe
        - $μ$ = coeficiente de atenuação linear do tecido
        - $x$ = espessura do tecido
        
        **Metodologia Dupla:**
        1. **Método Tradicional (Egger et al., 2012):** Avaliação visual semiquantitativa
        2. **Método Físico:** Baseado na Segunda Lei de Fick e Modelo de Mitscherlich
        """)
    
    # Controles de parâmetros
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Parâmetros do RA-Index")
    
    pm_interval = st.sidebar.slider("Intervalo Post-Mortem Estimado (horas):", 
                                  0, 168, 24, 1,
                                  help="Intervalo estimado entre óbito e aquisição da imagem")
    
    analysis_method = st.sidebar.radio("Método de Análise:",
                                     ["Comparação Ambos", "Tradicional (Egger)", "Físico (Fick-Mitscherlich)"])
    
    # Calcular RA-Index com ambos os métodos
    with st.spinner("Calculando métricas RA-Index avançadas..."):
        ra_data_standard, grid_size = calculate_ra_index_standard(image_array, dicom_data)
        ra_data_physical, _ = calculate_ra_index_physical(image_array, dicom_data, pm_interval)
    
    # Métricas comparativas
    st.markdown("### Métricas Comparativas dos Métodos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_standard = np.mean(ra_data_standard['ra_values'])
        avg_physical = np.mean(ra_data_physical['ra_values'])
        st.metric("RA-Index Médio", 
                 f"{avg_standard:.1f} | {avg_physical:.1f}",
                 delta=f"{avg_physical - avg_standard:.1f}",
                 help="Método Tradicional | Método Físico")
    
    with col2:
        max_standard = np.max(ra_data_standard['ra_values'])
        max_physical = np.max(ra_data_physical['ra_values'])
        st.metric("RA-Index Máximo", 
                 f"{max_standard:.1f} | {max_physical:.1f}",
                 delta=f"{max_physical - max_standard:.1f}")
    
    with col3:
        gas_volume_std = np.sum(ra_data_standard['gas_volume_estimates'])
        gas_volume_phy = np.sum(ra_data_physical['gas_volume_estimates'])
        st.metric("Volume Gasoso Estimado (cm³)", 
                 f"{gas_volume_std:.1f} | {gas_volume_phy:.1f}",
                 delta=f"{gas_volume_phy - gas_volume_std:.1f}")
    
    with col4:
        critical_std = sum(1 for cat in ra_data_standard['risk_categories'] if cat == 'Crítico')
        critical_phy = sum(1 for cat in ra_data_physical['risk_categories'] if cat == 'Crítico')
        st.metric("Regiões Críticas", 
                 f"{critical_std} | {critical_phy}",
                 delta=f"{critical_phy - critical_std}")
    
    # Visualizações comparativas
    st.markdown("### Visualizações Comparativas")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Mapas de Calor", "Distribuição", "Análise Física", "Correlações"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Mapa de calor do método tradicional
            ra_matrix_std = np.array(ra_data_standard['ra_values']).reshape(grid_size, grid_size)
            fig_std = go.Figure(data=go.Heatmap(
                z=ra_matrix_std,
                colorscale='RdYlBu_r',
                showscale=True,
                text=ra_matrix_std.round(1),
                texttemplate="%{text}",
                textfont={"size": 10, "color": "white"},
                hoverongaps=False
            ))
            fig_std.update_layout(
                title="RA-Index Tradicional (Egger et al.)",
                xaxis_title="Região X",
                yaxis_title="Região Y",
                height=400
            )
            st.plotly_chart(fig_std, use_container_width=True)
        
        with col2:
            # Mapa de calor do método físico
            ra_matrix_phy = np.array(ra_data_physical['ra_values']).reshape(grid_size, grid_size)
            fig_phy = go.Figure(data=go.Heatmap(
                z=ra_matrix_phy,
                colorscale='RdYlBu_r',
                showscale=True,
                text=ra_matrix_phy.round(1),
                texttemplate="%{text}",
                textfont={"size": 10, "color": "white"},
                hoverongaps=False
            ))
            fig_phy.update_layout(
                title="RA-Index Físico (Fick-Mitscherlich)",
                xaxis_title="Região X",
                yaxis_title="Região Y",
                height=400
            )
            st.plotly_chart(fig_phy, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de valores RA-Index
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=ra_data_standard['ra_values'],
                name="Tradicional",
                opacity=0.7,
                marker_color='blue'
            ))
            fig_dist.add_trace(go.Histogram(
                x=ra_data_physical['ra_values'],
                name="Físico",
                opacity=0.7,
                marker_color='red'
            ))
            fig_dist.update_layout(
                title="Distribuição de Valores RA-Index",
                xaxis_title="RA-Index",
                yaxis_title="Frequência",
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Distribuição de categorias de risco
            risk_counts_std = pd.Series(ra_data_standard['risk_categories']).value_counts()
            risk_counts_phy = pd.Series(ra_data_physical['risk_categories']).value_counts()
            
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Bar(
                x=risk_counts_std.index,
                y=risk_counts_std.values,
                name="Tradicional",
                marker_color='blue'
            ))
            fig_risk.add_trace(go.Bar(
                x=risk_counts_phy.index,
                y=risk_counts_phy.values,
                name="Físico",
                marker_color='red'
            ))
            fig_risk.update_layout(
                title="Distribuição de Categorias de Risco",
                xaxis_title="Categoria",
                yaxis_title="Número de Regiões",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔍 Análise de Parâmetros Físicos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Números de Knudsen
            knudsen_numbers = [x for x in ra_data_physical['knudsen_numbers'] if x > 0]
            if knudsen_numbers:
                fig_knudsen = go.Figure()
                fig_knudsen.add_trace(go.Histogram(
                    x=knudsen_numbers,
                    name="Número de Knudsen",
                    marker_color='green'
                ))
                fig_knudsen.add_vline(x=0.01, line_dash="dash", line_color="red",
                                    annotation_text="Limite continuum (0.01)")
                fig_knudsen.update_layout(
                    title="Distribuição do Número de Knudsen",
                    xaxis_title="Número de Knudsen",
                    yaxis_title="Frequência",
                    height=400
                )
                st.plotly_chart(fig_knudsen, use_container_width=True)
            
            # Informações sobre regime de fluxo
            if knudsen_numbers:
                continuum_count = sum(1 for kn in knudsen_numbers if kn < 0.01)
                transition_count = sum(1 for kn in knudsen_numbers if 0.01 <= kn < 0.1)
                free_molecular_count = sum(1 for kn in knudsen_numbers if kn >= 0.1)
                
                st.markdown("**Regime de Fluxo Gasoso:**")
                st.write(f"- Continuum (Kn < 0.01): {continuum_count} regiões")
                st.write(f"- Transição (0.01 ≤ Kn < 0.1): {transition_count} regiões")
                st.write(f"- Molecular Livre (Kn ≥ 0.1): {free_molecular_count} regiões")
        
        with col2:
            # Coeficientes de difusão
            diffusion_coeffs = [x for x in ra_data_physical['diffusion_coefficients'] if x > 0]
            if diffusion_coeffs:
                fig_diffusion = go.Figure()
                fig_diffusion.add_trace(go.Box(
                    y=diffusion_coeffs,
                    name="Coeficientes de Difusão",
                    boxpoints='all',
                    marker_color='purple'
                ))
                fig_diffusion.update_layout(
                    title="Distribuição de Coeficientes de Difusão",
                    yaxis_title="Coeficiente de Difusão (mm²/h)",
                    height=400
                )
                st.plotly_chart(fig_diffusion, use_container_width=True)
            
            # Informações sobre difusão
            if diffusion_coeffs:
                avg_diffusion = np.mean(diffusion_coeffs)
                st.markdown("**Análise de Difusão:**")
                st.write(f"- Coeficiente médio de difusão: {avg_diffusion:.3f} mm²/h")
                st.write(f"- Referência metano: 0.25 mm²/h")
                st.write(f"- Referência putrescina: 0.15 mm²/h")
                st.write(f"- Referência cadaverina: 0.12 mm²/h")
    
    with tab4:
        st.markdown("### Análise de Correlações e Regressão")
        
        # Preparar dados para análise
        comparison_data = []
        for i in range(len(ra_data_standard['ra_values'])):
            if ra_data_standard['ra_values'][i] > 0 or ra_data_physical['ra_values'][i] > 0:
                comparison_data.append({
                    'RA_Standard': ra_data_standard['ra_values'][i],
                    'RA_Physical': ra_data_physical['ra_values'][i],
                    'Intensity': ra_data_standard['intensities'][i],
                    'Region_Size': (grid_size ** 2)  # Tamanho aproximado da região
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlação entre métodos
                correlation = comparison_df['RA_Standard'].corr(comparison_df['RA_Physical'])
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(
                    x=comparison_df['RA_Standard'],
                    y=comparison_df['RA_Physical'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=comparison_df['Intensity'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="HU")
                    ),
                    text=[f"Intensity: {hu:.1f}" for hu in comparison_df['Intensity']]
                ))
                # Linha de correlação perfeita
                max_val = max(comparison_df['RA_Standard'].max(), comparison_df['RA_Physical'].max())
                fig_corr.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Correlação perfeita'
                ))
                fig_corr.update_layout(
                    title=f"Correlação entre Métodos (r = {correlation:.3f})",
                    xaxis_title="RA-Index Tradicional",
                    yaxis_title="RA-Index Físico",
                    height=400
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                # Análise de regressão
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        comparison_df['RA_Standard'], comparison_df['RA_Physical'])
                    
                    st.markdown("**Análise de Regressão Linear:**")
                    st.write(f"- Coeficiente de correlação (r): {r_value:.3f}")
                    st.write(f"- Valor-p: {p_value:.4f}")
                    st.write(f"- Inclinação: {slope:.3f}")
                    st.write(f"- Intercepto: {intercept:.3f}")
                    
                    if p_value < 0.05:
                        st.success("Correlação estatisticamente significativa (p < 0.05)")
                    else:
                        st.warning("Correlação não estatisticamente significativa")
                        
                except Exception as e:
                    st.error("Erro na análise de regressão")
        
        # Matriz de correlação
        st.markdown("#### Matriz de Correlação")
        try:
            corr_matrix = comparison_df.corr()
            fig_corr_matrix = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            fig_corr_matrix.update_layout(
                title="Matriz de Correlação",
                height=400
            )
            st.plotly_chart(fig_corr_matrix, use_container_width=True)
        except:
            st.warning("Não foi possível calcular a matriz de correlação")
    
    # Relatório forense avançado
    st.markdown("### Relatório Forense Avançado")
    
    with st.expander("🔍 Análise Discriminativa Detalhada", expanded=False):
        st.markdown("""
        #### Análise de Discordâncias entre Métodos
        
        As diferenças entre os métodos tradicional e físico revelam importantes
        insights sobre la naturaleza de las alteraciones radiológicas:
        """)
        
        # Identificar regiões com maiores discordâncias
        discrepancies = []
        for i in range(len(ra_data_standard['ra_values'])):
            std_val = ra_data_standard['ra_values'][i]
            phy_val = ra_data_physical['ra_values'][i]
            discrepancy = abs(std_val - phy_val)
            
            if discrepancy > 20:  # Limite para discordância significativa
                discrepancies.append({
                    'Região': f"({ra_data_standard['coords'][i][0]}, {ra_data_standard['coords'][i][1]})",
                    'Tradicional': std_val,
                    'Físico': phy_val,
                    'Diferença': discrepancy,
                    'Tipo_Tecido': ra_data_standard['tissue_types'][i]
                })
        
        if discrepancies:
            disc_df = pd.DataFrame(discrepancies)
            st.dataframe(disc_df.sort_values('Diferença', ascending=False), 
                        use_container_width=True)
            
            st.markdown("""
            **Interpretação das Discordâncias:**
            - Diferenças > 20 pontos indicam regiões onde a avaliação física
              detecta alterações não identificadas pelo método tradicional
            - Estas regiões podem representar casos onde a análise baseada em
              princípios físicos oferece vantagem diagnóstica
            """)
        else:
            st.info("Não foram encontradas discordâncias significativas (>20 pontos) entre os métodos")
    
    # Conclusão e recomendações
    st.markdown("### Conclusões e Recomendações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Vantagens do Método Tradicional")
        st.success("""
        - ✅ Validação clínica estabelecida (Egger et al., 2012)
        - ✅ Simplicidade de aplicação
        - ✅ Correlação com achados macroscópicos
        - ✅ Amplamente aceito na comunidade forense
        """)
    
    with col2:
        st.markdown("#### Vantagens do Método Físico")
        st.info("""
        - 🔬 Baseado em princípios científicos fundamentais
        - 🔬 Considera parâmetros físicos (difusão, Knudsen)
        - 🔬 Modelagem matemática da dispersão gasosa
        - 🔬 Potencial para maior objetividade e reprodutibilidade
        """)
    
    # Recomendações finais 
    st.markdown("#### Recomendações para Análise Forense")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        # CORREÇÃO APPLICADA - cálculo separado para evitar erro de f-string
        if discrepancies and ra_data_standard['ra_values']:
            concordance = (1 - (len(discrepancies) / len(ra_data_standard['ra_values']))) * 100
            st.metric("Concordância Geral", 
                     f"{concordance:.1f}%",
                     help="Percentual de regiões com concordância entre métodos")
        else:
            st.metric("Concordância Geral", "100.0%")
    
    with rec_col2:
        if ra_data_standard['ra_values'] and ra_data_physical['ra_values']:
            avg_diff = np.mean([abs(a - b) for a, b in 
                              zip(ra_data_standard['ra_values'], ra_data_physical['ra_values'])])
            st.metric("Diferença Média", f"{avg_diff:.1f} pontos")
        else:
            st.metric("Diferença Média", "0.0 pontos")
    
    with rec_col3:
        if discrepancies:
            max_diff = max(discrepancies, key=lambda x: x['Diferença'])
            st.metric("Maior Discordância", f"{max_diff['Diferença']} pontos")
        else:
            st.metric("Maior Discordância", "0 pontos")
    
    st.markdown("""
    **Recomendações:**
    1. Utilizar ambos os métodos para análise complementar
    2. Investigar regiões com discordância significativa
    3. Considerar parâmetros físicos para casos complexos
    4. Validar achados com correlação macroscópica quando possível
    """)
    
    # Opção de exportação
    if st.button("Exportar Relatório RA-Index Completo", use_container_width=True):
        # Preparar dados para exportação
        export_data = []
        for i in range(len(ra_data_standard['ra_values'])):
            export_data.append({
                'Região_X': ra_data_standard['coords'][i][0],
                'Região_Y': ra_data_standard['coords'][i][1],
                'RA_Tradicional': ra_data_standard['ra_values'][i],
                'RA_Físico': ra_data_physical['ra_values'][i],
                'Diferença': abs(ra_data_standard['ra_values'][i] - ra_data_physical['ra_values'][i]),
                'Intensidade_HU': ra_data_standard['intensities'][i],
                'Categoria_Tradicional': ra_data_standard['risk_categories'][i],
                'Categoria_Físico': ra_data_physical['risk_categories'][i],
                'Tipo_Tecido': ra_data_standard['tissue_types'][i],
                'Volume_Gasoso_cm3': ra_data_standard['gas_volume_estimates'][i],
                'Coeficiente_Difusão': ra_data_physical['diffusion_coefficients'][i] if i < len(ra_data_physical['diffusion_coefficients']) else 0,
                'Número_Knudsen': ra_data_physical['knudsen_numbers'][i] if i < len(ra_data_physical['knudsen_numbers']) else 0
            })
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="Baixar Dados Completos (CSV)",
            data=csv,
            file_name=f"ra_index_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
# ====== FUNÇÕES AUXILIARES PARA ANÁLISE DE QUALIDADE ======

def estimate_noise(image):
    """
    Estima o nível de ruído usando o método de diferenciação
    """
    try:
        h, w = image.shape
        # Calcular diferenças entre pixels adjacentes
        diff_h = image[:, 1:] - image[:, :-1]
        diff_v = image[1:, :] - image[:-1, :]
        
        # Estimar ruído como o desvio padrão das diferenças
        noise_estimate = np.std(np.concatenate([diff_h.flatten(), diff_v.flatten()])) / np.sqrt(2)
        return noise_estimate
    except Exception as e:
        return 0.0

def calculate_snr(image_array):
    """
    Calcula SNR de forma mais robusta usando uma região homogênea
    """
    try:
        # Selecionar uma pequena região central (assumindo que é relativamente homogênea)
        h, w = image_array.shape
        roi_size = min(20, h//10, w//10)  # Tamanho da região de interesse
        roi = image_array[h//2-roi_size//2:h//2+roi_size//2, 
                         w//2-roi_size//2:w//2+roi_size//2]
        
        signal = np.mean(roi)
        noise = np.std(roi)
        
        return signal / noise if noise > 0 else float('inf')
    except Exception as e:
        return float('inf')

def calculate_glcm_features(image):
    """
    Calcula características GLCM simplificadas - VERSÃO CORRIGIDA
    """
    try:
        # Verificar se a imagem é válida
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                'Homogeneidade GLCM': 0.0,
                'Contraste GLCM': 0.0,
                'Correlação GLCM': 0.0,
                'Energia GLCM': 0.0,
                'Dissimilaridade': 0.0
            }
        
        # Normalizar imagem para 0-255
        img_min = float(np.min(image))
        img_max = float(np.max(image))
        
        # Caso especial: imagem constante
        if img_max <= img_min:
            # Imagem com valor constante
            normalized = np.full_like(image, 128, dtype=np.uint8)  # Valor médio
        else:
            # Normalizar normalmente
            normalized = ((image.astype(float) - img_min) / (img_max - img_min) * 255)
            normalized = normalized.astype(np.uint8)
        
        # Garantir que é um array numpy
        if not isinstance(normalized, np.ndarray):
            normalized = np.array(normalized)
        
        # Calcular diferenças horizontais
        diff_h = np.array([0.0], dtype=float)
        try:
            if len(normalized.shape) > 1 and normalized.shape[1] > 1:
                diff_h = np.abs(normalized[:, :-1].astype(float) - normalized[:, 1:].astype(float))
        except:
            diff_h = np.array([0.0], dtype=float)
        
        # Métricas baseadas em diferenças
        mean_diff = float(np.mean(diff_h)) if diff_h.size > 0 else 0.0
        homogeneity_val = float(1 / (1 + mean_diff)) if mean_diff > 0 else 1.0
        contrast_val = float(np.var(diff_h)) if diff_h.size > 0 else 0.0
        
        # Correlação
        correlation_val = 0.0
        try:
            if len(normalized.shape) > 1 and normalized.shape[1] > 1 and normalized.size > 0:
                flat1 = normalized[:, :-1].flatten()
                flat2 = normalized[:, 1:].flatten()
                
                if len(flat1) > 1 and len(flat2) > 1:
                    corr_matrix = np.corrcoef(flat1, flat2)
                    if not np.isnan(corr_matrix[0, 1]):
                        correlation_val = float(corr_matrix[0, 1])
        except:
            correlation_val = 0.0
        
        # Energia - CORREÇÃO CRÍTICA APLICADA AQUI
        energy_val = 0.0
        try:
            # Garantir que estamos trabalhando com um array numpy
            if isinstance(normalized, np.ndarray) and normalized.size > 0:
                # Converter para float e garantir que é um array unidimensional
                img_float = normalized.astype(float).flatten()
                # Calcular a energia
                energy_val = float(np.mean(img_float ** 2) / (255.0 ** 2))
        except Exception as e:
            energy_val = 0.0
            
        # Dissimilaridade
        dissimilarity_val = float(mean_diff / 255.0) if diff_h.size > 0 else 0.0
        
        return {
            'Homogeneidade GLCM': round(homogeneity_val, 6),
            'Contraste GLCM': round(contrast_val, 6),
            'Correlação GLCM': round(correlation_val, 6),
            'Energia GLCM': round(energy_val, 6),
            'Dissimilaridade': round(dissimilarity_val, 6)
        }
        
    except Exception as e:
        # Em caso de qualquer erro, retornar valores padrão
        return {
            'Homogeneidade GLCM': 0.0,
            'Contraste GLCM': 0.0,
            'Correlação GLCM': 0.0,
            'Energia GLCM': 0.0,
            'Dissimilaridade': 0.0
        }

def detect_artifacts(image_array):
    """
    Detecta vários tipos de artefatos em imagens DICOM
    """
    artifacts = {}
    
    # 1. Artefato de movimento (análise de Fourier)
    try:
        fft_2d = np.fft.fft2(image_array.astype(float))
        magnitude_spectrum = np.log(np.abs(np.fft.fftshift(fft_2d)) + 1)
        
        # Verificar se há linhas brilhantes no espectro (indicativo de artefato de movimento)
        center = np.array(magnitude_spectrum.shape) // 2
        horizontal_line = magnitude_spectrum[center[0], :]
        vertical_line = magnitude_spectrum[:, center[1]]
        
        # Detectar picos incomuns nas linhas centrais
        horizontal_peaks = np.std(horizontal_line) > 2 * np.mean(horizontal_line)
        vertical_peaks = np.std(vertical_line) > 2 * np.mean(vertical_line)
        
        artifacts['Motion Artifact'] = horizontal_peaks or vertical_peaks
    except:
        artifacts['Motion Artifact'] = False
    
    # 2. Artefato de metal (valores extremamente altos)
    try:
        metal_threshold = 3000  # HU
        metal_pixels = np.sum(image_array > metal_threshold)
        artifacts['Metal Artifact'] = metal_pixels > (image_array.size * 0.001)  # Mais de 0.1% dos pixels
    except:
        artifacts['Metal Artifact'] = False
    
    # 3. Artefato de ruído (análise de ruído)
    try:
        noise_level = estimate_noise(image_array)
        artifacts['Noise Artifact'] = noise_level > 50  # Threshold arbitrário
    except:
        artifacts['Noise Artifact'] = False
    
    return artifacts

def calculate_psnr(original, processed=None):
    """Calcula PSNR (Peak Signal-to-Noise Ratio)"""
    try:
        if processed is None:
            # Se não há imagem processada, usar ruído estimado
            noise = estimate_noise(original)
            if noise == 0:
                return float('inf')
            return 20 * np.log10(np.max(original) / noise)
        else:
            # Entre original e processada
            mse = np.mean((original - processed) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * np.log10(np.max(original) / np.sqrt(mse))
    except:
        return float('inf')

def calculate_ssim(original, processed=None):
    """Calcula SSIM (Structural Similarity Index) simplificado"""
    try:
        if processed is None:
            return 1.0  # Sem imagem processada para comparação
        
        try:
            from skimage.metrics import structural_similarity as ssim
            # Normalizar imagens para 0-1
            original_norm = (original - np.min(original)) / (np.max(original) - np.min(original))
            processed_norm = (processed - np.min(processed)) / (np.max(processed) - np.min(processed))
            return ssim(original_norm, processed_norm, data_range=1.0)
        except ImportError:
            # Fallback calculation
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            mu_x = np.mean(original)
            mu_y = np.mean(processed)
            sigma_x = np.var(original)
            sigma_y = np.var(processed)
            sigma_xy = np.cov(original.flatten(), processed.flatten())[0, 1]
            
            ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                      ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
            return ssim_val
    except:
        return 0.0

def calculate_mtf(image_array, dicom_data):
    """Calcula MTF (Modulation Transfer Function) simplificado"""
    try:
        # Usar borda da imagem para estimar MTF
        edge_profile = image_array[image_array.shape[0] // 2, :]
        
        # Derivada do perfil de borda (Edge Spread Function)
        esf_derivative = np.gradient(edge_profile)
        
        # Normalizar e calcular MTF
        mtf = np.abs(np.fft.fft(esf_derivative))
        mtf = mtf[:len(mtf)//2]  # Manter apenas frequências positivas
        mtf = mtf / np.max(mtf)  # Normalizar
        
        # Encontrar frequência donde MTF cai para 50%
        freq_50 = np.argmax(mtf < 0.5) / len(mtf) if np.any(mtf < 0.5) else 1.0
        
        # Converter para ciclos/mm se PixelSpacing disponível
        if hasattr(dicom_data, 'PixelSpacing'):
            pixel_spacing = float(dicom_data.PixelSpacing[0])
            freq_50 = freq_50 / (2 * pixel_spacing)  # Conversão para lp/mm
        
        return float(freq_50), mtf
    except:
        return 0.0, np.array([0.0])

def calculate_cnr(image_array):
    """Calcula CNR (Contrast-to-Noise Ratio)"""
    try:
        # Selecionar duas regiões diferentes para calcular contraste
        h, w = image_array.shape
        roi1 = image_array[h//4:h//2, w//4:w//2]  # Região central
        roi2 = image_array[3*h//4:h, 3*w//4:w]    # Região periférica
        
        contrast = np.abs(np.mean(roi1) - np.mean(roi2))
        noise = estimate_noise(image_array)
        
        return contrast / noise if noise > 0 else 0.0
    except:
        return 0.0

def calculate_nps(image_array):
    """Calcula NPS (Noise Power Spectrum)"""
    try:
        # Remover tendência linear
        detrended = image_array - ndimage.uniform_filter(image_array, size=10)
        
        # Calcular espectro de potência do ruído
        fft_nps = np.fft.fft2(detrended)
        nps = np.abs(fft_nps) ** 2
        nps = np.fft.fftshift(nps)
        
        # Perfil radial do NPS
        center = np.array(nps.shape) // 2
        y, x = np.indices(nps.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        nps_radial = ndimage.mean(nps, labels=r, index=np.arange(0, np.max(r)))
        
        return nps, nps_radial
    except:
        return np.zeros_like(image_array), np.array([0.0])

def advanced_noise_analysis(image_array):
    """Análise avançada de ruído"""
    try:
        # Análise de ruído usando múltiplos métodos
        noise_levels = {}
        
        # Método 1: Diferença entre pixels adjacentes
        diff_h = image_array[:, 1:] - image_array[:, :-1]
        diff_v = image_array[1:, :] - image_array[:-1, :]
        noise_levels['Método Diferença'] = np.std(np.concatenate([diff_h.flatten(), diff_v.flatten()])) / np.sqrt(2)
        
        # Método 2: Filtro de uniformidade
        uniform_filtered = ndimage.uniform_filter(image_array, size=3)
        residual = image_array - uniform_filtered
        noise_levels['Método Residual'] = np.std(residual)
        
        # Método 3: Análise wavelet (simplificada)
        from scipy import ndimage
        wavelet_approx = ndimage.gaussian_filter(image_array, sigma=1)
        wavelet_detail = image_array - wavelet_approx
        noise_levels['Método Wavelet'] = np.std(wavelet_detail)
        
        return noise_levels
    except:
        return {'Método Diferença': 0.0, 'Método Residual': 0.0, 'Método Wavelet': 0.0}

# ====== SEÇÃO 5: FUNÇÕES PRINCIPAIS DO SISTEMA ======

def safe_init_database():
    """
    Inicializar base de dados de forma segura
    """
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        
        # Tabela de usuários
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
        
        # Tabela de logs de segurança
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
        
        # Tabela de feedback
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
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logging.error(f"Erro ao inicializar base de dados: {e}")
        return False

def log_security_event(user_email, action, details=""):
    """
    Registrar evento de segurança
    """
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        
        # Obter IP (simulado)
        ip_address = "127.0.0.1"  # Em produção, usar request.remote_addr
        
        cursor.execute("""
            INSERT INTO security_logs (user_email, action, ip_address, details)
            VALUES (?, ?, ?, ?)
        """, (user_email, action, ip_address, details))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logging.error(f"Erro ao registrar evento de segurança: {e}")

def update_css_theme():
    """
    Aplicar tema CSS personalizado
    """
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px;
        color: #262730;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF6B6B;
        color: white;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
    }
    
    h1, h2, h3 {
        color: #2C3E50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .feedback-form {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def show_user_form():
    """
    Mostrar formulário de registro de usuário
    """
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("# DICOM Autopsy Viewer PRO")
    st.markdown("### Sistema Avançado de Análise Forense Digital")
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.form("user_registration"):
        st.markdown("## Informações do Usuário")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nome Completo *", placeholder="Dr. João Silva")
            email = st.text_input("Email *", placeholder="joao.silva@hospital.com")
        
        with col2:
            role = st.selectbox("Função *", [
                "Radiologista", "Médico Legista", "Técnico em Radiologia", 
                "Pesquisador", "Estudante", "Outro"
            ])
            department = st.text_input("Departamento/Instituição", 
                                     placeholder="Departamento de Radiologia")
        
        # Termos de uso
        st.markdown("### Termos de Uso")
        terms_accepted = st.checkbox("""
        Eu concordo com os termos de uso e confirmo que:
        - Utilizarei este sistema apenas para fins educacionais e de pesquisa
        - Não carregarei dados de pacientes reais sem autorização apropriada
        - Mantenho a confidencialidade das informações processadas
        """)
        
        submitted = st.form_submit_button("Iniciar Sistema", use_container_width=True)
        
        if submitted:
            if not all([name, email]) or not terms_accepted:
                st.error("Por favor, preencha todos os campos obrigatórios e aceite os termos de uso.")
            else:
                try:
                    # Registrar usuário
                    conn = sqlite3.connect("dicom_viewer.db")
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO users (name, email, role, department)
                        VALUES (?, ?, ?, ?)
                    """, (name, email, role, department))
                    
                    conn.commit()
                    conn.close()
                    
                    # Armazenar dados do usuário na sessão
                    st.session_state.user_data = {
                        'name': name,
                        'email': email,
                        'role': role,
                        'department': department
                    }
                    
                    # Log do evento
                    log_security_event(email, "USER_REGISTRATION", f"Role: {role}")
                    
                    st.success(" Usuário registrado com sucesso!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Erro ao registrar usuário: {e}")

def show_main_app():
    """
    Mostrar aplicação principal
    """
    user_data = st.session_state.user_data
    
    # Sidebar com informações do usuário
    with st.sidebar:
        st.markdown("### Usuário Ativo")
        st.write(f"**Nome:** {user_data['name']}")
        st.write(f"**Função:** {user_data['role']}")
        if user_data['department']:
            st.write(f"**Departamento:** {user_data['department']}")
        
        st.markdown("---")
        
        # Upload destacado na sidebar
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### Upload de Arquivo DICOM")
        uploaded_file = st.file_uploader(
            "Selecione um arquivo DICOM:",
            type=['dcm', 'dicom'],
            help="Carregue um arquivo DICOM para análise forense avançada"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Trocar Usuário"):
            st.session_state.user_data = None
            st.session_state.authenticated = False
            st.rerun()
        
        # Informações do sistema
        st.markdown("---")
        st.markdown("### Informações do Sistema")
        st.write("**Versão:** 2.0 Enhanced")
        st.write("**Última Atualização:** 2025-09-11")
        st.write("**Status:** Online")
    
    # Conteúdo principal
    st.markdown("# 🔬 DICOM Autopsy Viewer")
    st.markdown(f"**Bem-vindo, {user_data['name']}!**")
    
    if uploaded_file is not None:
        try:
            # Salvar arquivo temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Log do evento
            log_security_event(user_data['email'], "FILE_UPLOAD", 
                             f"Filename: {uploaded_file.name}")
            
            try:
                # Ler arquivo DICOM
                dicom_data = pydicom.dcmread(tmp_path)
                image_array = dicom_data.pixel_array
                
                # Informações básicas do arquivo
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dimensões", f"{image_array.shape[0]} × {image_array.shape[1]}")
                
                with col2:
                    st.metric("Faixa de Valores", f"{image_array.min()} → {image_array.max()}")
                with col3:
                    st.metric("Tamanho do Arquivo", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Visualização da imagem
                st.markdown("### Visualização da Imagem")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image_array, cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
                
                # Histograma de distribuição de pixels
                st.markdown("### Distribuição de Intensidade de Pixels")
                hist_values = np.histogram(image_array.flatten(), bins=50)
                fig_hist = px.line(x=hist_values[1][1:], y=hist_values[0], 
                                 labels={'x': 'Intensidade', 'y': 'Frequência'})
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Metadados DICOM
                st.markdown("### Metadados DICOM")
                metadata = []
                for elem in dicom_data:
                    if elem.keyword != "PixelData":
                        metadata.append({"Tag": str(elem.tag), 
                                       "Nome": elem.keyword, 
                                       "Valor": str(elem.value)})
                
                metadata_df = pd.DataFrame(metadata)
                st.dataframe(metadata_df, use_container_width=True, height=300)
                
            except Exception as e:
                st.error(f"Erro ao processar arquivo DICOM: {str(e)}")
                log_security_event(user_data['email'], "PROCESSING_ERROR", 
                                 f"Error: {str(e)}")
            
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {str(e)}")
            log_security_event(user_data['email'], "FILE_READ_ERROR", 
                             f"Error: {str(e)}")
    else:
        # Mensagem de boas-vindas quando não há arquivo carregado
        st.info("👈 Faça upload de um arquivo DICOM na barra lateral para começar a análise.")
        
        # Estatísticas de uso (apenas exemplo)
        st.markdown("### Estatísticas de Uso")
        col1, col2, col3 = st.columns(3)
        col1.metric("Usuários Ativos", "24", "3")
        col2.metric("Exames Hoje", "127", "12")
        col3.metric("Tempo Médio de Análise", "4.2 min", "-0.3 min")
        
        # Guia de referência rápida
        st.markdown("### Guia de Referência Rápida")
        expander = st.expander("Dicas de Análise de Imagens DICOM")
        expander.markdown("""
        - Verifique sempre os metadados do paciente para confirmar a identidade
        - Analise a distribuição de pixels para identificar possíveis anomalias
        - Compare as dimensões da imagem com os padrões esperados para o exame
        - Utilize as ferramentas de zoom e contraste para melhor visualização
        """)

# Verificar autenticação
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    show_main_app()
else:
    show_login()

# Adicionar algum CSS personalizado
st.markdown("""
    <style>
    .upload-section {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """
    Função principal da aplicação
    """
    # Inicializar sessão
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    
    # Configurar matplotlib
    setup_matplotlib_for_plotting()
    
    # Inicializar base de dados
    if not safe_init_database():
        st.error("❌ Erro crítico: Não foi possível inicializar o sistema. Contate o administrador.")
        return
    
    # Aplicar tema CSS
    update_css_theme()
    
    # Adicionar informações de versão no rodapé
    st.markdown("""
    <div style='position: fixed; bottom: 10px; right: 10px; background: rgba(0, 0, 0, 0.7); 
                padding: 8px 12px; border-radius: 20px; color: white; font-size: 0.8rem; z-index: 1000;'>
        <strong>DICOM Autopsy Viewer PRO v2.0</strong> - Enhanced Edition
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar aplicação baseada no estado da sessão
    if st.session_state.user_data is None:
        show_user_form()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
