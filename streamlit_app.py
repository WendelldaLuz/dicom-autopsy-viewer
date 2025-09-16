import streamlit as st
import pydicom
import tempfile
import os
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy import ndimage  # Import corrigido para evitar NameError

# Configurações iniciais da página
st.set_page_config(
    page_title="DICOM Autopsy Viewer PRO",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Upload e leitura DICOM ---
def upload_and_read_dicom():
    uploaded_file = st.sidebar.file_uploader("Selecione um arquivo DICOM:", type=['dcm', 'dicom'])
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            dicom_data = pydicom.dcmread(tmp_path)
            image_array = dicom_data.pixel_array

            st.session_state['dicom_data'] = dicom_data
            st.session_state['image_array'] = image_array
            st.session_state['uploaded_file_name'] = uploaded_file.name

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            return dicom_data, image_array

        except Exception as e:
            st.sidebar.error(f"Erro ao processar arquivo DICOM: {e}")
            logging.error(f"Erro ao processar arquivo DICOM: {e}")
            return None, None
    else:
        return None, None

# --- Visualização ---
def enhanced_visualization_tab(dicom_data, image_array):
    st.subheader("Visualização Avançada")
    st.image(image_array, clamp=True, channels="L", use_column_width=True)
    st.markdown("Ajuste a janela Hounsfield:")
    window_center = st.slider("Centro da Janela", int(np.min(image_array)), int(np.max(image_array)), int(np.mean(image_array)))
    window_width = st.slider("Largura da Janela", 1, int(np.ptp(image_array)), int(np.ptp(image_array)//2))
    windowed = apply_hounsfield_windowing(image_array, window_center, window_width)
    st.image(windowed, clamp=True, channels="L", caption="Imagem com Janelamento Hounsfield", use_column_width=True)

def apply_hounsfield_windowing(image, center, width):
    min_val = center - width // 2
    max_val = center + width // 2
    windowed = np.clip(image, min_val, max_val)
    windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return windowed

# --- Estatísticas ---
def enhanced_statistics_tab(dicom_data, image_array):
    st.subheader("Análise Estatística")
    flat = image_array.flatten()
    st.metric("Média (HU)", f"{np.mean(flat):.2f}")
    st.metric("Mediana (HU)", f"{np.median(flat):.2f}")
    st.metric("Desvio Padrão", f"{np.std(flat):.2f}")
    st.metric("Mínimo (HU)", f"{np.min(flat):.2f}")
    st.metric("Máximo (HU)", f"{np.max(flat):.2f}")
    fig = px.histogram(flat, nbins=100, title="Histograma de Intensidades HU")
    st.plotly_chart(fig, use_container_width=True)

# --- Análise Técnica ---
def enhanced_technical_analysis_tab(dicom_data, image_array):
    st.subheader("Análise Técnica")
    st.write("Aqui você pode implementar análises técnicas detalhadas, como análise de ruído, compressão, etc.")
    # Exemplo simples: mostrar metadados básicos
    st.markdown("### Metadados DICOM básicos")
    patient_name = getattr(dicom_data, 'PatientName', 'Desconhecido')
    study_date = getattr(dicom_data, 'StudyDate', 'Desconhecido')
    modality = getattr(dicom_data, 'Modality', 'Desconhecido')
    st.write(f"Paciente: {patient_name}")
    st.write(f"Data do Estudo: {study_date}")
    st.write(f"Modalidade: {modality}")

# --- Qualidade ---
def enhanced_quality_metrics_tab(dicom_data, image_array):
    st.subheader("Métricas de Qualidade")
    noise = np.std(image_array - ndimage.median_filter(image_array, size=3))
    st.metric("Ruído Estimado", f"{noise:.2f}")
    contrast = np.percentile(image_array, 75) - np.percentile(image_array, 25)
    st.metric("Contraste Interquartil", f"{contrast:.2f}")

# --- Análise Post-Mortem ---
def enhanced_post_mortem_analysis_tab(dicom_data, image_array):
    st.subheader("Análise Post-Mortem")
    st.info("Funcionalidade avançada de análise post-mortem será implementada aqui.")

# --- RA-Index ---
def enhanced_ra_index_tab(dicom_data, image_array):
    st.subheader("RA-Index")
    st.info("Cálculo e visualização do RA-Index serão implementados aqui.")

# --- Relatórios ---
def enhanced_reporting_tab(dicom_data, image_array, user_data):
    st.subheader("Relatórios")
    st.info("Geração, visualização e download de relatórios serão implementados aqui.")

# --- Exibir informações básicas ---
def display_basic_info(dicom_data, image_array):
    st.header("Informações do Arquivo DICOM")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dimensões", f"{image_array.shape[0]} × {image_array.shape[1]}")
    with col2:
        st.metric("Tipo de Dados", str(image_array.dtype))
    with col3:
        st.metric("Faixa de Valores", f"{image_array.min()} → {image_array.max()}")
    with col4:
        size_kb = len(dicom_data.PixelData) / 1024 if hasattr(dicom_data, 'PixelData') else 0
        st.metric("Tamanho da Imagem", f"{size_kb:.1f} KB")

# --- Main ---
def main():
    st.title("DICOM Autopsy Viewer PRO - Enhanced")

    dicom_data, image_array = upload_and_read_dicom()

    if dicom_data is not None and image_array is not None:
        display_basic_info(dicom_data, image_array)

        tabs = st.tabs([
            "Visualização", "Estatísticas", "Análise Técnica",
            "Qualidade", "Análise Post-Mortem", "RA-Index", "Relatórios"
        ])

        with tabs[0]:
            enhanced_visualization_tab(dicom_data, image_array)
        with tabs[1]:
            enhanced_statistics_tab(dicom_data, image_array)
        with tabs[2]:
            enhanced_technical_analysis_tab(dicom_data, image_array)
        with tabs[3]:
            enhanced_quality_metrics_tab(dicom_data, image_array)
        with tabs[4]:
            enhanced_post_mortem_analysis_tab(dicom_data, image_array)
        with tabs[5]:
            enhanced_ra_index_tab(dicom_data, image_array)
        with tabs[6]:
            enhanced_reporting_tab(dicom_data, image_array, st.session_state.get('user_data', {}))

    else:
        st.info("Por favor, carregue um arquivo DICOM válido na barra lateral para começar.")

if __name__ == "__main__":
    main()
