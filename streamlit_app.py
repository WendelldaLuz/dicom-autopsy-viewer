import streamlit as st
import pydicom
import tempfile
import os
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy import ndimage, stats
from skimage import feature

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

# --- Funções auxiliares para análise post-mortem ---
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

def segment_muscle_tissue(image_array):
    muscle_mask = (image_array >= 35) & (image_array <= 55)
    return muscle_mask.astype(float)

def calculate_muscle_density(image_array, muscle_mask):
    muscle_values = image_array[muscle_mask > 0.5]
    return np.mean(muscle_values) if len(muscle_values) > 0 else 0

def estimate_rigor_stage(muscle_density):
    if muscle_density < 40:
        return "inicial"
    elif 40 <= muscle_density < 48:
        return "progressivo"
    elif 48 <= muscle_density < 55:
        return "completo"
    else:
        return "resolucao"

def detect_putrefaction_gases(image_array):
    gas_mask = (image_array <= -100) & (image_array >= -1000)
    gas_map = ndimage.gaussian_filter(gas_mask.astype(float), sigma=3)
    return gas_map

def classify_putrefaction_stage(image_array):
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
    saponification_mask = (image_array >= 100) & (image_array <= 300)
    calcification_mask = image_array >= 500
    conservation_map = np.zeros_like(image_array, dtype=float)
    conservation_map[saponification_mask] = 0.5
    conservation_map[calcification_mask] = 1.0
    return conservation_map

def classify_conservation_type(image_array):
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

# --- Aba Análise Post-Mortem ---
def enhanced_post_mortem_analysis_tab(dicom_data, image_array):
    st.subheader("Análise Avançada Post-Mortem")

    ambient_temp = st.slider("Temperatura Ambiente (°C)", 10, 40, 25)
    body_mass = st.slider("Massa Corporal (kg)", 40, 120, 70)
    clothing = st.select_slider("Vestuário", options=["Leve", "Moderado", "Abrigado"], value="Moderado")

    thermal_simulation = simulate_body_cooling(image_array)
    blood_pooling_map = detect_blood_pooling(image_array)
    muscle_mask = segment_muscle_tissue(image_array)
    muscle_density = calculate_muscle_density(image_array, muscle_mask)
    gas_map = detect_putrefaction_gases(image_array)
    conservation_map = analyze_conservation_features(image_array)

    ipm_algor = estimate_pmi_from_cooling(thermal_simulation, ambient_temp, body_mass, clothing)
    fixation_ratio = assess_livor_fixation(blood_pooling_map)
    rigor_stage = estimate_rigor_stage(muscle_density)
    putrefaction_stage = classify_putrefaction_stage(image_array)
    conservation_type = classify_conservation_type(image_array)

    st.markdown(f"### Estimativa de Intervalo Post-Mortem (IPM)")
    st.metric("Por Algor Mortis (Esfriamento)", f"{ipm_algor:.1f} horas")

    st.markdown(f"### Fenômenos Cadavéricos")
    st.write(f"- Livor Mortis (fixação): {fixation_ratio:.2f}")
    st.write(f"- Rigor Mortis (estágio): {rigor_stage}")
    st.write(f"- Putrefação (estágio): {putrefaction_stage}")
    st.write(f"- Fenômenos Conservadores: {conservation_type}")

    st.markdown("### Mapas de Análise")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure(data=go.Heatmap(z=thermal_simulation, colorscale='jet', showscale=True))
        fig1.update_layout(title="Simulação de Distribuição Térmica")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = go.Figure(data=go.Heatmap(z=blood_pooling_map, colorscale='hot', showscale=True))
        fig2.update_layout(title="Mapa de Acúmulo Sanguíneo (Livor Mortis)")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = go.Figure(data=go.Heatmap(z=muscle_mask, colorscale='gray', showscale=False))
        fig3.update_layout(title="Segmentação de Tecido Muscular")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = go.Figure(data=go.Heatmap(z=gas_map, colorscale='viridis', showscale=True))
        fig4.update_layout(title="Mapa de Gases de Putrefação")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Observações")
    notes = []
    if ipm_algor > 24:
        notes.append("Esfriamento sugere IPM prolongado.")
    if fixation_ratio > 0.7:
        notes.append("Hipóstase fixa indica corpo não movido após 12h.")
    if putrefaction_stage == "gasoso":
        notes.append("Presença significativa de gases de putrefação.")
    if conservation_type != "none":
        notes.append(f"Evidências de {conservation_type} detectadas.")
    if notes:
        for note in notes:
            st.write(f"- {note}")
    else:
        st.write("Nenhuma observação adicional.")

# --- Aba RA-Index simplificada ---
def enhanced_ra_index_tab(dicom_data, image_array):
    st.subheader("RA-Index - Análise de Risco Aprimorada")
    # Exemplo simples: risco baseado em intensidade média regional
    grid_size = 8
    h, w = image_array.shape
    h_step, w_step = h // grid_size, w // grid_size
    ra_values = []
    for i in range(grid_size):
        for j in range(grid_size):
            region = image_array[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            mean_intensity = np.mean(region)
            ra_values.append(mean_intensity)
    ra_array = np.array(ra_values).reshape(grid_size, grid_size)
    fig = go.Figure(data=go.Heatmap(z=ra_array, colorscale='RdYlBu_r', showscale=True))
    fig.update_layout(title="Mapa RA-Index (média regional de HU)")
    st.plotly_chart(fig, use_container_width=True)

# --- Aba Relatórios (placeholder) ---
def enhanced_reporting_tab(dicom_data, image_array, user_data):
    st.subheader("Relatórios")
    st.info("Funcionalidade de geração e download de relatórios será implementada aqui.")

# --- Outras abas básicas ---
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

def enhanced_technical_analysis_tab(dicom_data, image_array):
    st.subheader("Análise Técnica")
    st.markdown("### Metadados DICOM básicos")
    patient_name = getattr(dicom_data, 'PatientName', 'Desconhecido')
    study_date = getattr(dicom_data, 'StudyDate', 'Desconhecido')
    modality = getattr(dicom_data, 'Modality', 'Desconhecido')
    st.write(f"Paciente: {patient_name}")
    st.write(f"Data do Estudo: {study_date}")
    st.write(f"Modalidade: {modality}")

def enhanced_quality_metrics_tab(dicom_data, image_array):
    st.subheader("Métricas de Qualidade")
    noise = np.std(image_array - ndimage.median_filter(image_array, size=3))
    st.metric("Ruído Estimado", f"{noise:.2f}")
    contrast = np.percentile(image_array, 75) - np.percentile(image_array, 25)
    st.metric("Contraste Interquartil", f"{contrast:.2f}")

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
