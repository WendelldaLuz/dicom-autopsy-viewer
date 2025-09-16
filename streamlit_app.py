import streamlit as st
import pydicom
import tempfile
import os
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import ndimage
from skimage import feature
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import skew, kurtosis

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

# --- Estatísticas Avançadas ---
def advanced_statistics(image_array):
    flat = image_array.flatten()
    stats_dict = {}

    # Estatísticas básicas
    stats_dict['mean'] = np.mean(flat)
    stats_dict['median'] = np.median(flat)
    stats_dict['std'] = np.std(flat)
    stats_dict['min'] = np.min(flat)
    stats_dict['max'] = np.max(flat)
    stats_dict['skewness'] = skew(flat)
    stats_dict['kurtosis'] = kurtosis(flat)

    # Histogramas
    hist, bin_edges = np.histogram(flat, bins=100, density=True)
    cdf = np.cumsum(hist) * np.diff(bin_edges)

    # Estatísticas regionais (dividir em 4x4 blocos)
    grid_size = 4
    h, w = image_array.shape
    h_step, w_step = h // grid_size, w // grid_size
    regional_means = []
    regional_stds = []
    for i in range(grid_size):
        for j in range(grid_size):
            region = image_array[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            regional_means.append(np.mean(region))
            regional_stds.append(np.std(region))
    stats_dict['regional_mean_mean'] = np.mean(regional_means)
    stats_dict['regional_mean_std'] = np.std(regional_means)
    stats_dict['regional_std_mean'] = np.mean(regional_stds)
    stats_dict['regional_std_std'] = np.std(regional_stds)

    # Textura GLCM (níveis reduzidos para 64 para performance)
    image_64 = np.uint8((image_array - stats_dict['min']) / (stats_dict['max'] - stats_dict['min']) * 63)
    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(image_64, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)

    contrast = greycoprops(glcm, 'contrast').mean()
    dissimilarity = greycoprops(glcm, 'dissimilarity').mean()
    homogeneity = greycoprops(glcm, 'homogeneity').mean()
    energy = greycoprops(glcm, 'energy').mean()
    correlation = greycoprops(glcm, 'correlation').mean()
    ASM = greycoprops(glcm, 'ASM').mean()

    stats_dict.update({
        'glcm_contrast': contrast,
        'glcm_dissimilarity': dissimilarity,
        'glcm_homogeneity': homogeneity,
        'glcm_energy': energy,
        'glcm_correlation': correlation,
        'glcm_ASM': ASM
    })

    # Guardar histogramas para plotagem
    stats_dict['histogram'] = (hist, bin_edges)
    stats_dict['cdf'] = (cdf, bin_edges[:-1])

    return stats_dict

def enhanced_statistics_tab(dicom_data, image_array):
    st.subheader("Análise Estatística Avançada")

    stats_dict = advanced_statistics(image_array)

    # Mostrar métricas principais
    col1, col2, col3 = st.columns(3)
    col1.metric("Média (HU)", f"{stats_dict['mean']:.2f}")
    col1.metric("Mediana (HU)", f"{stats_dict['median']:.2f}")
    col1.metric("Desvio Padrão", f"{stats_dict['std']:.2f}")
    col2.metric("Assimetria", f"{stats_dict['skewness']:.2f}")
    col2.metric("Curtose", f"{stats_dict['kurtosis']:.2f}")
    col2.metric("Mínimo (HU)", f"{stats_dict['min']:.2f}")
    col3.metric("Máximo (HU)", f"{stats_dict['max']:.2f}")
    col3.metric("GLCM Contraste", f"{stats_dict['glcm_contrast']:.2f}")
    col3.metric("GLCM Homogeneidade", f"{stats_dict['glcm_homogeneity']:.2f}")

    # Histogramas
    hist, bin_edges = stats_dict['histogram']
    cdf, cdf_bins = stats_dict['cdf']

    fig_hist = px.bar(x=bin_edges[:-1], y=hist, labels={'x':'Intensidade HU', 'y':'Densidade'}, title="Histograma de Intensidades")
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_cdf = px.line(x=cdf_bins, y=cdf, labels={'x':'Intensidade HU', 'y':'CDF'}, title="Função de Distribuição Acumulada (CDF)")
    st.plotly_chart(fig_cdf, use_container_width=True)

    # Boxplot regional
    regional_means = []
    regional_stds = []
    grid_size = 4
    h, w = image_array.shape
    h_step, w_step = h // grid_size, w // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            region = image_array[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            regional_means.append(np.mean(region))
            regional_stds.append(np.std(region))

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=regional_means, name='Médias Regionais'))
    fig_box.add_trace(go.Box(y=regional_stds, name='Desvios Regionais'))
    fig_box.update_layout(title="Boxplot de Estatísticas Regionais")
    st.plotly_chart(fig_box, use_container_width=True)

    # Salvar no session_state para relatório
    st.session_state['advanced_stats'] = stats_dict

# --- Função principal do app ---
def main():
    dicom_data, image_array = upload_and_read_dicom()
    if dicom_data is not None and image_array is not None:
        st.title("DICOM Autopsy Viewer PRO")
        enhanced_visualization_tab(dicom_data, image_array)
        enhanced_statistics_tab(dicom_data, image_array)
        # Aqui você pode chamar outras abas e funções conforme desejar

if __name__ == "__main__":
    main()
