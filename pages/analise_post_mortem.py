import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def simulate_body_cooling(image_array):
    # Simulação simplificada
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

def enhanced_post_mortem_analysis_tab(dicom_data, image_array):
    st.subheader("Análise Avançada de Períodos Post-Mortem")
    ambient_temp = st.slider("Temperatura Ambiente (°C)", 10, 40, 25)
    body_mass = st.slider("Massa Corporal (kg)", 40, 120, 70)
    clothing = st.select_slider("Vestuário", options=["Leve", "Moderado", "Abrigado"], value="Moderado")

    thermal_simulation = simulate_body_cooling(image_array)
    fig = go.Figure(data=go.Heatmap(
        z=thermal_simulation,
        colorscale='jet',
        showscale=True,
        hovertemplate='Temperatura: %{z:.1f}°C<extra></extra>'
    ))
    fig.update_layout(title="Simulação de Distribuição Térmica Corporal", height=400)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Estimar Intervalo Post-Mortem (IPM)"):
        ipm_estimate = estimate_pmi_from_cooling(thermal_simulation, ambient_temp, body_mass, clothing)
        st.metric("Intervalo Post-Mortem Estimado", f"{ipm_estimate:.1f} horas")
