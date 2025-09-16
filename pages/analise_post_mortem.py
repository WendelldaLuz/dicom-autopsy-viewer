import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

def simulate_body_cooling(image_array: np.ndarray) -> np.ndarray:
    hu_min, hu_max = np.min(image_array), np.max(image_array)
    normalized = (image_array - hu_min) / (hu_max - hu_min + 1e-8)
    center_y, center_x = np.array(image_array.shape) / 2
    y, x = np.indices(image_array.shape)
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    center_effect = 1 - (distance_from_center / (max_distance + 1e-8))
    simulated_temp = 25 + 10 * normalized + 5 * center_effect
    return simulated_temp

def estimate_pmi_from_cooling(thermal_map: np.ndarray, ambient_temp: float, body_mass: float, clothing: str) -> float:
    core_temp = np.max(thermal_map)
    temp_difference = core_temp - ambient_temp
    mass_factor = body_mass / 70
    clothing_factor = {"Leve": 0.8, "Moderado": 1.0, "Abrigado": 1.2}.get(clothing, 1.0)
    pmi_hours = (temp_difference * mass_factor * clothing_factor) / 0.8
    return max(0, min(pmi_hours, 48))

def enhanced_post_mortem_analysis_tab(dicom_data, image_array: np.ndarray):
    st.header("🔬 Análise Avançada Post-Mortem")

    with st.expander("📚 Referências Científicas (Normas ABNT)"):
        st.markdown("""
        - ALTAIMIRANO, R. Técnicas de imagem aplicadas à tanatologia forense. Revista de Medicina Legal, 2022.
        - MEGO, M. et al. Análise quantitativa de fenômenos cadavéricos por TC multidetectores. J Forensic Sci, 2017.
        - GÓMEZ, H. Avanços na estimativa do intervalo post-mortem por métodos de imagem. Forense Internacional, 2021.
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parâmetros de Entrada")
        ambient_temp = st.slider("Temperatura Ambiente (°C)", 10, 40, 25, help="Temperatura do ambiente onde o corpo foi encontrado")
        body_mass = st.slider("Massa Corporal (kg)", 40, 120, 70, help="Massa estimada do corpo")
        clothing = st.selectbox("Vestuário", ["Leve", "Moderado", "Abrigado"], index=1, help="Tipo de vestuário que o corpo estava usando")

    with col2:
        st.subheader("Informações do Paciente")
        patient_name = getattr(dicom_data, "PatientName", "Desconhecido")
        patient_id = getattr(dicom_data, "PatientID", "Desconhecido")
        study_date = getattr(dicom_data, "StudyDate", "Desconhecido")
        st.markdown(f"**Nome:** {patient_name}")
        st.markdown(f"**ID:** {patient_id}")
        st.markdown(f"**Data do Estudo:** {study_date}")

    st.markdown("---")

    thermal_simulation = simulate_body_cooling(image_array)

    st.subheader("🌡️ Simulação de Distribuição Térmica Corporal")
    fig = go.Figure(data=go.Heatmap(
        z=thermal_simulation,
        colorscale='Jet',
        colorbar=dict(title="Temperatura (°C)"),
        hovertemplate='Temperatura: %{z:.1f}°C<extra></extra>'
    ))
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Calcular Intervalo Post-Mortem (IPM)"):
        ipm_estimate = estimate_pmi_from_cooling(thermal_simulation, ambient_temp, body_mass, clothing)
        st.success(f"Estimativa de IPM: **{ipm_estimate:.1f} horas**")

        hours = np.linspace(0, 48, 100)
        cooling_curve = ambient_temp + (thermal_simulation.max() - ambient_temp) * np.exp(-hours / (ipm_estimate + 1e-8))
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=hours, y=cooling_curve, mode='lines', name='Curva de Resfriamento'))
        fig_curve.update_layout(
            title="Curva Teórica de Resfriamento Corporal",
            xaxis_title="Tempo Post-Mortem (horas)",
            yaxis_title="Temperatura (°C)",
            height=400
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    st.markdown("---")
    st.caption(f"Análise gerada em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
