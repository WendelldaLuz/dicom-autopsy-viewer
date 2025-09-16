import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import ndimage

def calculate_forensic_quality(image_array: np.ndarray) -> dict:
    """Calcula métricas de qualidade forense da imagem."""
    grad_x = np.gradient(image_array, axis=1)
    grad_y = np.gradient(image_array, axis=0)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    sharp_edges = gradient_magnitude > np.percentile(gradient_magnitude, 95)
    edge_sharpness = np.mean(gradient_magnitude[sharp_edges]) if np.any(sharp_edges) else 0

    contrast = np.percentile(image_array, 75) - np.percentile(image_array, 25)
    max_contrast = np.max(image_array) - np.min(image_array)
    detectable_contrast = contrast / max_contrast if max_contrast > 0 else 0

    resolution_score = min(1.0, edge_sharpness / 5.0)

    suitability_identification = min(1.0, resolution_score * 0.7 + detectable_contrast * 0.3)
    suitability_analysis = min(1.0, resolution_score * 0.5 + detectable_contrast * 0.5)
    suitability_documentation = min(1.0, resolution_score * 0.3 + detectable_contrast * 0.7)

    limitations = []
    if resolution_score < 0.5:
        limitations.append("Resolução insuficiente para análise detalhada")
    if detectable_contrast < 0.2:
        limitations.append("Contraste limitado pode dificultar a análise")

    overall_quality = (suitability_identification + suitability_analysis + suitability_documentation) / 3

    return {
        'overall_quality': overall_quality,
        'effective_resolution': edge_sharpness,
        'detectable_contrast': detectable_contrast,
        'suitability_identification': suitability_identification,
        'suitability_analysis': suitability_analysis,
        'suitability_documentation': suitability_documentation,
        'limitations': limitations
    }

def detect_artifacts(image_array: np.ndarray) -> dict:
    """Detecta artefatos comuns na imagem."""
    artifact_map = np.zeros_like(image_array, dtype=bool)

    # Ruído excessivo
    noise_std = np.std(image_array - ndimage.median_filter(image_array, size=3))
    noise_detected = noise_std > 20
    noise_mask = (np.abs(image_array - ndimage.median_filter(image_array, size=3)) > noise_std)
    artifact_map = np.logical_or(artifact_map, noise_mask)

    # Artefatos de movimento (simples detecção por gradiente)
    grad_x = np.gradient(image_array, axis=1)
    grad_y = np.gradient(image_array, axis=0)
    motion_mask = (np.abs(grad_x) + np.abs(grad_y)) > np.percentile(np.abs(grad_x) + np.abs(grad_y), 95)
    motion_detected = np.any(motion_mask)
    artifact_map = np.logical_or(artifact_map, motion_mask)

    # Artefatos metálicos (valores muito altos)
    metal_mask = image_array > 1000
    metal_detected = np.any(metal_mask)
    artifact_map = np.logical_or(artifact_map, metal_mask)

    artifacts = []
    if noise_detected:
        artifacts.append({'type': 'Ruído', 'description': 'Ruído excessivo detectado', 'severity': 'médio'})
    if motion_detected:
        artifacts.append({'type': 'Movimento', 'description': 'Artefatos de movimento detectados', 'severity': 'alto'})
    if metal_detected:
        artifacts.append({'type': 'Metal', 'description': 'Artefatos metálicos detectados', 'severity': 'alto'})

    affected_area = np.sum(artifact_map) / artifact_map.size * 100

    return {
        'artifacts': artifacts,
        'artifact_map': artifact_map.astype(float),
        'affected_area': affected_area
    }

def enhanced_quality_metrics_tab(dicom_data, image_array):
    st.header(" Métricas Avançadas de Qualidade de Imagem")

    quality_metrics = calculate_forensic_quality(image_array)
    artifacts_report = detect_artifacts(image_array)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Métricas de Qualidade")
        st.metric("Qualidade Geral", f"{quality_metrics['overall_quality']:.0%}")
        st.metric("Resolução Efetiva", f"{quality_metrics['effective_resolution']:.2f}")
        st.metric("Contraste Detectável", f"{quality_metrics['detectable_contrast']:.2f}")

    with col2:
        st.subheader("Adequação Forense")
        st.metric("Para Identificação", f"{quality_metrics['suitability_identification']:.0%}")
        st.metric("Para Análise", f"{quality_metrics['suitability_analysis']:.0%}")
        st.metric("Para Documentação", f"{quality_metrics['suitability_documentation']:.0%}")

    with col3:
        st.subheader("Limitações")
        if quality_metrics['limitations']:
            for lim in quality_metrics['limitations']:
                st.warning(lim)
        else:
            st.success("Sem limitações significativas")

    st.markdown("---")
    st.subheader("🛑 Detecção de Artefatos")
    if artifacts_report['artifacts']:
        for art in artifacts_report['artifacts']:
            severity_color = {"alto": "error", "médio": "warning", "baixo": "info"}.get(art['severity'], "info")
            st.markdown(f"- **{art['type']}**: {art['description']} (Severidade: {art['severity'].capitalize()})")
    else:
        st.success("Nenhum artefato significativo detectado")

    if artifacts_report['artifact_map'] is not None:
        fig = px.imshow(artifacts_report['artifact_map'], color_continuous_scale='hot', title="Mapa de Artefatos")
        st.plotly_chart(fig, use_container_width=True)

    st.metric("Área Afetada por Artefatos", f"{artifacts_report['affected_area']:.1f}%")
