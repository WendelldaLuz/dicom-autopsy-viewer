import streamlit as st

def enhanced_technical_analysis_tab(dicom_data, image_array):
    st.subheader("Análise Técnica Forense Avançada")
    st.write("Aqui você pode implementar análise técnica detalhada, incluindo metadados, integridade e artefatos.")
    if hasattr(dicom_data, 'PatientName'):
        st.write(f"Paciente: {dicom_data.PatientName}")
    else:
        st.write("Metadados do paciente não disponíveis.")

def show(dicom_data, image_array):
    st.title("Análise Técnica")
    enhanced_technical_analysis_tab(dicom_data, image_array)
