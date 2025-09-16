import streamlit as st
import pydicom
import tempfile
import os

from analysis import post_mortem, statistics, technical, quality, ra_index, report
from utils import db, security

# Configuração da página
st.set_page_config(
    page_title="DICOM Autopsy Viewer PRO",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("DICOM Autopsy Viewer PRO")
    uploaded_files = st.sidebar.file_uploader(
        "Selecione até 10 arquivos DICOM:",
        type=['dcm', 'dicom'],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.sidebar.error("Selecione no máximo 10 arquivos.")
            uploaded_files = uploaded_files[:10]

        dicom_datasets = []
        image_arrays = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            dicom_data = pydicom.dcmread(tmp_path)
            image_array = dicom_data.pixel_array
            dicom_datasets.append(dicom_data)
            image_arrays.append(image_array)
            os.unlink(tmp_path)

        selected_index = st.sidebar.selectbox(
            "Selecione a imagem para análise",
            options=range(len(dicom_datasets)),
            format_func=lambda i: uploaded_files[i].name
        )

        dicom_data = dicom_datasets[selected_index]
        image_array = image_arrays[selected_index]

        st.title(f"Análise da Imagem: {uploaded_files[selected_index].name}")
        st.image(image_array, use_container_width=True)

        # Chamar abas de análise
        tabs = st.tabs([
            "Visualização", "Estatísticas", "Análise Técnica",
            "Qualidade", "Análise Post-Mortem", "RA-Index", "Relatórios"
        ])

        with tabs[0]:
            technical.enhanced_visualization_tab(dicom_data, image_array)
        with tabs[1]:
            statistics.enhanced_statistics_tab(dicom_data, image_array)
        with tabs[2]:
            technical.enhanced_technical_analysis_tab(dicom_data, image_array)
        with tabs[3]:
            quality.enhanced_quality_metrics_tab(dicom_data, image_array)
        with tabs[4]:
            post_mortem.enhanced_post_mortem_analysis_tab(dicom_data, image_array)
        with tabs[5]:
            ra_index.enhanced_ra_index_tab(dicom_data, image_array)
        with tabs[6]:
            report.enhanced_reporting_tab(dicom_data, image_array)

    else:
        st.info("Carregue até 10 arquivos DICOM na sidebar para começar a análise.")

if __name__ == "__main__":
    main()
