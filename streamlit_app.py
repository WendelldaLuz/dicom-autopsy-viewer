import streamlit as st
import pydicom
import tempfile
import os

# Importar módulos das páginas (ajuste conforme sua estrutura)
from pages import (
    analise_post_mortem,
    analise_tecnica,
    estatisticas,
    qualidade,
    ra_index,
    relatorios,
    visualizacao
)

# Configuração da página
st.set_page_config(
    page_title="DICOM Autopsy Viewer PRO",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_dicom_files(uploaded_files):
    """
    Carrega múltiplos arquivos DICOM e retorna listas de datasets e arrays de imagem.
    """
    dicom_datasets = []
    image_arrays = []
    for uploaded_file in uploaded_files:
        # Cria arquivo temporário para leitura com pydicom
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        try:
            dicom_data = pydicom.dcmread(tmp_path)
            image_array = dicom_data.pixel_array
        except Exception as e:
            st.error(f"Erro ao ler arquivo DICOM {uploaded_file.name}: {e}")
            continue
        finally:
            os.unlink(tmp_path)  # Remove arquivo temporário

        dicom_datasets.append(dicom_data)
        image_arrays.append(image_array)

    return dicom_datasets, image_arrays

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

        dicom_datasets, image_arrays = load_dicom_files(uploaded_files)

        if not dicom_datasets:
            st.warning("Nenhum arquivo DICOM válido foi carregado.")
            return

        selected_index = st.sidebar.selectbox(
            "Selecione a imagem para análise",
            options=range(len(dicom_datasets)),
            format_func=lambda i: uploaded_files[i].name
        )

        dicom_data = dicom_datasets[selected_index]
        image_array = image_arrays[selected_index]

        st.title(f"Análise da Imagem: {uploaded_files[selected_index].name}")

        # Visualização da imagem DICOM
        st.image(image_array, use_container_width=True)

        # Abas para diferentes análises
        tabs = st.tabs([
            "Visualização",
            "Estatísticas",
            "Análise Técnica",
            "Qualidade",
            "Análise Post-Mortem",
            "RA-Index",
            "Relatórios"
        ])

        with tabs[0]:
            visualizacao.visualizacao_tab(dicom_data, image_array)

        with tabs[1]:
            estatisticas.estatisticas_tab(dicom_data, image_array)

        with tabs[2]:
            analise_tecnica.analise_tecnica_tab(dicom_data, image_array)

        with tabs[3]:
            qualidade.qualidade_tab(dicom_data, image_array)

        with tabs[4]:
            analise_post_mortem.enhanced_post_mortem_analysis_tab(dicom_data, image_array)

        with tabs[5]:
            ra_index.ra_index_tab(dicom_data, image_array)

        with tabs[6]:
            relatorios.enhanced_reporting_tab(dicom_data, image_array)

    else:
        st.info("Carregue até 10 arquivos DICOM na sidebar para começar a análise.")

if __name__ == "__main__":
    main()
