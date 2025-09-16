import streamlit as st
import numpy as np

def visualizacao_tab(dicom_data, image_array):
    st.title("Visualização Avançada DICOM")

    st.write("Visualização avançada da imagem DICOM com janelamento e ferramentas colorimétricas.")

    # Controle de janelamento (window center e window width)
    window_center = st.slider("Window Center (Centro da Janela)", min_value=0, max_value=255, value=128)
    window_width = st.slider("Window Width (Largura da Janela)", min_value=1, max_value=255, value=128)

    # Aplicar janelamento na imagem
    def apply_window(image, center, width):
        lower = center - (width / 2)
        upper = center + (width / 2)
        windowed = np.clip(image, lower, upper)
        windowed = ((windowed - lower) / width) * 255.0
        return windowed.astype(np.uint8)

    windowed_image = apply_window(image_array, window_center, window_width)

    # Opção para aplicar mapa de cores
    colormap_option = st.selectbox("Aplicar Mapa de Cores", ["Nenhum", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"])

    import matplotlib.pyplot as plt

    if colormap_option != "Nenhum":
        cmap = plt.get_cmap(colormap_option.lower())
        colored_image = cmap(windowed_image)
        # converter RGBA para RGB e escalar para 0-255
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        st.image(colored_image, caption=f"Imagem com mapa de cores: {colormap_option}", use_column_width=True)
    else:
        st.image(windowed_image, caption="Imagem com janelamento aplicado", use_column_width=True)

    # Mostrar metadados básicos se disponíveis
    if dicom_data and isinstance(dicom_data, dict):
        st.subheader("Metadados DICOM")
        for key, value in dicom_data.get("metadata", {}).items():
            st.write(f"**{key}:** {value}")

# Função legacy para compatibilidade, caso seja usada em outro lugar
def show(dicom_data, image_array):
    visualizacao_tab(dicom_data, image_array)
