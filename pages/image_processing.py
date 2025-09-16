import numpy as np

def calcular_snr(image_array):
    """
    Calcula o Signal-to-Noise Ratio (SNR) da imagem.
    """
    mean = np.mean(image_array)
    std = np.std(image_array)
    return mean / (std + 1e-6)
