import numpy as np

def calcular_snr(image_array):
    mean = np.mean(image_array)
    std = np.std(image_array)
    return mean / (std + 1e-6)

def normalizar_imagem(image_array):
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    return (image_array - min_val) / (max_val - min_val)
