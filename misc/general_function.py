import numpy as np

def normalization(image):
    return (image / 255.0).astype(np.float32)

def denormalization(image):
    return (image * 255.0).astype(np.uint8)