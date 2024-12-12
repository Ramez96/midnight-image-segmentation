import cv2
import os
import numpy as np
from scipy.signal import convolve2d




def apply_spf(image):
    psf = np.array([
    [0,  0,   1,   4,   6,   4,   100],
    [0,  4,   0,   2,  40,   100,   4],
    [1,  0,  10,  50,  100,  50,  10],
    [4,  2,  50, 100, 7, 5,  50],
    [6, 40,  100, 20, 20, 0,  1],
    [4,  100,  50, 2, 0, 0,  0],
    [100,  4,  10,  4,  1,  0,  0]
], dtype=np.float64)
    psf /= psf.sum()
    height, width, channels = image.shape
    output = np.zeros_like(image, dtype=np.float64)
    for c in range(channels):
        output[:, :, c] = convolve2d(image[:, :, c], psf, mode='same', boundary='wrap')
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output

# gaussian blur
folder_path = "stanford_background_dataset/images"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"stanford_background_dataset/images/{file_name}")
    gaussian_blurred = apply_spf(image)
    cv2.imwrite(f"stanford_background_dataset/preprocessing4/{file_name}", gaussian_blurred)
