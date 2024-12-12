import cv2
import os
import numpy as np
from scipy.signal import wiener


def so_noisy(image, mean=0, var=0.001):
    # Create an empty image to store the Wiener filtered result
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = np.clip(image + gauss * 255, 0, 255).astype(np.uint8)
    return noisy_image


# gaussian blur
folder_path = "../dataset/test/images"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"{folder_path}/{file_name}")
    filtered_image = so_noisy(image)
    cv2.imwrite(f"../dataset/test/preprocessing3/{file_name}", filtered_image)

folder_path = "../dataset/train/images"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"{folder_path}/{file_name}")
    filtered_image = so_noisy(image)
    cv2.imwrite(f"../dataset/train/preprocessing3/{file_name}", filtered_image)

folder_path = "../dataset/val/images"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"{folder_path}/{file_name}")
    filtered_image = so_noisy(image)
    cv2.imwrite(f"../dataset/val/preprocessing3/{file_name}", filtered_image)
