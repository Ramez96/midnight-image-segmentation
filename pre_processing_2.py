import cv2
import os
import numpy as np



def laplacian_of_gaussian(image, kernel_size=9, sigma=0.5):
    log_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # Process each channel separately
        blurred = cv2.GaussianBlur(image[..., i], (kernel_size, kernel_size), sigma)
        log_image[..., i] = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.clip(log_image, 0, 255).astype(np.uint8)


folder_path = "stanford_background_dataset/images"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"stanford_background_dataset/images/{file_name}")
    filterd = laplacian_of_gaussian(image)
    cv2.imwrite(f"stanford_background_dataset/preprocessing2/{file_name}", filterd)
