import cv2
import os
import numpy as np

def gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 0.6)

def laplacian_of_gaussian(image, kernel_size=9, sigma=0.5):
    log_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # Process each channel separately
        blurred = cv2.GaussianBlur(image[..., i], (kernel_size, kernel_size), sigma)
        log_image[..., i] = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.clip(log_image, 0, 255).astype(np.uint8)

def add_random_pixels(image, num_pixels):
    height, width, channels = image.shape

    # Create a copy of the image to avoid modifying the original
    noisy_image = image.copy()

    # Randomly choose indices for the pixels to modify
    random_y = np.random.randint(0, height, num_pixels)
    random_x = np.random.randint(0, width, num_pixels)

    # Modify the pixels
    for y, x in zip(random_y, random_x):
        noisy_image[y, x] = np.random.randint(0, 256, size=(3,))  # Random RGB value

    return noisy_image

folder_path = "stanford_background_dataset/images"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"stanford_background_dataset/images/{file_name}")
    
    filterd = add_random_pixels(image,10000)
    filterd = gaussian(filterd)
    cv2.imwrite(f"stanford_background_dataset/preprocessing2/{file_name}", filterd)
