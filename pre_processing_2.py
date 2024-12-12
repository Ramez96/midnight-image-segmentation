import cv2
import os
import numpy as np

def gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 0.6)

def laplacian_of_gaussian(image, kernel_size=9, sigma=0.5):
    log_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3): 
        blurred = cv2.GaussianBlur(image[..., i], (kernel_size, kernel_size), sigma)
        log_image[..., i] = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.clip(log_image, 0, 255).astype(np.uint8)

def add_random_pixels(image, num_pixels):
    height, width, channels = image.shape
    noisy_image = image.copy()
    random_y = np.random.randint(0, height, num_pixels)
    random_x = np.random.randint(0, width, num_pixels)

    for y, x in zip(random_y, random_x):
        noisy_image[y, x] = np.random.randint(0, 256, size=(3,)) 

    return noisy_image

def sharpen_image(image):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with 3 channels.")
    sharpen_kernel = np.array([[ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]])
    sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)

    return sharpened_image



folder_path = "dataset/test/preprocessing1"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for file_name in file_list:
    print(file_name)
    image = cv2.imread(f"dataset/test/preprocessing1/{file_name}")
    
    #filterd = add_random_pixels(image,10000)
    #filterd = laplacian_of_gaussian(image)
    filterd = sharpen_image(image)
    cv2.imwrite(f"dataset/test/preprocessing35/{file_name}", filterd)
