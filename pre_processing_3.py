import cv2
import os
import numpy as np
from scipy.signal import wiener


def apply_wiener_rgb(image, mysize=1, noise=1e-3):
    # Create an empty image to store the Wiener filtered result
    wiener_image = np.zeros_like(image, dtype=np.float32)

    # Apply the Wiener filter to each channel
    for i in range(3):  # 0=Red, 1=Green, 2=Blue
        channel = image[..., i]
        # Apply SciPy Wiener filter with noise regularization to avoid division by zero
        wiener_image[..., i] = wiener(channel, mysize=mysize, noise=noise)

    # Convert NaNs and Infs to valid values (e.g., zero or max 255)
    wiener_image = np.nan_to_num(wiener_image, nan=0.0, posinf=255, neginf=0)

    # Clip the values to valid range [0, 255] and convert back to uint8
    wiener_image = np.clip(wiener_image, 0, 255).astype(np.uint8)

    return wiener_image


def pixelate_image(image, pixel_size):
    """
    Pixelates the RGB image by reducing its resolution and scaling it back.
    
    Parameters:
        image (numpy.ndarray): The input image as a 3D numpy array (H, W, C).
        pixel_size (int): The size of the pixel blocks.

    Returns:
        numpy.ndarray: The pixelated RGB image.
    """
    # Get the dimensions of the image
    height, width, channels = image.shape
    
    # Ensure the dimensions are divisible by pixel_size
    new_height = (height // pixel_size) * pixel_size
    new_width = (width // pixel_size) * pixel_size
    image_cropped = image[:new_height, :new_width]
    
    # Downsample the image
    small_height = new_height // pixel_size
    small_width = new_width // pixel_size
    downsampled = image_cropped.reshape(small_height, pixel_size, small_width, pixel_size, channels).mean(axis=(1, 3))
    
    # Upscale back to the original size
    pixelated = np.repeat(np.repeat(downsampled, pixel_size, axis=0), pixel_size, axis=1)
    
    return pixelated


# gaussian blur
folder_path = "stanford_background_dataset/images"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"stanford_background_dataset/images/{file_name}")
    gaussian_blurred = pixelate_image(image, 3)
    cv2.imwrite(f"stanford_background_dataset/preprocessing3/{file_name}", gaussian_blurred)
