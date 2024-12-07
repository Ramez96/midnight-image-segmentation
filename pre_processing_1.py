import cv2
import os

def gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 2.0)

# gaussian blur
folder_path = "stanford_background_dataset/images"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"stanford_background_dataset/images/{file_name}")
    gaussian_blurred = gaussian(image)
    cv2.imwrite(f"stanford_background_dataset/preprocessing1/{file_name}", gaussian_blurred)
