import cv2
import os

def gaussian(image):
    return cv2.GaussianBlur(image, (11, 11), 2.0)

# gaussian blur
folder_path = "../dataset/test/preprocessing3"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"{folder_path}/{file_name}")
    filtered_image = gaussian(image)
    cv2.imwrite(f"../dataset/test/preprocessing1/{file_name}", filtered_image)

# gaussian blur
folder_path = "../dataset/val/preprocessing3"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"{folder_path}/{file_name}")
    filtered_image = gaussian(image)
    cv2.imwrite(f"../dataset/val/preprocessing1/{file_name}", filtered_image)

# gaussian blur
folder_path = "../dataset/train/preprocessing3"
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for file_name in file_list:
    image = cv2.imread(f"{folder_path}/{file_name}")
    filtered_image = gaussian(image)
    cv2.imwrite(f"../dataset/train/preprocessing1/{file_name}", filtered_image)
