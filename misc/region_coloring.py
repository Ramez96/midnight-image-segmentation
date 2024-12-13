import cv2
import numpy as np
import json
from general_function import normalization, denormalization




def display_segments(text_file, path=None):
    # enable both direct text file and path as input 
    # this makes sure, we can compare calculated results with comparison data
    if path != None:
        text_file = np.loadtxt(path, dtype=np.uint8)
    # retrieve colors and map
    json_file = open("colors.json", "r")
    colors = json.load(json_file)
    json_file.close()
    mapped = np.zeros((text_file.shape[0], text_file.shape[1], 3), dtype=np.uint8)    
    for i,row in enumerate(text_file):
        for j,value in enumerate(row):
            mapped[i][j] = colors[str(value)]
    return mapped
    



if __name__ == "__main__":
    img = display_segments(None, path="../dataset/test/labels/1001770.regions.txt")
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
