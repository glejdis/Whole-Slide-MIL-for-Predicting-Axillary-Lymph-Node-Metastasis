# delete images with an average intensity threshold : greater than 180

import os
import cv2
import numpy as np

path='./dataset/original_WSI_patches'
for folder in os.listdir(path):	
	for file in os.listdir(os.path.join(path,folder)):
	    if (np.mean(cv2.imread(os.path.join(path,folder,file)))>180):
		print(file)
		os.remove(os.path.join(path,folder,file)) 