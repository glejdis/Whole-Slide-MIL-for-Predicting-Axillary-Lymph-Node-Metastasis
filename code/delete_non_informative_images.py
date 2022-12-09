""" Delete the images with Shannon entropy less than the threshold """
from PIL import Image
from skimage import io

path='./dataset/original_WSI_patches'
for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path,folder)):
        img = io.imread(os.path.join(path,folder,file), as_gray=True)
        if skimage.measure.shannon_entropy(img)<8.5:
            print(os.path.join(path,folder,file))
            os.remove(os.path.join(path,folder,file)) 