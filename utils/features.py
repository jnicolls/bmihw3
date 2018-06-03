'''
	features.py
'''

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert
from sklearn.decomposition import PCA
from skimage.measure import regionprops
from file_ops import *


def get_props(data): 
    ret,thresh_img = cv2.threshold(data,0,255,cv2.THRESH_BINARY)
    thresh_props =  regionprops(thresh_img.astype(int))
    thresh_area = thresh_props[0].area
    thresh_perimeter = thresh_props[0].perimeter
    chull = convex_hull_image(thresh_img)
    props = regionprops(chull.astype(int))
    chull_area = props[0].area
    chull_perimeter = props[0].perimeter 
    convexity = chull_perimeter/thresh_perimeter
    solidity = thresh_area/chull_area
    return convexity, solidity, thresh_props[0].extent, thresh_props[0].major_axis_length, thresh_props[0].minor_axis_length
    

def reduce_dimensionality(raw_data, new_dims=3):
    '''
        input:
            * raw_data, the raw matrix that will be reduced in dimensionality 
        output:
            * the dimensionality-reduced data 
        
    '''
    print("preparing to reduce dimensionality")
    pca = PCA()
    pca.fit(raw_data)
    print(">>> variance explained by each principal component")
    print(pca.explained_variance_ratio_)  
    print(">>> the first principal component")
    print(pca.components_[0])
    reduced = pca.transform(raw_data)[:,:new_dims]
    return reduced


def mean_center_normalize(data):
    data -= np.mean(data, axis = 0)
    data /= np.std(data, axis = 0)
    return data
