import numpy as np
import pandas as pd
import os 
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt
import copy

class Image():
    def __init__(self, path):
        self.path = path

    # This function rescales the image pixel values between 0 to 255
    def rescale(self):
        minval = self.image_data.min()
        maxval = self.image_data.max()
        rescaled_image = (255/(maxval-minval))*(self.image_data-minval)
        self.image_data = rescaled_image
    
    """
    This method is used to import image that are of the type .png
    No other type is supported
    """
    def import_image(self):
        self.image_data = plt.imread(self.path)
        
    """
        If the image is more than 2 dimensions, 
        only a slice will be taken for demonstration of histogram equalization
        and the output image will be grayscale
    """
    def import_image_nib_compatible(self):
        if self.path:
            img = nib.load(self.path)
            self.image_data = img.get_fdata()
        else:
            example_filename = os.path.join(data_path,'example4d.nii.gz')
            img = nib.load(self.path()
            self.image_data = img.get_fdata()

    def imhist(self,im):
        # calculates normalized histogram of an image
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

    def cumsum(self,h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

    """
    This method takes image_data as input, 
    which is a numpy array with information about all image pixels
    The input array must be a 2D array only.
    """
    def histeq(self,im):
        #calculate Histogram
	orig_hist = self.imhist(im)
	cdf = np.array(self.cumsum(orig_hist)) #cumulative distribution function
	sk = np.uint8(255 * cdf) #finding transfer function values
	s1, s2 = im.shape
	Y = np.zeros_like(im)
	# applying transfered values for each pixels
        for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	new_hist = self.imhist(Y)
	#return transformed image, original and new histogram and transform function
        return Y,orig_hist,new_hist,sk

    def main()
        return True







    
