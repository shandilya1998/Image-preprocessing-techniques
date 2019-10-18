import numpy as np
import pandas as pd
import os 
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt
import copy

class Image():
    def __init__(self, path=False):
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
            img = nib.load(example_filename)
            self.image_data = img.get_fdata()

    def imhist(self,im):
        # calculates normalized histogram of an image
        m, n = im.shape
        h = [0.0] * 256
        for i in range(m):
            for j in range(n):
                h[int(im[i, j])]+=1
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
                Y[i, j] = sk[int(im[i, j])]
            new_hist = self.imhist(Y)
    	#return transformed image, original and new histogram and transform function
        return Y,orig_hist,new_hist,sk

    def main(self):
        self.image_data = self.image_data.astype('int64')
        if len(self.image_data.shape)==3:
            print('3')
            fig, axs = plt.subplots(self.image_data.shape[2], sharex=True, sharey=True)
            for i in range(self.image_data.shape[2]):
                Y, orig_hist, new_hist, sk = self.histeq(self.image_data[:,:,i])
                axs[i].imshow(Y)
            plt.show()
        elif len(self.image_data.shape) == 4:
            print('4')
            fig, axs = plt.subplots(self.image_data.shape[2], self.image_data.shape[3], sharex = True, sharey = True)
            k = 0
            for i in range(self.image_data.shape[2]):
                for j in range(self.image_data.shape[3]):
                        Y, orig_hist, new_hist, sk = self.histeq(self.image_data[:,:,i,j])
                        axs[k].imshow(Y)
                        k+=1
                plt.show()

class gradient():
    def __init__(self,path = None):
        self.path = path
        self.img = nib.load(self.path)
        self.image_data = self.img.get_fdata()

    def grad_x(self, im, formulation):
        """
            im - np.ndarray() n*m
            formulation - Choice of formulation of gradient: 0 or 1
            Padding applied : 1
            Output : np.ndarray() (n)*(m)
        """
        grad_x = np.zeros((im.shape[0], im.shape[1])) 
        temp = np.zeros((im.shape[0] + 1, im.shape[1] + 1))
        temp[0:temp.shape[0] - 1, 1 : temp.shape[1]-1] = im
        for i in range(0, temp.shape[0]) - 1:
            for j in range(0, temp.shape[1] - 1):
                if formulation != 0:
                    grad_x[i][j] = 0.5*(temp[i][j + 1]-temp[i][j]+temp[i + 1][j + 1]-temp[i + 1][j])
                else:
                    grad[i][j] = temp[i + 1][j]-temp[i][j]

    
    def grad_y(self, im, formulation = 0):
        """
            im - np.ndarray() n*m
            formulation - Choice of formulation of gradient: 0 or 1
            Padding applied : 1
            Output : np.ndarray() (n)*(m)
        """
        grad_y = np.zeros((im.shape[0], im.shape[1]))
        temp = np.zeros((im.shape[0] + 1, im.shape[1] + 1))
        temp[0 : temp.shape[0] - 1, 0 : temp.shape[1] - 1] = im
        for i in range(0, temp.shape[0] - 1):
            for j in range(0, temp.shape[1] - 1):
                if formulation != 0:
                    grad_y[i][j] = 0.5*(temp[i][j + 1]-temp[i][j]+temp[i + 1][j + 1]-temp[i + 1][j])
                else:
                    grad_y[i][j] = im[i + 1][j]-im[i][j]

    def grad_2_x(self, im):
        """
            im - np,ndarray() n*m
            Padding applied - 1
            Output : np.ndarray() n*m
        """
        grad_2_x = np.zeros(im.shape[0],im.shape[1])
        temp = np.zeros(im.shape[0]+2, im.shape[1]+2)
        temp[1 : temp.shape[0] - 1][1 : temp.shape[1] - 1] = im
        for i in range(1, temp.shape[0] - 1):
            for j in range(1, temp.shape[1] - 1):
                grad_2_x[i][j] = temp[i + 1][j] + temp[i - 1][j] - 2*temp[i][j]

    def grad_2_y(self, im):
        """
            im - np.ndarray() n*m
            Padding applied - 1
            Output - np.ndarray() n*m
        """
        grad_2_y = np.ndarray(im.shape[0], im.shape[1])
        temp = np.zeros(im.shape + 2, im.shape[1] + 2)
        temp[1 : temp.shape[0] - 1][1 : temp.shape[1]-]  = im
            for i in range(1, temp.shape[0]-1):
                for j in range(1, temp.shape[1]-1):
                    grad_2_y = temp[i][j + 1] + temp[i][j - 1] - 2*temp[i][j]


class convolution():
    def __init__(self, path = None, fltr = np.asarray([1]), padding = 0, stride = 1):
        self.path = path
        self.fltr = fltr
        self.padding = padding
        self.stride = stride

    def convolve_with_impulse_response(self, im, magnitude = 1):
        """
            convolves input image with an impulse response of magnitude
            im - np.ndarray() n*m
            Output - np.ndarray() n*m
        """
        y = np.zeros(im.shape[0], im.shape[1]) # [y] = [im]*[delta]
        for i in range(0, im.shape[0] - 1):
            for j in range(0, im.shape[1] - 1):
                y[i][j] = im[i][j]*magnitude
        return y

    def convolve(self, im):
        """
            convolves im with filter
            im - np.ndarray() n*m
            filter - np.ndarray() x*y
            Output - (((n - filter.shape[0] + 2*padding)/stride) + 1)*(((m - filter.shape[1] +2*padding)/stride) + 1)
        """
        y = np.zeros(((im.shape[0] - self.fltr.shape[0] + 2*self.padding)/self.stride)+1, ((im.shape[1] - self.fltr.shape[1] + 2*self.padding)/stride)+1)

        for i in range(0, ((im.shape[0] - self.fltr.shape[0] + 2*self.padding)/self.stride+1):
            for j in range(0, ((im.shape[1] - self.fltr.shape[1] + 2*self.padding)/stride)+1):
                y[i][j] = np.dot(im[][],  self.fltr)
