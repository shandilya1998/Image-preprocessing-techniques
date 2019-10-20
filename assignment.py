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

    def rescale(self, im, des_max = 255, des_min = 0):
        """
            Rescales the image pixel values between 0 to 255.
            im - np.ndarray n*m
            output - np.ndarray n*m
        """
        minval = im.min()
        maxval = im.max()
        rescaled_image = ((des_max-des_min)/(maxval-minval))*(im-minval+des_min)
        return rescaled_image
    
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
        #rescale image to have pixel values between 0 to 255
        minval = im.min()
        maxval = im.max()
        im = self.rescale(im)
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
        Y = self.rescale(Y, maxval, minval)
        return Y,orig_hist,new_hist,sk,maxval,minval

class Gradient():
    def __init__(self, path = None):
        self.path = path
    
    def grad_x(self, im):
        """
            im - np.ndarray() n*m
            Padding applied : 1
            Output : np.ndarray() (n)*(m)
        """
        grad_x = np.zeros(im.shape) 
        temp = np.zeros((im.shape[0] + 1, im.shape[1] + 1))
        temp[0 : temp.shape[0] - 1, 0 : temp.shape[1]-1] = im
        for i in range(0, temp.shape[0] - 1):
            for j in range(0, temp.shape[1] - 1):
                grad_x[i][j] = temp[i + 1][j]-temp[i][j]
        return grad_x

    def grad_y(self, im):
        """
            im - np.ndarray() n*m
            Padding applied : 1
            Output : np.ndarray() (n)*(m)
        """
        grad_y = np.zeros(im.shape)
        temp = np.zeros((im.shape[0] + 1, im.shape[1] + 1))
        temp[0 : temp.shape[0] - 1, 0 : temp.shape[1] - 1] = im
        for i in range(0, temp.shape[0] - 1):
            for j in range(0, temp.shape[1] - 1):
                grad_y[i][j] = temp[i][j + 1]-temp[i][j]
        return grad_y

    def grad_2_x(self, im):
        """
            im - np,ndarray() n*m
            Padding applied - 1
            Output : np.ndarray() n*m
        """
        grad_2_x = np.zeros(im.shape)
        temp = np.zeros((im.shape[0]+2, im.shape[1]+2))
        print(temp.shape)
        print(temp.shape[0])
        print(temp.shape[1])
        print(temp[1 : temp.shape[0] - 1, 1 : temp.shape[1] - 1].shape)
        temp[1 : temp.shape[0] - 1, 1 : temp.shape[1] - 1] = im
        for i in range(1, temp.shape[0] - 2):
            for j in range(1, temp.shape[1] - 2):
                grad_2_x[i][j] = temp[i + 1][j] + temp[i - 1][j] - 2*temp[i][j]
        return grad_2_x

    def grad_2_y(self, im):
        """
            im - np.ndarray() n*m
            Padding applied - 1
            Output - np.ndarray() n*m
        """
        grad_2_y = np.ndarray(im.shape)
        temp = np.zeros((im.shape[0] + 2, im.shape[1] + 2))
        temp[1 : temp.shape[0] - 1, 1 : temp.shape[1] - 1]  = im
        for i in range(1, temp.shape[0]-2):
            for j in range(1, temp.shape[1]-2):
                grad_2_y[i][j] = temp[i][j + 1] + temp[i][j - 1] - 2*temp[i][j]
        return grad_2_y

    def main(self, im):
        grad_x = self.grad_x(im)
        grad_y = self.grad_y(im)
        grad_2_x = self.grad_2_x(im)
        grad_2_y = self.grad_2_y(im)
        return grad_x, grad_y, grad_2_x, grad_2_y

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
        y = np.zeros(((im.shape[0] - self.fltr.shape[0] + 2*self.padding)/self.stride)+1, 
                     ((im.shape[1] - self.fltr.shape[1] + 2*self.padding)/self.stride)+1)
        k = 0
        l = 0
        for i in range(0, ((im.shape[0] - self.fltr.shape[0] + 2*self.padding)/self.stride+1)):
            for j in range(0, ((im.shape[1] - self.fltr.shape[1] + 2*self.padding)/self.stride)+1):
                y[i][j] = np.dot(im[k : k+self.fltr.shape[0]][l : self.fltr.shape[1]],  self.fltr)
                l = l + 1
            k = k + 1
        return (y)

def main():
    path = input("enter the path to the image here")
    image = Image(path)
    grad = Gradient(path)
    
    if path[-3:] == 'png':
        image.import_image()
    else:
        image.import_image_nib_compatible()
        
    image.image_data = image.image_data.astype('int64')
    
    if len(image.image_data.shape) == 2:
        print('2')
        Y, orig_hist, new_hist, sk, maxval, minval = image.histeq(image.image_data)
        grad_x, grad_y, grad_2_x, grad_2_y = grad.main(image.image_data)
        plot_func(Y, image.image_data, maxval, minval, grad_x, grad_y, grad_2_x, grad_2_y)
    elif len(image.image_data.shape)==3:
        print('3')
        for i in range(image.image_data.shape[2]):
            Y, orig_hist, new_hist, sk, maxval, minval = image.histeq(image.image_data[:,:,i])
            grad_x, grad_y, grad_2_x, grad_2_y = grad.main(image.image_data[:, :, i])
            plot_func(Y, image.image_data[:, :, i], maxval, minval, grad_x, grad_y, grad_2_x, grad_2_y)
    elif len(image.image_data.shape) == 4:
        print('4')
        for i in range(image.image_data.shape[2]):
            for j in range(image.image_data.shape[3]):
                Y, orig_hist, new_hist, sk, maxval, minval = image.histeq(image.image_data[:,:,i,j])
                grad_x, grad_y, grad_2_x, grad_2_y = grad.main(image.image_data[:, :, i, j])
                plot_func(Y, image.image_data[:, :, i, j], maxval, minval, grad_x, grad_y, grad_2_x, grad_2_y)

def plot_func(Y, im, maxval, minval, grad_x, grad_y, grad_2_x, grad_2_y):
    fig, axs = plt.subplots(3,2, sharex = True, sharey = True)
    axs[0, 0].imshow(Y)
    axs[0, 0].set_title('Transformed Image')
    axs[0, 1].imshow(im)
    axs[0, 1].set_title('Original Image')
    axs[1, 0].imshow(grad_x)
    axs[1, 0].set_title('First order derivative wrt x')
    axs[1, 1].imshow(grad_y)
    axs[1, 1].set_title('First order derivative wrt y')
    axs[2, 0].imshow(grad_2_x)
    axs[2, 0].set_title('Second order derivative wrt x')
    axs[2, 1].imshow(grad_2_y)
    axs[2, 1].set_title('Second order derivative wrt y')
    plt.show()

main()
