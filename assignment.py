import numpy as np
import pandas as pd
import os 
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt
import copy
import math

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
                try:
                    h[int(im[i, j])]+=1
                except ValueError:
                    pass
                    
        print('out of the loop')
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
        print('Rescaling Image')
        im = self.rescale(im)
        print('Calculating the Image Histogram')
        orig_hist = self.imhist(im)
        cdf = np.array(self.cumsum(orig_hist)) #cumulative distribution function
        sk = np.uint8(255 * cdf) #finding transfer function values
        s1, s2 = im.shape
        Y = np.zeros_like(im)
    	# applying transfered values for each pixels
        for i in range(0, s1):
            for j in range(0, s2):
                try:
                    Y[i, j] = sk[int(im[i, j])]
                except ValueError:
                    pass
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

class Convolution():
    def __init__(self, path = None, fltr = np.asarray([1]), padding = 0, stride = 1):
        self.fltr = fltr
        self.padding = padding
        self.stride = stride

    def convolve_with_impulse_response(self, im, magnitude = 50):
        """
            convolves input image with an impulse response of magnitude
            im - np.ndarray() n*m
            Output - np.ndarray() n*m
        """
        y = np.zeros((im.shape[0], im.shape[1])) # [y] = [im]*[delta]
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
        y = np.zeros((int((im.shape[0] - self.fltr.shape[0] + 2*self.padding)/self.stride+1), 
                     int((im.shape[1] - self.fltr.shape[1] + 2*self.padding)/self.stride+1)))
        for i in range(0, int((im.shape[0] - self.fltr.shape[0] + 2*self.padding)/self.stride+1)):
            for j in range(0, int((im.shape[1] - self.fltr.shape[1] + 2*self.padding)/self.stride+1)):
                y[i][j] = self.product(im[i : i + self.fltr.shape[0], j : j + self.fltr.shape[1]],  self.fltr)
        return (y)
    
    def product(self, im, fltr):
        prod = 0
        for i in range(fltr.shape[0]):
            for j in range(fltr.shape[1]):
                prod = prod + im[i][j]*fltr[i][j]
        return prod                
    
def main():
    path = input("enter the path to the image here")
    size = input('input the size of the gaussian filter for part 4')
    size = math.floor(float(size)/2)
    g = gaussian_kernel(size)
    image = Image(path)
    grad = Gradient(path)
    conv = Convolution(fltr = g, 
                       padding = 0,
                       stride = 1)
    if path[-3:] == 'png':
        print('.png file')
        image.import_image()
    else:
        print('medical image')
        image.import_image_nib_compatible()
        
    image.image_data = image.image_data.astype('int64')
    print(image.image_data)
    print(image.image_data.shape)
    
    if len(image.image_data.shape) == 2:
        print('2')
        print('Performing histogram equalization')
        Y, orig_hist, new_hist, sk, maxval, minval = image.histeq(image.image_data)
        grad_x, grad_y, grad_2_x, grad_2_y = grad.main(image.image_data)
        print('Computing gradient')
        plot_func(Y, image.image_data, maxval, minval, grad_x, grad_y, grad_2_x, grad_2_y)
        # The magnitude of the impulse response has been assumed to be 5 here. Please change here to observe changes in the output
        print('Performing Convolution')
        plot_conv_results(image.image_data, conv)
    elif len(image.image_data.shape)==3:
        print('3')
        for i in range(image.image_data.shape[2]):
            print('Performing histogram equalization on layer {i}')
            Y, orig_hist, new_hist, sk, maxval, minval = image.histeq(image.image_data[:,:,i])
            print('Computing gradient')
            grad_x, grad_y, grad_2_x, grad_2_y = grad.main(image.image_data[:, :, i])
            plot_func(Y, image.image_data[:, :, i], maxval, minval, grad_x, grad_y, grad_2_x, grad_2_y)
            print('Performing Convolution')
            plot_conv_results(image.image_data[:, :, i], conv)
    elif len(image.image_data.shape) == 4:
        print('4')
        for i in range(image.image_data.shape[2]):
            for j in range(image.image_data.shape[3]):
                print('Performing histogram equalization on layer {j} of slice{i}')
                Y, orig_hist, new_hist, sk, maxval, minval = image.histeq(image.image_data[:,:,i,j])
                print('Computing gradient')
                grad_x, grad_y, grad_2_x, grad_2_y = grad.main(image.image_data[:, :, i, j])
                plot_func(Y, image.image_data[:, :, i, j], maxval, minval, grad_x, grad_y, grad_2_x, grad_2_y)
                print('Performing Convolution')
                plot_conv_results(image.image_data[:, :, i, j], conv)
    
def plot_func(Y, im, maxval, minval, grad_x, grad_y, grad_2_x, grad_2_y):
    fig, axs = plt.subplots(3,2, sharex = True, sharey = True)
    axs[0, 0].imshow(Y, cmap = 'gray')
    axs[0, 0].set_title('Transformed Image')
    axs[0, 1].imshow(im, cmap = 'gray')
    axs[0, 1].set_title('Original Image')
    axs[1, 0].imshow(grad_x, cmap = 'gray')
    axs[1, 0].set_title('First order derivative wrt x')
    axs[1, 1].imshow(grad_y, cmap = 'gray')
    axs[1, 1].set_title('First order derivative wrt y')
    axs[2, 0].imshow(grad_2_x, cmap = 'gray')
    axs[2, 0].set_title('Second order derivative wrt x')
    axs[2, 1].imshow(grad_2_y, cmap = 'gray')
    axs[2, 1].set_title('Second order derivative wrt y')
    plt.show()
    
def plot_conv_results(im, conv):
    out_conv_impulse_response = conv.convolve_with_impulse_response(im, 5)
    out_conv_g = conv.convolve(im)
    fig, axs = plt.subplots(3, 1, sharex = True, sharey = True)
    axs[0].imshow(im)
    axs[1].imshow(out_conv_impulse_response)
    axs[2].imshow(out_conv_g)
    axs[0].set_title('Original Image')
    axs[1].set_title('Convolution with Impulse Response')
    axs[2].set_title('Convolution with Gaussian smoothing filter')
    plt.show()
    
def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()

main()
