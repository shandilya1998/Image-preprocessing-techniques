import numpy as np
import os 
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt


class Image():
    """
        Image is used to import and store the input image
    """
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

class Convolution():
    """
        Convolution object is used to perform spatial filtering
    """
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
    
    
class Freq_Domain_filtering():
    def __init__(self, img):
        self.image_data = img
        self.image_data_fft = np.fft.fft2(self.image_data)
        self.image_data_fft_fshift = np.fft.fftshift(self.image_data_fft)
        
    def laplacian_prod(self):
        """
            Output : np.ndarray of dimensions img.shape Contains filtered image data
        """
        rows, cols = self.image_data_fft_fshift.shape
        masklap = np.zeros((rows,cols))
        for i in range(int(-rows/2),int(rows/2)):
            for j in range(int(-cols/2),int(cols/2)):
                masklap[i][j] = -(i**2 + j**2)
        img_fft_fshift = self.image_data_fft_fshift + self.image_data_fft_fshift*masklap
        img_fft = np.fft.ifftshift(img_fft_fshift)
        img = np.fft.ifft2(img_fft)
        img = np.abs(img)
        return img
        
def main():
    path = input("enter the path to the image here")
    fltr_laplacian = np.asarray([[-1, -1, -1],[-1, 9,-1],[-1, -1, -1]])
    image = Image(path)
    conv = Convolution(fltr = fltr_laplacian, 
                       padding = 0,
                       stride = 1)
    if path[-3:] == 'png':
        print('.png file')
        image.import_image()
    else:
        print('medical image')
        image.import_image_nib_compatible()
    print('Applying convolution in spatial domain')
    if len(image.image_data.shape) == 2:
        print('2')
        print('Performing Convolutions')
        freq_fltr = Freq_Domain_filtering(image.image_data)
        plot_conv_results(image.image_data, conv, freq_fltr)
    elif len(image.image_data.shape)==3:
        print('3')
        for i in range(image.image_data.shape[2]):
            print('Performing Convolutions')
            freq_fltr = Freq_Domain_filtering(image.image_data[:, :, i])
            plot_conv_results(image.image_data[:, :, i], conv, freq_fltr)
    elif len(image.image_data.shape) == 4:
        print('4')
        for i in range(image.image_data.shape[2]):
            for j in range(image.image_data.shape[3]):
                print('Performing Convolution')
                freq_fltr = Freq_Domain_filtering(image.image_data[:, :, i, j])
                plot_conv_results(image.image_data[:, :, i, j], conv, freq_fltr)

def plot_conv_results(im, conv, freq_fltr):
    out_conv_g = conv.convolve(im)
    out = freq_fltr.laplacian_prod()
    fig, axs = plt.subplots(1, 3, sharex = True, sharey = True)
    axs[0].imshow(im)
    axs[1].imshow(out_conv_g)
    axs[2].imshow(out)
    axs[0].set_title('Original Image')
    axs[1].set_title('Spatial Convolution with Laplacian filter')
    axs[2].set_title('Laplacian filtering applied in frequency domain')
    plt.show()
    
main()