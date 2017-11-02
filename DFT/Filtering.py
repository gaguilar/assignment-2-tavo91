# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, misc

import cv2

def display_image(window_name, image):
    """A function to display image"""
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order


    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""
        rows, cols = self.image.shape
        crow, ccol = rows / 2, cols / 2

        mask = np.zeros((rows, cols), np.uint8)

        for row in range(rows):
            for col in range(cols):
                dist = ((row - crow) ** 2 + (col - ccol) ** 2) ** 0.5
                if dist <= cutoff:
                    mask[row, col] = 1
        return mask


    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        
        return 0

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""

        
        return 0

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        
        return 0

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""

        
        return 0

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        
        return 0

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        cdf = hist.cumsum()

        # cdf_normalized = cdf * hist.max() / cdf.max()
        # plt.plot(cdf_normalized, color='b')
        # plt.hist(image.flatten(), 256, [0, 256], color='r')
        # plt.xlim([0, 256])
        # plt.legend(('cdf', 'histogram'), loc='upper left')
        # plt.show()


        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        print(cdf.shape)
        print(np.min(image), np.max(image))

        # temp = image
        # temp_img = np.abs(temp - np.min(image) * 255 / np.max(image) - np.min(image)).astype(int)
        # print(temp_img[:10, :10])
        image2 = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image2[i, j] = cdf[int(image[i, j])]
                # image2[i, j] = cdf[temp_img[i, j]]

        # print()
        # print()
        print(image2[:10, :10])
        return image2


    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)

        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey
        scale full contrast stretch and dtype=uint8
        """

        # # Step 1
        # fft = np.fft.fft2(self.image)
        # # Step 2
        # shifted_fft = np.fft.fftshift(fft)
        # # Step 3
        # magnitude_spectrum = np.log(np.abs(shifted_fft))
        # img_step_3 = self.filter(img_step_2)
        # img_step_4 = np.convolve(img_step_3)
        # img_step_5 = np.fft.ifftshift(img_step_4)
        # img_step_6 = np.fft.ifft2(img_step_5)
        # img_step_7 = np.absolute(img_step_6)
        # img_step_8 = self.post_process_image(img_step_7)

        def mag(matrix):
            matrix = np.abs(matrix)
            matrix = np.log(matrix)
            return matrix

        f, axarr = plt.subplots(2, 4)
        axarr[0, 0].imshow(self.image, cmap='gray')
        axarr[0, 0].set_title('Input Image')

        fft = np.fft.fft2(self.image)
        axarr[0, 1].imshow(mag(fft), cmap='gray')
        axarr[0, 1].set_title('FFT')

        shifted_fft = np.fft.fftshift(fft)
        axarr[0, 2].imshow(mag(shifted_fft), cmap='gray')
        axarr[0, 2].set_title('Shifted FFT')

        mask = self.filter(None, self.cutoff)
        # mask[crow - round(frow / 2):crow + round(frow / 2), ccol - round(fcol/2):ccol + round(fcol/2)] = filt
        axarr[0, 3].imshow(mask, cmap='gray')
        axarr[0, 3].set_title('Filter')

        # conv_image = signal.convolve2d(mag(shifted_fft), mask)
        conv_image = shifted_fft * mask
        axarr[1, 0].imshow(mag(conv_image), cmap='gray')
        axarr[1, 0].set_title('Conv SFFT')

        inv_shifted_image = np.fft.ifftshift(conv_image)
        axarr[1, 1].imshow(mag(inv_shifted_image), cmap='gray')
        axarr[1, 1].set_title('Shifted IFFT')

        iff_image = np.fft.ifft2(inv_shifted_image)
        axarr[1, 2].imshow(mag(iff_image), cmap='gray')
        axarr[1, 2].set_title('IFFT')

        processed_image = self.post_process_image(mag(iff_image))
        axarr[1, 3].imshow(processed_image, cmap='gray')
        axarr[1, 3].set_title('Post')

        plt.setp([a.get_xticklabels() for b in axarr for a in b], visible=False)
        plt.setp([a.get_yticklabels() for b in axarr for a in b], visible=False)
        plt.show()

        return [self.image, self.image, self.image]
