# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv

import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, image, filter_name, cutoff, order=0):
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
        rows, cols = shape
        x = np.linspace(-0.5, 0.5, self.image.shape[0])  * rows
        y = np.linspace(-0.5, 0.5, self.image.shape[1])  * cols
        radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
        filt = np.ones(self.image.shape)
        filt[radius > cutoff] = 0
        return filt

    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""
        return 1. - self.get_ideal_low_pass_filter(shape, cutoff)

    def get_butterworth_low_pass_filter(self, shape, cutoff):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""
        rows, cols = shape
        x = np.linspace(-0.5, 0.5, cols) * cols
        y = np.linspace(-0.5, 0.5, rows) * rows
        radius = np.sqrt((x ** 2)[np.newaxis] + (y ** 2)[:, np.newaxis])
        filt = 1 / (1.0 + (radius / cutoff) ** (2 * self.order))
        return filt


    def get_butterworth_high_pass_filter(self, shape, cutoff):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""
        return 1. - self.get_butterworth_low_pass_filter(shape, cutoff)

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        rows, cols = shape
        x = np.linspace(-0.5, 0.5, cols) * cols
        y = np.linspace(-0.5, 0.5, rows) * rows
        radius = np.sqrt((x ** 2)[np.newaxis] + (y ** 2)[:, np.newaxis])
        sigma = cutoff
        filt = np.exp(-(radius ** 2) / (2 * (sigma ** 2)))
        return filt

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""
        return 1. - self.get_gaussian_low_pass_filter(shape, cutoff)

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """
        img = np.round(image).astype(np.uint8)
        A, B = np.min(img), np.max(img)
        K = 256.
        frac = (K - 1.) / (B - A)
        img = np.round((img - A) * frac + 0.5)
        return img

    def apply(self, filter_fn):
        fft_orig = np.fft.fftshift(np.fft.fft2(self.image))
        filt = filter_fn(self.image.shape, self.cutoff)
        fft_new = fft_orig * filt
        recon_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))
        recon_image = self.post_process_image(recon_image)
        return fft_orig, filt, fft_new, recon_image

    def add_subplot(self, r, c, indexes, fft_orig, filt, fft_new, recon_image):
        plt.subplot(r, c, indexes[0]), plt.title('Original')
        plt.imshow(self.image), plt.gray(), plt.axis('off')

        plt.subplot(r, c, indexes[1]), plt.title('FFT')
        plt.imshow(np.log(np.abs(fft_orig))), plt.gray(), plt.axis('off')

        plt.subplot(r, c, indexes[2]), plt.title('Filter')
        plt.imshow(filt), plt.gray(), plt.axis('off')

        plt.subplot(r, c, indexes[3]), plt.title('Filtered FFT')
        plt.imshow(np.log(np.abs(fft_new))), plt.gray(), plt.axis('off')

        plt.subplot(r, c, indexes[4]), plt.title('Restored')
        plt.imshow(recon_image), plt.gray(), plt.axis('off')

    def debug(self):
        r, c = 6, 5

        fft_orig, filt, fft_new, recon_image = self.apply(self.get_ideal_low_pass_filter)
        self.add_subplot(r, c, range(1, 6), fft_orig, filt, fft_new, recon_image)

        fft_orig, filt, fft_new, recon_image = self.apply(self.get_ideal_high_pass_filter)
        self.add_subplot(r, c, range(6, 11), fft_orig, filt, fft_new, recon_image)

        temp = self.order
        self.order = 2
        fft_orig, filt, fft_new, recon_image = self.apply(self.get_butterworth_low_pass_filter)
        self.add_subplot(r, c, range(11, 16), fft_orig, filt, fft_new, recon_image)

        fft_orig, filt, fft_new, recon_image = self.apply(self.get_butterworth_high_pass_filter)
        self.add_subplot(r, c, range(16, 21), fft_orig, filt, fft_new, recon_image)
        self.order = temp

        fft_orig, filt, fft_new, recon_image = self.apply(self.get_gaussian_low_pass_filter)
        self.add_subplot(r, c, range(21, 26), fft_orig, filt, fft_new, recon_image)

        fft_orig, filt, fft_new, recon_image = self.apply(self.get_gaussian_high_pass_filter)
        self.add_subplot(r, c, range(26, 31), fft_orig, filt, fft_new, recon_image)

        plt.show()

    @staticmethod
    def mag(mat):
        mat = np.abs(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[0]):
                if mat[i, j] != 0:
                    mat[i, j] = np.log(mat[i, j])
        return mat

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
        # This print the whole pipeline for each of the algortihms in both high and low levels
        # self.debug()

        fft_orig = np.fft.fftshift(np.fft.fft2(self.image))
        filt = self.filter(self.image.shape, self.cutoff)
        fft_new = fft_orig * filt
        restored = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))

        # self.add_subplot(1, 5, range(1, 6), fft_orig, filt, fft_new, restored)
        # plt.show()

        return [
            self.post_process_image(restored),
            self.post_process_image(Filtering.mag(fft_orig)),
            self.post_process_image(Filtering.mag(fft_new))
        ]


