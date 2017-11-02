# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import numpy as np
from scipy import fftpack
import cv2


class DFT:
    @staticmethod
    def apply_to_matrix(matrix, func, dtype):
        mat = np.zeros(matrix.shape, dtype=dtype)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i, j] = func(matrix, i, j)
        return mat

    @staticmethod
    def fft(matrix, x, y):
        (M, N) = matrix.shape
        total = 0
        for m in range(M):
            for n in range(N):
                period   = (x * m / M) + (y * n / N)
                exponent = -2j * np.pi * period
                result   = matrix[m, n] * np.exp(exponent)
                total += result
        return total

    @staticmethod
    def ifft(matrix, x, y):
        (M, N) = matrix.shape
        total = 0
        for m in range(M):
            for n in range(N):
                period   = (x * m / M) + (y * n / N)
                exponent = 2j * np.pi * period
                result   = matrix[m, n] * np.exp(exponent)
                total += result
        return total / (M * N)

    # @staticmethod
    # def dct(matrix, p, q):
    #     """ Taken from Matlab documentation """
    #     (M, N) = matrix.shape
    #     total = 0.
    #     for m in range(M):
    #         for n in range(N):
    #             xangle = np.pi * (2 * m + 1) * p / (2 * M)
    #             yangle = np.pi * (2 * n + 1) * q / (2 * N)
    #             total += matrix[m, n] * np.cos(xangle) * np.cos(yangle)
    #
    #     alpha_p = (1. / (M ** 0.5)) if p == 0 else ((2. / M) ** 0.5)
    #     alpha_q = (1. / (N ** 0.5)) if q == 0 else ((2. / N) ** 0.5)
    #     return alpha_p * alpha_q * total


    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""
        return DFT.apply_to_matrix(matrix, DFT.fft, complex)

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""
        return DFT.apply_to_matrix(matrix, DFT.ifft, complex)


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        return DFT.apply_to_matrix(matrix, DFT.fft, complex).real


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        mag_func = lambda mat, i, j: (mat[i, j].real ** 2 + mat[i, j].imag ** 2) ** 0.5

        return DFT.apply_to_matrix(matrix, mag_func, np.float)


def tester(attempt):
    matrix1 = np.int_(np.random.rand(3, 3) * 256)
    matrix2 = np.int_(np.random.rand(3, 2) * 256)
    matrix3 = np.int_(np.random.rand(15, 5) * 256)
    matrix4 = np.int_(np.random.rand(15, 15) * 256)
    attempt(matrix1)
    attempt(matrix2)
    attempt(matrix3)
    attempt(matrix4)


def unit_test_fft():
    def attempt(matrix):
        dft_obj = DFT()

        my_mat = dft_obj.forward_transform(matrix)
        np_mat = np.fft.fft2(matrix)

        assert np.allclose(my_mat, np_mat), "Matrices differ: \nNumpy:\n{} \n\nMine:\n{}".format(np_mat, my_mat)

    tester(attempt)
    print("[DFT] FFT Test passed!")

def unit_test_ifft():
    def attempt(matrix):
        dft_obj = DFT()

        ftrans = dft_obj.forward_transform(matrix)
        my_mat = dft_obj.inverse_transform(ftrans)
        np_mat = np.fft.ifft2(ftrans)

        assert np.allclose(my_mat, matrix), "Inverse FT did not get the original matrix: \nOriginal:\n{} \n\nIFT:\n{}".format(matrix, my_mat)
        assert np.allclose(my_mat, np_mat), "Matrices differ: \nNumpy:\n{} \n\nMine:\n{}".format(np_mat, my_mat)

    tester(attempt)
    print("[DFT] IFFT Test passed!")

def unit_test_magnitude():
    def attempt(matrix):
        dft_obj = DFT()

        ftrans = dft_obj.forward_transform(matrix)
        my_mat = dft_obj.magnitude(ftrans)
        np_mat = np.absolute(ftrans)

        assert np.allclose(my_mat, np_mat), "Matrices differ: \nNumpy:\n{} \n\nMine:\n{}".format(np_mat, my_mat)

    tester(attempt)
    print("[DFT] Magnitude Test passed!")

def unit_test_dct():
    def attempt(matrix):
        dft_obj = DFT()

        # my_mat = dft_obj.discrete_cosine_tranform(matrix)
        my_mat = dft_obj.forward_transform(matrix).real

        # np_mat = fftpack.dct(fftpack.dct(matrix.T, norm='ortho').T, norm='ortho')
        # np_mat = fftpack.dct(fftpack.dct(matrix, axis=1), axis=0)
        np_mat = np.fft.fft2(matrix).real

        # assert np.allclose(my_mat, cv_mat), "Matrices differ: \nOpenCV:\n{} \n\nMine:\n{}".format(cv_mat, my_mat)
        # assert np.allclose(my_mat, np_mat), "Matrices differ: \nSciPy:\n{} \n\nMine:\n{}".format(np_mat, my_mat)
        assert np.allclose(my_mat, np_mat), "Matrices differ: \nNumPy:\n{} \n\nMine:\n{}".format(np_mat, my_mat)

    tester(attempt)
    print("[DFT] DCT Test passed!")


if __name__ == '__main__':
    print("Running unit tests!")
    unit_test_fft()
    unit_test_ifft()
    unit_test_magnitude()
    unit_test_dct()
    print("\nAll tests passed!")
