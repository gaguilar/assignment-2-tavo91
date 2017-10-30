# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import numpy as np
from scipy.fftpack import dct as scipydct


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

    @staticmethod
    def dct(matrix, i, j):
        (M, N) = matrix.shape
        total = 0.
        for x in range(M):
            for y in range(N):
                xangle = (2 * x + 1) * i * np.pi / (2 * M)
                yangle = (2 * y + 1) * j * np.pi / (2 * N)

                total += matrix[x, y] * np.cos(xangle) * np.cos(yangle)

        icoef = 1 if i > 0 else 2 ** -0.5
        jcoef = 1 if j > 0 else 2 ** -0.5
        total *= icoef * jcoef * (2 * N) ** 0.5
        return round(total)


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
        return DFT.apply_to_matrix(matrix, DFT.dct, np.float32)


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
    print("[FFT] Test passed!")

def unit_test_ifft():
    def attempt(matrix):
        dft_obj = DFT()

        ftrans = dft_obj.forward_transform(matrix)
        my_mat = dft_obj.inverse_transform(ftrans)
        np_mat = np.fft.ifft2(ftrans)

        assert np.allclose(my_mat, matrix), "Inverse FT did not get the original matrix: \nOriginal:\n{} \n\nIFT:\n{}".format(matrix, my_mat)
        assert np.allclose(my_mat, np_mat), "Matrices differ: \nNumpy:\n{} \n\nMine:\n{}".format(np_mat, my_mat)

    tester(attempt)
    print("[IFFT] Test passed!")

def unit_test_magnitude():
    def attempt(matrix):
        dft_obj = DFT()

        ftrans = dft_obj.forward_transform(matrix)
        my_mat = dft_obj.magnitude(ftrans)
        np_mat = np.absolute(ftrans)

        assert np.allclose(my_mat, np_mat), "Matrices differ: \nNumpy:\n{} \n\nMine:\n{}".format(np_mat, my_mat)

    tester(attempt)
    print("[Magnitude] Test passed!")

def unit_test_dct():
    matrix = np.array([
        [140, 144, 147, 140, 140, 155, 179, 175],
        [144, 152, 140, 147, 140, 148, 167, 179],
        [152, 155, 136, 167, 163, 162, 152, 172],
        [168, 145, 156, 160, 152, 155, 136, 160],
        [162, 148, 156, 148, 140, 136, 147, 162],
        [147, 167, 140, 155, 155, 140, 136, 162],
        [136, 156, 123, 167, 162, 144, 140, 147],
        [148, 155, 136, 155, 152, 147, 147, 136]
    ])

    def attempt(matrix):
        dft_obj = DFT()

        my_mat = dft_obj.discrete_cosine_tranform(matrix)
        np_mat = scipydct(scipydct(matrix, axis=0), axis=1)

        assert np.allclose(my_mat, np_mat), "Matrices differ: \nSciPy:\n{} \n\nMine:\n{}".format(np_mat, my_mat)

    attempt(matrix)
    print("[DCT] Test passed!")


if __name__ == '__main__':
    unit_test_fft()
    unit_test_ifft()
    unit_test_magnitude()
    unit_test_dct()
