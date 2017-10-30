# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import numpy as np

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
        return 0

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
        return DFT.apply_to_matrix(matrix, DFT.ifft, np.float32)


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        return matrix


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        return matrix



def unit_test_fft():
    matrix1 = np.int_(np.random.rand(3, 3) * 256)
    matrix2 = np.int_(np.random.rand(3, 2) * 256)
    matrix3 = np.int_(np.random.rand(15, 5) * 256)

    def attempt(matrix):
        dft_obj = DFT()

        my_mat = dft_obj.forward_transform(matrix)
        np_mat = np.fft.fft2(matrix)

        assert np.allclose(my_mat, np_mat), "Matrices differ: \nNumpy:\n{} \n\nMine:\n{}".format(np_mat, my_mat)

    attempt(matrix1)
    attempt(matrix2)
    attempt(matrix3)

    print("[FFT] Test passed!")


if __name__ == '__main__':
    unit_test_fft()
