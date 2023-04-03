import numpy as np
from numpy.linalg import inv

np.set_printoptions(precision=7, suppress=True, linewidth=100)


def function(t, y):
    return t - y**2


def eulers(start_of_t, end_of_t, iterations, original_w):
    h = (end_of_t - start_of_t) / iterations

    for cur_iteration in range(0, iterations):
        t = start_of_t
        w = original_w
        h = h

        # do calculations
        incremented_w = w + h * function(t, w)

        # reassign w and change t
        start_of_t = t + h
        original_w = incremented_w

    print("%.5f" % original_w + "\n")


def runge_kutta(start_of_t, end_of_t, iterations, original_w):
    h = (end_of_t - start_of_t) / iterations

    for cur_iteration in range(0, iterations):
        t = start_of_t
        w = original_w
        h = h

        k_1 = h * function(t, w)
        k_2 = h * function(t + (h/2), w + (1/2)*k_1)
        k_3 = h * function(t + (h/2), w + (1/2)*k_2)
        k_4 = h * function(t + h, w + k_3)

        incremented_w = w + (1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)

        start_of_t = t + h
        original_w = incremented_w

    print("%.5f" % original_w + "\n")


def gaussian(A, b):
    n = len(b)

    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)

    # Eliminate entries below pivot
    for i in range(n):

        # Find pivot row
        max_row = i
        for j in range(i + 1, n):
            if abs(Ab[j, i]) > abs(Ab[max_row, i]):
                max_row = j
        # Swap rows to bring pivot element to diagonal
        Ab[[i, max_row], :] = Ab[[max_row, i], :]  # operation 1 of row operations

        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, :] = Ab[j, :] - factor * Ab[i, :]

    x = np.zeros(n)

    # Perform back-substitution
    x[n-1] = Ab[n-1, n] / Ab[n-1, n-1]

    for i in range(n - 2, -1, -1):
        x[i] = Ab[i, n]
        for j in range(i + 1, n):
            x[i] = x[i] - Ab[i, j] * x[j]
        x[i] = x[i] / Ab[i, i]

    print(str(x) + "\n")


def l_and_u_matrix(matrix):
    n = len(matrix)

    # prep for L matrix (starts with reduced row echelon form)
    x = np.zeros((n, n))
    for i in range(n):
        for j in range(i, i + 1):
            x[i, j] = 1

    for i in range(n):
        for j in range(i+1, n):
            # U matrix
            factor = matrix[j, i] / matrix[i, i]
            matrix[j, :] = matrix[j, :] - factor * matrix[i, :]

            # L matrix
            x[i, j] = factor

    u_matrix = matrix
    l_matrix = x.transpose()

    print(str(l_matrix) + "\n")
    print(str(u_matrix) + "\n")

    # # Check if multiplication of L and U results in original matrix
    # print(np.dot(l_matrix, u_matrix))


def diagonally_dom(matrix):
    n = len(matrix)

    for i in range(n):
        row_sum = 0
        for j in range(n):
            if i != j:
                row_sum = row_sum + abs(matrix[i, j])
                if abs(matrix[i, i]) <= row_sum:
                    return False
    return True


def positive_def(matrix):
    n = len(matrix)

    # prove symmetric (transposed matrix = original)
    transposed = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            transposed[i, j] = matrix[j, i]

    if (transposed == matrix).all():
        symmetric = True
    else:
        symmetric = False

    # check all eigenvalues positive
    eigenvalues = np.all(np.linalg.eigvals(matrix) > 0)

    if eigenvalues and symmetric:
        print("True" + "\n")
    else:
        print("False" + "\n")


def main():
    # function is t - y**2; if not, MUST update def function
    start_of_t = 0
    end_of_t = 2
    iterations = 10
    original_w = 1

    # QUESTION 1 / Euler Method
    eulers(start_of_t, end_of_t, iterations, original_w)

    # QUESTION 2 / Runge-Kutta
    runge_kutta(start_of_t, end_of_t, iterations, original_w)

    # QUESTION 3 / Gaussian elimination + backward sub to solve linear system of eq.
    A = np.array([[2.0, -1, 1],
                  [1, 3, 1],
                  [-1, 5, 4]])
    b = np.array([6.0, 0, -3])
    gaussian(A, b)

    # QUESTION 4 / LU Factorization
    matrix = np.array([[1.0, 1, 0, 3],
                       [2, 1, -1, 1],
                       [3, -1, -1, 2],
                       [-1, 2, 3, -1]])
    determinant = np.linalg.det(matrix)
    print("%.5f" % determinant + "\n")
    l_and_u_matrix(matrix)

    # QUESTION 5 / Diagonally dominate?
    matrix = np.array([[9.0, 0, 5, 2, 1],
                       [3, 9, 1, 2, 1],
                       [0, 1, 7, 2, 3],
                       [4, 2, 3, 12, 2],
                       [3, 2, 4, 0, 8]])
    print(str(diagonally_dom(matrix)) + "\n")

    # QUESTION 6 / Positive definite?
    matrix = np.array([[2.0, 2, 1],
                       [2, 3, 0],
                       [1, 0, 2]])
    positive_def(matrix)


if __name__ == '__main__':
    main()
