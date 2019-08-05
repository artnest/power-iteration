import numpy as np


def power_method(matrix, approx, eps):
    eigenvector = approx.T
    k = 0

    converge = False
    while not converge:
        eigenvector_new = matrix * eigenvector
        eigenvector_new /= np.linalg.norm(eigenvector_new)
        k += 1

        converge = np.linalg.norm(eigenvector_new - eigenvector) <= eps
        eigenvector = eigenvector_new

    eigenvalue_max = np.dot(eigenvector.T, matrix * eigenvector) / np.dot(eigenvector.T, eigenvector)
    return eigenvalue_max, k


def numpy_eigenvalues(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues_abs = list(abs(eigenvalues))
    idx = eigenvalues_abs.index(max(eigenvalues_abs))
    eigenvalue_max = eigenvalues[idx]

    return eigenvalue_max


def pm_calculation(A, b, eps):
    matrix = np.matrix(A)
    vector = np.matrix(b)
    print()

    eigenvalue_max_p, k = power_method(matrix, vector, eps)
    print("Computed eigenvalue:", eigenvalue_max_p, ",", k, "iterations")
    eigenvalue_max = numpy_eigenvalues(A)
    print("Largest eigenvalue:", eigenvalue_max)

    eigenvalue_min_p, k = power_method(matrix.I, vector, eps)
    eigenvalue_min_p = 1 / eigenvalue_min_p
    print("Computed eigenvalue:", eigenvalue_min_p, ",", k, "iterations")
    eigenvalue_min = 1 / numpy_eigenvalues(matrix.I)
    print("Smallest eigenvalue:", eigenvalue_min)


def main():
    print("Power method")
    alpha = int(input("Enter the alpha: "))
    eps = float(input("Enter the epsilon: "))

    A = [[7 + alpha, 2.5, 2, 1.5, 1],
         [2.5, 8 + alpha, 2.5, 2, 1.5],
         [2, 2.5, 9 + alpha, 2.5, 2],
         [1.5, 2, 2.5, 10 + alpha, 2.5],
         [1, 1.5, 2, 2.5, 11 + alpha]]
    b = [1, 1, 1, 1, 1]

    pm_calculation(A, b, eps)


main()
