import math
import numpy as np
from prettytable import PrettyTable


def explicit_solve1(a, b, g1, g2, T, fi, k, f, n, m):
    h = (b - a) / n
    tau = T / m
    vector_x = np.linspace(a, b, n + 1)
    vector_t = np.linspace(0, T, m + 1)
    matrix_u = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        matrix_u[i][0] = g1(vector_t[i])

    for i in range(n + 1):
        matrix_u[0][i] = fi(vector_x[i])

    for i in range(1, m + 1):
        for j in range(1, n):
            matrix_u[i][j] = (
                    k * tau / (h ** 2) * (matrix_u[i - 1][j + 1] - 2 * matrix_u[i - 1][j] + matrix_u[i - 1][j - 1]) +
                    tau * f(vector_x[j], vector_t[i - 1]) + matrix_u[i - 1][j]
            )
        matrix_u[i][n] = matrix_u[i][n - 1] + h * g2(vector_t[i])

    return matrix_u


def explicit_solve2(a, b, g1, g2, T, fi, k, f, n, m):
    h = (b - a) / n
    tau = T / m
    vector_x = np.linspace(a, b, n + 1)
    vector_t = np.linspace(0, T, m + 1)
    matrix_u = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        matrix_u[i][0] = g1(vector_t[i])

    for i in range(n + 1):
        matrix_u[0][i] = fi(vector_x[i])

    for i in range(1, m + 1):
        for j in range(1, n):
            matrix_u[i][j] = (
                k * tau / (h ** 2) * (matrix_u[i - 1][j + 1] - 2 * matrix_u[i - 1][j] + matrix_u[i - 1][j - 1]) +
                tau * f(vector_x[j], vector_t[i - 1]) + matrix_u[i - 1][j]
            )
        matrix_u[i][n] = (
            k * tau / h ** 2 * (matrix_u[i - 1][n - 1] - 2 * matrix_u[i - 1][n] + (2 * h * g2(vector_t[i - 1]) +
            matrix_u[i - 1][n - 1])) + tau * f(vector_x[j], vector_t[i - 1]) + matrix_u[i - 1][n]
        )
    return matrix_u


def implicit_solve1(a, b, g1, g2, T, fi, k, f, n, m):
    h = (b - a) / n
    tau = T / m
    vector_x = np.linspace(a, b, n + 1)
    vector_t = np.linspace(0, T, m + 1)
    matrix_u = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        matrix_u[i][0] = g1(vector_t[i])

    for i in range(n + 1):
        matrix_u[0][i] = fi(vector_x[i])

    for i in range(1, m + 1):
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)

        A[0][0] = A[n][n] = 1
        A[n][n - 1] = -1
        b[0] = matrix_u[i][0]
        b[n] = h * g2(vector_t[i])

        for j in range(1, n):
            A[j][j - 1] = -k / h ** 2
            A[j][j] = 2 * k / h ** 2 + 1 / tau
            A[j][j + 1] = -k / h ** 2
            b[j] = f(vector_x[j], vector_t[j]) + matrix_u[i - 1][j] / tau

        ans = np.linalg.solve(A, b)
        matrix_u[i][:] = ans[:]

    return matrix_u


def implicit_solve2(a, b, g1, g2, T, fi, k, f, n, m):
    h = (b - a) / n
    tau = T / m
    vector_x = np.linspace(a, b, n + 1)
    vector_t = np.linspace(0, T, m + 1)
    matrix_u = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        matrix_u[i][0] = g1(vector_t[i])

    for i in range(n + 1):
        matrix_u[0][i] = fi(vector_x[i])

    for i in range(1, m + 1):
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)

        A[0][0] = A[n][n] = 1
        A[n][n - 2] = -1
        b[0] = matrix_u[i][0]
        b[n] = h * g2(vector_t[i])

        for j in range(1, n):
            A[j][j - 1] = -k / h ** 2
            A[j][j] = 2 * k / h ** 2 + 1 / tau
            A[j][j + 1] = -k / h ** 2
            b[j] = f(vector_x[j], vector_t[j]) + matrix_u[i - 1][j] / tau

        ans = np.linalg.solve(A, b)
        matrix_u[i][:] = ans[:]

    return matrix_u


a, b = 0, 2
k = 0.5
T = 0.4

fi = lambda x: 1
g1 = lambda t: math.e ** (-t)
g2 = lambda t: math.e ** (-5 * t)
f = lambda x, t: 1


all_methods = [explicit_solve1, explicit_solve2, implicit_solve1, implicit_solve2]


def get_s_for_tau(tni, matrix_u1, matrix_u2):
    sum = 0
    for i in range(len(matrix_u1[tni])):
        sum += (matrix_u1[tni][i] - matrix_u2[2 * tni][i]) ** 2
    sum /= len(matrix_u1[tni])
    return math.sqrt(sum)


def get_mod_for_tau(tni, matrix_u1, matrix_u2):
    maxx = 0
    for i in range(len(matrix_u1[tni])):
        maxx = max(maxx, matrix_u1[tni][i] - matrix_u2[2 * tni][i])
    return maxx


for method in all_methods:
    print(method.__name__)
    pretty_table = PrettyTable()
    pretty_table.field_names = ["tau", "s(tn1)", "s(tn2)", "mod(tn1)", "mod(tn2)"]

    for tau0 in [0.0001, 0.0005, 0.001]:
        n = 10
        h = (b - a) / n
        m = int(T / tau0)

        matrix_u1 = method(a, b, g1, g2, T, fi, k, f, n, m)
        matrix_u2 = method(a, b, g1, g2, T, fi, k, f, n, 2 * m)

        pretty_table.add_row([
            tau0, get_s_for_tau(5, matrix_u1, matrix_u2), get_s_for_tau(10, matrix_u1, matrix_u2),
            get_mod_for_tau(5, matrix_u1, matrix_u2), get_mod_for_tau(10, matrix_u1, matrix_u2)
        ])

    print(pretty_table)
    print()


def get_s_for_h(hni, matrix_u1, matrix_u2):
    sum = 0
    for i in range(len(matrix_u1)):
        sum += (matrix_u1[i][hni] - matrix_u2[i][2 * hni]) ** 2
    sum /= len(matrix_u1)
    return math.sqrt(sum)


def get_mod_for_h(hni, matrix_u1, matrix_u2):
    maxx = 0
    for i in range(len(matrix_u1)):
        maxx = max(maxx, matrix_u1[i][hni] - matrix_u2[i][2 * hni])
    return maxx


for method in all_methods:
    print(method.__name__)
    pretty_table = PrettyTable()
    pretty_table.field_names = ["h", "s(hn1)", "s(hn2)", "mod(hn1)", "mod(hn2)"]

    for h0 in [0.05, 0.1, 0.2]:
        tau = 0.0001
        m = int(T / tau)
        n = int((b - a) / h0)

        matrix_u1 = method(a, b, g1, g2, T, fi, k, f, n, m)
        matrix_u2 = method(a, b, g1, g2, T, fi, k, f, 2 * n, m)

        pretty_table.add_row([
            h0, get_s_for_h(5, matrix_u1, matrix_u2), get_s_for_h(10, matrix_u1, matrix_u2),
            get_mod_for_h(5, matrix_u1, matrix_u2), get_mod_for_h(10, matrix_u1, matrix_u2)
        ])

    print(pretty_table)
    print()
