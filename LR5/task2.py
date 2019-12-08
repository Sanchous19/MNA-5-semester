import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot, animation


def solve():
    vector_x = np.linspace(-a / 2, a / 2, n + 1)
    vector_y = np.linspace(-b / 2, b / 2, m + 1)
    matrix_u = np.zeros((k + 1, m + 1, n + 1))

    for i in range(m + 1):
        for j in range(n + 1):
            matrix_u[0][i][j] = u0(vector_x[j], vector_y[i])

    for i in range(1, m):
        for j in range(1, n):
            matrix_u[1][i][j] = (
                u0(vector_x[j], vector_y[i]) + tau * ud0(vector_x[j], vector_y[i]) +
                tau ** 2 * ((matrix_u[0][i - 1][j] - 2 * matrix_u[0][i][j] + matrix_u[0][i + 1][j]) / h2 ** 2 +
                (matrix_u[0][i][j - 1] - 2 * matrix_u[0][i][j] + matrix_u[0][i][j + 1]) / h1 ** 2)
            )

    for p in range(k):
        for j in range(1, n):
            matrix_u[p + 1][0][j] = (
                2 * matrix_u[p][0][j] - matrix_u[p - 1][0][j] + tau ** 2 * (2 * (matrix_u[p][1][j] -
                matrix_u[p][0][j]) / h2 ** 2 + (matrix_u[p][0][j - 1] -
                2 * matrix_u[p][0][j] + matrix_u[p][0][j + 1]) / h1 ** 2)
            )
            matrix_u[p + 1][m][j] = (
                2 * matrix_u[p][m][j] - matrix_u[p - 1][m][j] + tau ** 2 * (
                2 * (matrix_u[p][m - 1][j] - matrix_u[p][m][j]) / h2 ** 2 + (
                matrix_u[p][m][j - 1] - 2 * matrix_u[p][m][j] + matrix_u[p][m][j + 1]) / h1 ** 2)
            )

    for p in range(k + 1):
        for j in range(n + 1):
            matrix_u[p][0][j] = 0
            matrix_u[p][m][j] = 0

    for p in range(1, k):
        for j in range(1, n):
            for i in range(1, m):
                matrix_u[p + 1][i][j] = (
                    2 * matrix_u[p][i][j] - matrix_u[p - 1][i][j] + tau ** 2 * ((matrix_u[p][i - 1][j] -
                    2 * matrix_u[p][i][j] + matrix_u[p][i + 1][j]) / h2 ** 2 + (matrix_u[p][i][j - 1] -
                    2 * matrix_u[p][i][j] + matrix_u[p][i][j + 1]) / h1 ** 2)
                )

    return matrix_u


def animate(a, b, vector_x, vector_y, matrix_u):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate_surface(frame):
        axes = ax
        axes.clear()

        axes.set_xlim(-a / 2, a / 2)
        axes.set_ylim(-b / 2, b / 2)
        axes.set_zlim(-3, 3)

        x, y = np.meshgrid(vector_x, vector_y)
        surface = axes.plot_surface(y, x, matrix_u[frame], cmap='magma')

        return surface,

    ani = animation.FuncAnimation(fig, animate_surface, interval=1, frames=k)
    pyplot.show()


a, b = 2, 3
T = 3

u0 = lambda x, y: math.tan(math.cos(math.pi * y / b))
ud0 = lambda x, y: math.exp(math.sin(math.pi * x / a)) * math.sin(2 * math.pi * y / b)
h1, h2 = 0.1, 0.1
tau = 0.01
n = int(a / h1)
m = int(b / h2)
k = int(T / tau)

vector_x = np.linspace(-a / 2, a / 2, n + 1)
vector_y = np.linspace(-b / 2, b / 2, m + 1)
matrix_u = solve()

animate(a, b, vector_x, vector_y, matrix_u)
