import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def solve(L, ua, ub, E, r, p, n, m):
    h = L / n
    tau = T / m

    vector_x = np.linspace(0, L, n + 1)
    vector_t = np.linspace(0, T, m + 1)
    matrix_u = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        t = vector_t[i]
        matrix_u[i][0] = ua(t)
        matrix_u[i][n] = ub(t)

    for i in range(1, n):
        px = p(vector_x[i])
        matrix_u[0][i] = px
        matrix_u[1][i] = (
            px + (r * tau ** 2 / (2 * E * h ** 2)) *
            (matrix_u[0][i - 1] - 2 * matrix_u[0][i] + matrix_u[0][i + 1])
        )

    for i in range(2, m + 1):
        for j in range(1, n):
            matrix_u[i][j] = (
                    2 * matrix_u[i - 1][j] - matrix_u[i - 2][j] + (tau / h) ** 2 * (E / r) *
                    (matrix_u[i - 1][j - 1] - 2 * matrix_u[i - 1][j] + matrix_u[i - 1][j + 1])
            )

    return matrix_u


def animate(vector_x, vector_t, matrix_u, L, du, tau):
    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(-du, du))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(vector_x, matrix_u[i])
        return line,

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(vector_t), interval=tau, blit=True)
    plt.show()


L = 15
du = 0.1
E = 86e9
r = 8.5e6
T = 1
ua = lambda x: 0
ub = lambda x: 0


def p(x):
    if x < 0.5 * L:
        return 2 * du * x / L
    else:
        return 2 * du * (1 - x / L)


h = 0.3
tau = 0.001
n = int(L / h)
m = int(T / tau)

vector_x = np.linspace(0, L, n + 1)
vector_t = np.linspace(0, T, m + 1)
matrix_u = solve(L, ua, ub, E, r, p, n, m)

animate(vector_x, vector_t, matrix_u, L, du, tau)
