import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def implicit_euler(dy : Callable, y_initial : np.ndarray, interval : np.ndarray,
                   Npoints : int, tol = 1e-6, max_iterations = 100) -> np.ndarray:


    if type(y_initial) == np.ndarray:
        y = np.empty([Npoints, np.size(y_initial)])
        y[0] = y_initial.astype('float')
    else:
        y = np.empty(Npoints)
        y[0] = y_initial

    h = (interval[1] - interval[0]) / Npoints
    t0 = interval[0]

    for i in range(1,Npoints):
        y_new = y[i-1].astype('float')
        y_old = 2 * y_new
        iterations = 0
        while np.linalg.norm(y_new - y_old) > tol and iterations < max_iterations:
            iterations += 1
            y_old = y_new
            y_new = y[i-1] + h * dy (t0 + i*h, y_new)
        y[i] = y_new

    return y
def dy(t, y, k):
    return k * y

def main():
    n = 10000
    A = 1
    omega = 2
    phi0 = 0

    def x(t, A, omega, phi0):
        return A * np.cos(omega * t + phi0)
    def v(t, A, omega, phi0):
        return - A * omega * np.sin(omega * t + phi0)
    def dxdv (t, y, omega):
        return np.array([y[1], - omega ** 2 * y[0]])

    y0 = np.array([A,0])
    tmax = 10
    t = np.linspace(0, tmax, n)



    y = implicit_euler(lambda t, y: dxdv(t, y, omega), y0, np.array([0, tmax]), n)

    sigdifx = (y[:,0] - x(t, A, omega, phi0))
    sigdify =  (y[:,1] - v(t, A, omega, phi0))

    plt.plot(t, y[:,0], 'go', linewidth=0.01)
    plt.plot(t, x(t, A, omega, phi0), 'g-')


    plt.plot(t, y[:,1], 'ro', linewidth=0.01)
    plt.plot(t, v(t, A, omega, phi0), 'r-')
    plt.show()

    plt.figure()
    plt.plot(t, sigdifx,'g--')
    plt.plot(t,sigdify,'r--')
    plt.show()


if __name__ == "__main__":
     main()


