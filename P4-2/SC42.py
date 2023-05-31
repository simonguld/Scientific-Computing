import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# Define global variables
Dp = 1
Dq = 8
C = 4.5
# Make a list of relevant K-values
Kval = [7,8,9,10,11,12]

# Define end points for x and y
xa, xb = 0, 40
ya, yb = 0, 40

# Set number of grid points in each unit square
point_density = 1
M = 1 + int((xb-xa) * point_density)         # Number of points between 0 and 40 (including end points) given a point_density
N = M + 2           # Number of points in [xa-dx,xb+dx] (including end points)

T = 10  # Time, at which contour plots of P and Q are plotted
dt = 0.001     # Time spacing
dx = (xb - xa) / (M-1)         # Spatial spacing (the stability of the method is dependent on dt and dx. the smaller dx the bigger dt)


def explicit_euler_solve(dy, P0, Q0, t_interval, dt):
    # Construct matrices to hold new values of Q and P
    dimensions = np.shape(P0)

    Q = Q0.astype('float')
    P = P0.astype('float')

    #index array used to update all values of P and Q (excluding the boundary) at once
    index = np.linspace(1, N - 2, M).astype('int')

    for k in np.arange(t_interval[0], t_interval[1], dt):
        # Calculate new values of rows of P using discretization in time and space.
        # The value of the first and last column (values outside the grid) are not calculated, as these are fixed
        # by the Neumann boundary conditions

        dP, dQ = dy(P, Q)

        P[index, 1:N - 1] = P[index, 1:N - 1] + dt * dP[index, 1:N - 1]
        Q[index, 1:N - 1] = Q[index, 1:N - 1] + dt * dQ[index, 1:N - 1]

        # Update values of first and last columns/rows in accordance with Neumann boundary conditions
        boundary_index = [0, N - 1]
        inner_index = [1, N - 2]

        P[boundary_index, :] = P[inner_index, :]
        P[:, boundary_index] = P[:, inner_index]

        Q[boundary_index, :] = Q[inner_index, :]
        Q[:, boundary_index] = Q[:, inner_index]

    return P, Q
def implicit_euler_solve(dy, P0, Q0, t_interval, dt, tol=1e-7, max_iterations=100):
    # Construct matrices to hold new values of Q and P
    dimensions = np.shape(P0)

    Qnew = Q0.astype('float')
    Pnew = P0.astype('float')

    index = np.linspace(1, N - 2, M).astype('int')

    for k in np.arange(t_interval[0], t_interval[1], dt):
        # Calculate new values of rows of P using discretization in time and space.
        # The value of the first and last column (values outside the grid) are not calculated, as these are fixed
        # by the Neumann boundary conditions

        # Update values of P and Q, scaled to avoid initial convergence of while loop
        P = 2 * Pnew.astype('float')
        Q = 2 * Qnew.astype('float')

        P_previous = Pnew.astype('float')
        Q_previous = Qnew.astype('float')

        iterations = 0
        while iterations < max_iterations and \
                (dimensions[0] * np.linalg.norm(Pnew - P) > tol or dimensions[0] * np.linalg.norm(Qnew - Q) > tol):
            iterations += 1
            P = Pnew.astype('float')
            Q = Qnew.astype('float')

            dP, dQ = dy(P, Q)

            fixed_point_iteration, newton_rapson = True, False

            if fixed_point_iteration:
                Pnew[index, 1:N - 1] = P_previous[index, 1:N - 1] + dt * dP[index, 1:N - 1]
                Qnew[index, 1:N - 1] = Q_previous[index, 1:N - 1] + dt * dQ[index, 1:N - 1]

            if newton_rapson:
                pass

        # Update values of first and last columns/rows in accordance with Neumann boundary conditions
        boundary_index = [0, N - 1]
        inner_index = [1, N - 2]

        Pnew[boundary_index, :] = Pnew[inner_index, :]
        Pnew[:, boundary_index] = Pnew[:, inner_index]

        Qnew[boundary_index, :] = Qnew[inner_index, :]
        Qnew[:, boundary_index] = Qnew[:, inner_index]

        if iterations == max_iterations:
            print(f' max iterations reached for for k = {k}')

    return Pnew, Qnew
#crank nicolson doesn't work as implemented
def implicit_crank_nicolson_solve(dy, P0, Q0, t_interval, dt, tol=1e-7, max_iterations=100):
    # Construct matrices to hold new values of Q and P
    dimensions = np.shape(P0)

    Qnew = Q0.astype('float')
    Pnew = P0.astype('float')

    index = np.linspace(1, N - 2, M).astype('int')

    for k in np.arange(t_interval[0], t_interval[1], dt):
        # Calculate new values of rows of P using discretization in time and space.
        # The value of the first and last column (values outside the grid) are not calculated, as these are fixed
        # by the Neumann boundary conditions

        # Update values of P and Q, scaled to avoid initial convergence of while loop
        P = 2 * Pnew.astype('float')
        Q = 2 * Qnew.astype('float')

        P_previous = Pnew.astype('float')
        Q_previous = Qnew.astype('float')

        dP_previous, dQ_previous = dy(P, Q)

        iterations = 0
        while iterations < max_iterations and \
                (dimensions[0] * np.linalg.norm(Pnew - P) > tol or dimensions[0] * np.linalg.norm(Qnew - Q) > tol):
            iterations += 1
            P = Pnew.astype('float')
            Q = Qnew.astype('float')

            dP, dQ = dy(P, Q)

            Pnew[index, 1:N - 1] = P_previous[index, 1:N - 1] + 0.5 * dt * (dP_previous[index, 1:N - 1] + dP[index, 1:N - 1])
            Qnew[index, 1:N - 1] = Q_previous[index, 1:N - 1] + 0.5 * dt * (dQ_previous[index, 1:N - 1] + dQ[index, 1:N - 1])

        # Update values of first and last columns/rows in accordance with Neumann boundary conditions
        boundary_index = [0, N - 1]
        inner_index = [1, N - 2]

        Pnew[boundary_index, :] = Pnew[inner_index, :]
        Pnew[:, boundary_index] = Pnew[:, inner_index]

        Qnew[boundary_index, :] = Qnew[inner_index, :]
        Qnew[:, boundary_index] = Qnew[:, inner_index]

        if iterations == max_iterations:
            print(f' max iterations reached for for k = {k}')

    return Pnew, Qnew
def runge_kutta_4th(dy, P0, Q0, t_interval, dt):
    # Construct matrices to hold new values of Q and P
    dimensions = np.shape(P0)

    Q = Q0.astype('float')
    P = P0.astype('float')

    # Qnew = np.empty(dimensions)
    # Pnew = np.empty(dimensions)

    index = np.linspace(1, N - 2, M).astype('int')

    for k in np.arange(t_interval[0], t_interval[1], dt):
        # Calculate new values of rows of P using discretization in time and space.
        # The value of the first and last column (values outside the grid) are not calculated, as these are fixed
        # by the Neumann boundary conditions

        Pk1, Qk1 = dy(P, Q)
        Pk2, Qk2 = dy(P + 0.5 * dt * Pk1, Q + 0.5 * dt * Qk1)
        Pk3, Qk3 = dy(P + 0.5 * dt * Pk2, Q + 0.5 * dt * Qk2)
        Pk4, Qk4 = dy(P + dt * Pk3, Q + dt * Qk3)

        P[index, 1:N - 1] = P[index, 1:N - 1] + dt/6  * (Pk1[index, 1:N - 1] + 2 * Pk2[index, 1:N - 1] + 2 * Pk3[index, 1:N - 1] + Pk4[index, 1:N - 1])
        Q[index, 1:N - 1] = Q[index, 1:N - 1] + dt/6  * (Qk1[index, 1:N - 1] + 2 * Qk2[index, 1:N - 1] + 2 * Qk3[index, 1:N - 1] + Qk4[index, 1:N - 1])

        # Update values of first and last columns/rows in accordance with Neumann boundary conditions
        boundary_index = [0, N - 1]
        inner_index = [1, N - 2]

        P[boundary_index, :] = P[inner_index, :]
        P[:, boundary_index] = P[:, inner_index]

        Q[boundary_index, :] = Q[inner_index, :]
        Q[:, boundary_index] = Q[:, inner_index]
    return P, Q

def create_contour(x_range, P, Q, K):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.contour(x_range, x_range, P[1:N - 1, 1:N - 1], origin="lower")  # extent=[0,40,0,40]
    im2 = ax2.contour(x_range, x_range, Q[1:N - 1, 1:N - 1], origin="lower")

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    ax2.set_yticklabels([])
    ax1.set_title('P(x,y)', fontweight='bold', fontsize=16)
    ax2.set_title('Q(x,y)', fontweight='bold', fontsize=16)
    fig.suptitle(f'K = {K}', fontweight='bold', fontsize=18)
    plt.show()
def dy(P, Q, dx, Dq, Dp, C, K):
    # Construct matrices to hold values of dQ and dP
    shape = np.shape(P)

    dP = np.empty(shape)
    dQ = np.empty(shape)

    index = np.linspace(1, N - 2, M).astype('int')

    # Calculate dP

    dP[index, 1:N - 1] = Dp / dx ** 2 * (P[(index + 1), 1:N - 1] + P[index - 1, 1:N - 1] + P[index, 2::]
                        + P[index, 0:N - 2]) - (K + 1 + 4 * Dp / dx ** 2 - Q[index, 1:N - 1] * P[index, 1:N - 1]) \
                        * P[index, 1:N - 1] + C
    # Calculate dQ

    dQ[index, 1:N - 1] = Dq / dx ** 2 * ( Q[index + 1, 1:N - 1] + Q[index - 1, 1:N - 1] + Q[index, 2::]
                        + Q[index, 0:N - 2] - 4 * Q[index, 1:N - 1]) \
                         + (K - Q[index, 1:N - 1] * P[index, 1:N - 1]) * P[index, 1:N - 1]

    return dP, dQ

def main():
    # Construct plots for each of the relevant K-values
    for K in Kval:


        # Initialize Q and P. Using N, we are including points just outside of boundary
        Q0 = np.zeros([N, N])
        P0 = np.zeros([N, N])

        left_initial = 1 + 10 * point_density
        right_initial = 2 + 30 * point_density

        Q0[left_initial:right_initial, left_initial:right_initial] = K / C + 0.2
        P0[left_initial:right_initial, left_initial:right_initial] = C + 0.1

        euler_implicit, euler_explicit, RK4, crank_nicolson = True, True, False, False

        if euler_implicit:
            time_initial = time.time()

            PI, QI = implicit_euler_solve(lambda P,Q: dy(P, Q, dx, Dq, Dp, C, K), P0, Q0, [0,T], dt)
            time_end = time.time()
            print(f' Runtime implicit Euler for K = {K}: ', time_end - time_initial)
            # Make contour plots
            x_range = np.linspace(xa, xb, M)
            create_contour(x_range, PI, QI, K)

        if crank_nicolson:
            time_initial = time.time()

            PI, QI = implicit_crank_nicolson_solve(lambda P,Q: dy(P, Q, dx, Dq, Dp, C, K), P0, Q0, [0,T], 10*dt)
            time_end = time.time()
            print(f' Runtime implicit CN for K = {K}: ', time_end - time_initial)
            # Make contour plots
            x_range = np.linspace(xa, xb, M)
            create_contour(x_range, PI, QI, K)

        if RK4:
            time_initial = time.time()
            PRK, QRK = runge_kutta_4th(lambda P, Q: dy(P, Q, dx, Dq, Dp, C, K), P0, Q0, [0, T], dt)
            time_end = time.time()
            print(f' Runtime RK4 for K = {K}: ', time_end - time_initial)
            # Make contour plots
            x_range = np.linspace(xa, xb, M)
            create_contour(x_range, PRK, QRK, K)

        if euler_explicit:
            time_initial = time.time()

            P, Q = explicit_euler_solve(lambda P,Q: dy(P, Q, dx, Dq, Dp, C, K), P0, Q0, [0,T], dt)
            time_end = time.time()
            print(f' Runtime explicit for K = {K}: ', time_end - time_initial)

            # Make contour plots
            x_range = np.linspace(xa, xb, M)
            create_contour(x_range, P, Q, K)


if __name__ == "__main__":
    main()







#direct approach
if 0:
    for K in Kval:
        # time runtime for each K
        time_initial = time.time()

        # Initialize Q and P. Using N, we are including points just outside of boundary
        Q = np.zeros([N, N])
        P = np.zeros([N, N])

        left_initial = 1 + 10 * point_density
        right_initial = 2 + 30 * point_density

        Q[left_initial:right_initial, left_initial:right_initial] = K / C + 0.2
        P[left_initial:right_initial, left_initial:right_initial] = C + 0.1

        # Construct matrices to hold new values of Q and P
        Qnew = np.empty([N, N])
        Pnew = np.empty([N, N])

        index = np.linspace(1, N - 2, M).astype('int')

        for k in np.arange(0, T, dt):
            # Calculate new values of rows of P using discretization in time and space.
            # The value of the first and last column (values outside the grid) are not calculated, as these are fixed
            # by the Neumann boundary conditions

            Pnew[index, 1:N - 1] = P[index, 1:N - 1] + dt * (Dp / dx ** 2 * (
                        P[(index + 1), 1:N - 1] + P[index - 1, 1:N - 1] + P[index, 2::] + P[index, 0:N - 2])
                                                             - (K + 1 + 4 * Dp / dx ** 2 - Q[index, 1:N - 1] * P[
                                                                                                               index,
                                                                                                               1:N - 1]) * P[
                                                                                                                           index,
                                                                                                                           1:N - 1]) + C * dt

            # Calculate new values of row of Q using discretization in time and space
            # The value of the first and last column (values outside the grid) are not calculated, as these are fixed
            # by the Neumann boundary conditions

            Qnew[index, 1:N - 1] = Q[index, 1:N - 1] + dt * (Dq / dx ** 2 * (
                        Q[index + 1, 1:N - 1] + Q[index - 1, 1:N - 1] + Q[index, 2::] + Q[index, 0:N - 2] - 4 * Q[
                                                                                                                index,
                                                                                                                1:N - 1])
                                                             + (K - Q[index, 1:N - 1] * P[index, 1:N - 1]) * P[
                                                                                                             index,
                                                                                                             1:N - 1])

            # Update values of first and last columns/rows in accordance with Neumann boundary conditions
            boundary_index = [0, N - 1]
            inner_index = [1, N - 2]

            Pnew[boundary_index, :] = Pnew[inner_index, :]
            Pnew[:, boundary_index] = Pnew[:, inner_index]

            Qnew[boundary_index, :] = Qnew[inner_index, :]
            Qnew[:, boundary_index] = Qnew[:, inner_index]

            # Update values of P and Q
            P = Pnew.astype('float')
            Q = Qnew.astype('float')

        time_end = time.time()
        print(f' Runtime for K = {K}: ', time_end - time_initial)

        # Make contour plots
        x_range = np.linspace(xa, xb, M)
        create_contour(x_range, P, Q, K)
