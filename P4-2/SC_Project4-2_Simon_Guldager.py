import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define constants
Dp = 1
Dq = 8
C = 4.5
# Make a list of relevant K-values
Kval = [7,8,9,10,11,12]

# Define end points for x and y
xa, xb = 0, 40
ya, yb = 0, 40
M = 41          # Number of points between 0 and 40 (including end points) when spacing dx = 1
N = M+2           # Number of points between -1 and 41 (including end points) when spacing dx = 1

T = 2000       # Time, at which contour plots of P and Q are plotted
dt = 0.01       # Time spacing
dx = 1          # Spatial spacing


# Construct plots for each of the relevant K-values
for K in Kval:

    # Initialize Q and P. Using N, we are including points just outside of boundary
    Q = np.zeros([N, N])
    P = np.zeros([N, N])
    # At t=0, Q and P take non-zero values only when x,y are between 10 and 30
    Q[11:32, 11:32] = K / C + 0.2
    P[11:32, 11:32] = C + 0.1

    # Construct matrices to hold new values of Q and P
    Qnew = np.empty([N, N])
    Pnew = np.empty([N, N])

    for k in np.arange(0, T, dt):
        # Calculate new values of row i of P using discretization in time and space.
        # The value of the first and last column (values outside the grid) are not calculated, as these are fixed
        # by the Neumann boundary conditions
        for i in range(1, N-1):

            Pnew[i,1:N-1] = P[i,1:N-1] + dt * ( Dp / dx**2 * (P[i+1,1:N-1]+P[i-1,1:N-1] + P[i,2::] + P[i,0:N-2]  )
                            - (K + 1 + 4*Dp / dx**2 - Q[i,1:N-1]*P[i,1:N-1])* P[i,1:N-1]) + C * dt

            # Calculate new values of row of Q using discretization in time and space
            # The value of the first and last column (values outside the grid) are not calculated, as these are fixed
            # by the Neumann boundary conditions

            Qnew[i,1:N-1] = Q[i,1:N-1] + dt * ( Dq * (Q[i+1,1:N-1]+Q[i-1,1:N-1] + Q[i,2::] + Q[i,0:N-2]  - 4 * Q[i,1:N-1])
                                            + (K - Q[i,1:N-1]*P[i,1:N-1])* P[i,1:N-1])

        #Update values of first and last columns/rows in accordance with Neumann boundary conditions
        Pnew[0,:] = np.copy ( Pnew[1,:] )
        Pnew[N-1, :] = np.copy( Pnew[N-2, :] )
        Pnew[:, 0] = np.copy( Pnew[:,1] )
        Pnew[:,N-1] = np.copy( Pnew[:, N-2] )

        Qnew[0, :] = np.copy(Qnew[1, :])
        Qnew[N - 1, :] = np.copy(Qnew[N - 2, :])
        Qnew[:, 0] = np.copy(Qnew[:, 1])
        Qnew[:, N - 1] = np.copy(Qnew[:, N - 2])

        #Update values of P and Q
        P = np.ndarray.copy(np.array(Pnew, dtype='float'))
        Q = np.ndarray.copy(np.array(Qnew, dtype='float'))


    # Make contour plots
    fig,(ax1,ax2)=plt.subplots(1,2)
    im1 = ax1.contour(P[1:N-1,1:N-1],origin="lower",extent=[0,40,0,40])
    im2 = ax2.contour(Q[1:N-1,1:N-1],origin="lower",extent=[0,40,0,40])

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    ax2.set_yticklabels([])
    ax1.set_title('P(x,y)',fontweight='bold',fontsize=16)
    ax2.set_title('Q(x,y)',fontweight='bold',fontsize=16)
    fig.suptitle(f'K = {K}', fontweight='bold',fontsize=18)
    plt.show()
