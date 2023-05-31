import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
from LJhelperfunctions import V,EPSILON,SIGMA,distance,gradV,flat_V,flat_gradV
NA = newaxis

np.set_printoptions(precision=14, suppress=True)

##Helt styr på NA. Apply og forstå funktioner
##Styr på egne funktioner ad 2 omgange
###implementere inv bfgs
##Indføre metaheuristic methods

data = np.load('ArStart.npz')
Xstart2 = data['Xstart2']
Xstart3 = data['Xstart3']
Xstart4 = data['Xstart4']
Xstart5 = data['Xstart5']
Xstart6 = data['Xstart6']
Xstart7 = data['Xstart7']
Xstart8 = data['Xstart8']
Xstart9 = data['Xstart9']
X20 = data['Xopt20']
Xopt20 = X20.reshape(-1,60)

# Equilibrium distance for potential
r0 = 2 ** (1 / 6) * SIGMA


### James' helper functions for making nice 3D plots:
def create_3d_plot():
    plot_dict = dict(projection='3d')
    fig, ax = plt.subplots(subplot_kw=plot_dict)
    return fig, ax
def make_axis_equal(ax):
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    z1, z2 = ax.get_zlim()
    m1 = min(x1, y1, z1)
    m2 = max(x2, y2, z2)
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)
    ax.set_zlim(m1, m2)
    ax.set_proj_type("ortho")
    return ax
def transparent_axis_background(ax):
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # Disable the axes
    ax.axis("off")
    return ax
if 0:
    def plot_min_distances(ax, points, cutoff=1.01 * r0):
        # points: (N,3)-array of (x,y,z) coordinates for N points
        # distance(points): returns (N,N)-array of inter-point distances:
        displacement = points[:, np.newaxis] - points[np.newaxis, :]
        r = np.sqrt(np.sum(displacement*displacement, axis=-1))
        # Grab only those that are in upper triangular part
        r = np.triu(r)
        # Grab lengths that are above 0, but below the cutoff
        mask = ( r <= cutoff) * (r > 0) * (0.99 * r0 < r)
        # Grab the indices of elements that satisfy the mask above
        ii, jj = np.where(mask)
        # For each pair of indices (corresponding to a distance close to optimal)
        # we plot a line between the corresponding points
        for i, j in zip(ii, jj):
            p = points[[i, j]]
            ax.plot(*p.T, color='k', ls="-")
        return ax


def plot_min_distances(ax, points, minlen = r0):
    # points: (N,3)-array of (x,y,z) coordinates for N points
    # distance(points): returns (N,N)-array of inter-point distances:
    displacement = points[:, np.newaxis] - points[np.newaxis, :]
    r = np.sqrt(np.sum(displacement * displacement, axis=-1))

    # Grab only those that are in upper triangular part
    r = np.triu(r)

    # Grab lengths that are above 0, but below the cutoff
    mask = (r <= 1.3 * minlen) * (r > 0)

    # Grab the indices of elements that satisfy the mask above
    ii, jj = np.where(mask)

    # For each pair of indices (corresponding to a distance close to optimal)
    # we plot a line between the corresponding points
    for i, j in zip(ii, jj):
        p = points[[i, j]]
        # 0->1 0.1->0.5 0.3->0
      #  print (abs(r[i,j]))
        bond_strength = 1 - min(abs(r[i, j] - minlen), 0.3) / 0.3
       # print(bond_strength)
        ax.plot(*p.T, color='k', ls="-", alpha=bond_strength)
    return ax

def create_base_plot(points, cutoff= 1.01 * r0):
    # Create the figure and 3D axis
    fig, ax = create_3d_plot()
    # translate points so centre of mass is at origin
    m = np.mean(points, axis=0)
    points = np.array(points) - m[np.newaxis, :]
    # Plot points and lines between points that are near to optimal
    ax.scatter(*points.T)
    ax = plot_min_distances(ax, points, cutoff)
    return ax, points
def make_plot_pretty(ax):
    # Make the plot pretty
    ax = transparent_axis_background(ax)
    ax = make_axis_equal(ax)
    return ax

if 0:
    amp = 1.075
    X00 = np.array([[0,0,0],[amp * r0,0,0]])
    ax, points = create_base_plot(X00,r0)
    ax = make_plot_pretty(ax)
    plt.title(f"Minimum energy configuration of Argon atoms", fontsize=15)
    plt.show()




### My own functions:

### Functions for solving matrix equations (which we will need in the BFGS function):

def max_norm_index(A,k):
    ##Find the index and value of the column in submatrix A[k::, k::] having the greatest Euclidean norm
    m, n = np.shape(A)
    norm_list = np.empty(n-k)
    for i in np.arange(n-k):
        norm_list[i] = np.linalg.norm(A[k::, k+i])
    index = np.argmax(norm_list)
    #return the index of the column in A[k::,k::] having the greatest Euclidean norm along with that norm.
    return index + k, norm_list[index]

def backward_substitution(A,b):
    n,n = np.shape(A)
    y = np.zeros(n)                                          # Initial values of y set to 0
    for k in np.arange(n-1,-1,-1):
        if A[k,k]==0:                                        #Abort if diagonal element vanishes
            print("\n","Vanishing diagonal entries!","\n")
            return
        else:                                               #Calculate values using a dot product to avoid a for-loop.
             y[k] = ( b[k]-np.dot(A[k,:],y) ) / A[k,k]      #By setting the initial values of y to zero, the full
                                                            # dot product can be used every time, thus avoiding
                                                            # complicated indexing
    return y

def apply_reflection(A, v):             #v an mx1 vector, A an mxn matrix
    c = - 2 * np.dot(v, A)              #Calculate the 1xn row vector transpose(v)*A
    A = A +c * v[:,np.newaxis]          # Perform reflection of A by creating a mxn matrix v[:,np.newaxis],
                                        # i.e a matrix whose columns are all v, then multiplying each row elementwise
                                        # with c, and adding it to A
    return A

def qr_pivot_solve(A,b):                                    #QR-solver with column pivoting
    ###
    # Transform an mxn matrix A to QRP^T, where Q is orthogonal, R[0:n,:] is upper triangular and P is a permutation
    # matrix chosen so that at each step, the column with the largest norm of the remaining matrix R[k::,k::] is switched with the
    # k'th column. This collects all j linearly independent columns to the left, resultning in the factorization
    # A = Q [(R, S), (0 , 0)], where the reduced jxj matrix R is upper triangular and invertible, and the
    # jx(n-j) matrix S contains the singular part, i.e. spans the kernel of A. The solutions now take the form
    # Ax = [c_image, c_kernel, c_overdetermined]. The reduced system
    # Q @ R[0:j,0:j] @ (P^T @ x) [0:j] = c_image = Q^T @ b [0:j] can now be solved for x[0:j] - the invertible part of the system.
    #  --> x = P @ (x[0,j],zeros(m-j),
    # which is the full solution. [0's as solutions to the last m-n equations is simply because A can maximally span
    # n dimensions. Setting the n-j components mapping to the kernel to 0 is a particular solution, but we only
    # care about the invertible part of the system.

    # if A has full column rank, then upper part of A, namely A[0:n,:] is invertible.
    # Since orthogonal transformations preserve norms, they preserve the (norm defined) solution to the least squares
    # problem Ax ~ b, meaning that QRx = [c1, c2], where Rx = Q^t c1 = Q^t b, which can be solved by back substitution.

    # Applying a househoulder reflection vector v1 reflects each column vector in a in the direction of the v1'th
    # basis vector. Therefore, the column vector in the remaining submatrix A[:,1::] with the biggest norm is most
    # orthogonal to the first basis vector. We therefore switch that column with the second one.
    # Similarly, after applying v2, the column in A[:,2::] with the biggest norm is most orthogonal to the first two
    # basis vectors and is swapped with the third row.
    # In this way, we make sure to collect all linearly indepedent columns to the left in A, thus constructing R.
    # If the submatrix A[0:n,0:n] has rank k<n, the remaining n-k columns will be linearly dependent on the first k
    # columns. These columns represent S, the kernel part of A

    ## OBS: It is unnecessary and inefficient to calculate Q explicitly. We simply apply the transformations directly to
    # b
    ###

    m,n = np.shape(A)                                       # A an m x n matrix
    b_copy = np.ndarray.copy(np.array(b,dtype='float'))
    #Qt = np.identity(m)                                     #Q_transpose
    R = np.ndarray.copy(np.array(A,dtype='float'))
    P = np.linspace(0,n-1,n)                                #List keeping track of permutations

    for k in np.arange(n-1):
                                                            #Perform pivoting:
        max_index,max_norm = max_norm_index(R,k)            #Use max_norm_index function (defined above) to extract
                                                            #index corresponding to column of R[k::,k::] with biggest 2-norm
        R[:,[k,max_index]] = R[:,[max_index,k]]             # Pivot columns
        P[[k,max_index]] = P[[max_index,k]]

        a = R[k:m,k]                                        # Carry out QR-calculations as usual
        alpha = - np.copysign(1,a[0]) * np.linalg.norm(a)

        v = np.ndarray.copy(a)
        v[0] = v[0]-alpha                                   #construct Householder vector
        v = v / np.linalg.norm(v)                           #normalize

        R[k:m,k:n] = apply_reflection(R[k:m,k:n],v)     #Calculate values on entire sub-matrix to avoid for-loop
       # Qt[k:m,:] = apply_reflection(Qt[k:m,:],v)       #On Qt, we apply the reflection on all columns, as
        b_copy[k:m] = b_copy[k:m] - 2 * np.dot(v,b_copy[k:m]) * v
                                                        #columns Qt[k:m,i] do not necesarily vanish for i<k

    P_matrix = np.zeros([n,n])                         #Construct permutation matrix[permutes columns]
    for i in np.arange(n):
        P_matrix[int(P[i]),i]=1

    #Find dimension of "economy size" R, a rxr invertible matrix
    R_index = np.argwhere(np.abs(np.diag(R)) >1e-12)   # Find indices of non-vanishing diagonal entries

    r = 1 + int (max (R_index))                        # Find biggest index of non-vanishing diagonal entry
    R = R[0:r,0:r]                                     # Construct "economy size" R

    #y = Qt @ b                                          # Solve right hand side to obtain mx1 matrix

    x = backward_substitution(R,b_copy[0:r])           # Solve invertible part of system to obtain rx1 matrix solution
    x =  P_matrix @ np.block([x,np.zeros(n-r)])        # Obtain full nx1 solution by permutation of [x,np.zeros(n-r)]
    return   x

def pot_calculator(X0,x,a = 0, b = 0):
    """
    Calculate the Lennard Jones potential between N+1 particles, with N particles fixed in X0 and x allowed to vary
    with respect to the x-axis
    :param X0: N partciles in the format N x 3
    :param x: real value defining the free vector (x,0,0)
    :return:
    """
    X = np.r_['0,2',X0,np.array([x,a,b])]
    return V(X)

def potential_plotter(X0,xmin,xmax, points = 1e3):
    """
    This function calculates and plots the potential of N+1 particles, with N particles being fixed and contained
    in X0, and the last particle having coordinates (x,0,0). The plot is calculated in the range [xmin,xmax]
    """
    #contruct matrix of all points:

    pot_calculator_vectorized = np.vectorize(lambda x: pot_calculator(X0,x))
    x_range = np.linspace(xmin,xmax,int(points))
    potential_values = pot_calculator_vectorized(x_range)

    plt.plot(x_range,potential_values,'r-',label = 'Potential energy function')
    plt.ylabel("Potential energy (kJ/mol)", fontsize=16)
    plt.xlabel("First coordinate position (Å)", fontsize=16)
    plt.grid('True')
    plt.ylim(-2, 4)
    plt.xlim(xmin, xmax)
    plt.show()

def bisection_root(f, a, b, tolerance=1e-13):
    fa, fb = f(a), f(b)
    if np.sign(fa) == np.sign(fb):             # f(a) and f(b) must have opposite signs
        print('Bracket condition not met')
        return
    else:
        k, fm = 0, 1                                 # Count number of iterations. Set fm=1 to activate while loop

        while np.abs(fm) > tolerance:                # Continue until convergence criterion is met
            m = a + (b - a) / 2                    # Write midpoint like this to avoid ending up outside of interval
            k, fm = k+1, f(m)                        # Increase k, store f(m) and reuse to avoid additional evaluations of f
            if np.sign(fa) == np.sign(fm):          # If f(a) and f(m) has the same sign, there is a root in [m,b]
                a, fa = m, fm
            else:                                  # If f(b) and f(m) has the same sign, there is a root in [a,m]
                b, fb = m, fm
        times_called = 2+k                         #We call f two times initially and then 1 time per loop
        return m, times_called

def bisection_root_new(f, a, b, tol = 1e-13):
    """
    Assuming a bracket in [a,b], find a value of s.t. f(x) = 0 and return value of x along with no. of f-calls
    Linear convergence. covergence guaranteed given a bracket.
    :param f:
    :param a:
    :param b:
    :param tol:
    :return:
    """

    # initialize:
    m = a + (b - a) / 2
    fa, fb, fm = f(a), f(b), f(m)

    f_calls = 3
    iterations = 0
    max_iterations = 500

    # Verify bracket condition. f(a) and f(b) must have the same sign to ensure a bracket
    if np.sign(fa) * np.sign(fb) > 0:
        print ("Bracket condition not met")
        return

    while np.abs(fm) > tol and iterations < max_iterations:
        iterations += 1
        f_calls += 1
        # if fa and fm have the same sign, the root must be in [m,b]. Update a to m and find new midpoint value
        if np.sign(fa) * np.sign (fm) > 0:
            a, fa = m, fm
            m = a + (b - a) / 2
            fm = f(m)
        #Otherwise, fm and fb have the same sign and root must be in [a,m]. Update value of b and find new mid point
        else:
            b, fb = m, fm
            m = a + (b - a) / 2
            fm = f(m)

    return m, f_calls

def pot_derivative(r, epsilon = EPSILON, sigma = SIGMA):
    value = 4 * epsilon * ( 6 * np.power(sigma,6) / np.power(r,7) - 12 * np.power(sigma,12) / np.power(r, 13))
    return value

def newton_root(f, df, x0, tolerance = 1e-13, max_iterations = 500):
    #initialize
    x = x0
    fx = f(x0)
    f_calls = 1
    iterations = 0

    while np.abs(fx) > tolerance:
        iterations += 1
        x = x - fx/df(x)
        fx = f(x)
        f_calls += 2

    if iterations == max_iterations:
        print(f"Newton root solver not converged after {max_iterations} iterations.")
        return
    return x, f_calls

def bisection_newton_root_ext(f,df,a,b,tolerance=1e-12):
    fa, fb = f(a), f(b)
    if np.sign(fa) == np.sign(fb):  # f(a) and f(b) must have opposite signs
        print('Bracket condition not met')
        return
    else:
        fm, fx, x = 1, fa, a+(b-a)/2          # Intialize values. fx=fa, fm is updated anyway. x is chosen as midpoint
        k, max_iterations = 2, 1000      #k counts # of times f i scalled
        while True and k<max_iterations:                     # Continue until convergence criterion is met
            m = a + (b - a) / 2         # Write midpoint like this to avoid ending up outside of interval
            k, fm = k + 1, f(m)         # Store fm and update k for calling f(m) and df(m) or df(x)
            if np.abs(fm) < tolerance:
                return m,k     #Return if convergence criterion met

            #If abs(fm)<abs(fx), then m is closer to the solution (at least if there is only one root in [a,b]),
            #and we therefore use m to calculate new value of x via Newton's root method:
            if np.abs(fx) > np.abs(fm):               #If fm is closest to 0, calculate x_new as newton root from m
                x_new = m - fm/df(m)
                k += 1

            #If x, on the other hand, is closest to the solution, we use this value as in Newton's root method
            else:
                x_new = x - fx / df(x)                #If fx is closest to 0, calculate x_new as newton root from x
                k += 1

            # If x_new lies within [a,b], we make x_new a new endpoint of the interval:
            if a < x_new < b:                         # Check that x_new lies within interval
                k,fx = k+1,f(x_new)                   # update k and store new function value
                if np.abs(fx) < tolerance:
                    return x,k #Return if convergence criterion met

                if np.sign(fa) == np.sign(fx):      #If f(a) has the same sign as f(x_new), there is a root in [x_new,b]
                    a,fa = x_new,fx
                else:  # If f(b) and f(x_new) has the same sign, there is a root in [a,x_new]
                    b, fb = x_new, fx
                x = x_new
            # If x_new lies outside [a,b], we make m a new endpoint of the interval (standard bisection method)
            else:
                if np.sign(fa) == np.sign(fm):         # If f(a) and f(m) has the same sign, there is a root in [m,b]
                    a, fa = m, fm
                else:                                  # If f(b) and f(m) has the same sign, there is a root in [a,m]
                    b, fb = m, fm


def bisection_newton_root(f,df, xmin, xmax, x0, tolerance = 1e-13, max_iterations = 500):
    """
    This implementation combines the superior convergence of the Newton root solver with the
    guarantee of convergence of the bisection implementation given a bracket

    As a default, x is updated according to the newton iteration scheme, and the bracket [a,b] reduced in size
    each time according to the sign of the function at the end points. If x ends up outside of the interval,
    the method will instead update x according to the bisection scheme

    :param f:
    :param df:
    :param xmin:
    :param xmax:
    :param x0:
    :param tolerance:
    :param max_iterations:
    :return:
    """
    #initialize
    x, a, b = xmin + (xmax-xmin)/2, xmin, xmax
    fa, fb = f(xmin), f(xmax)
    fx = f(x)
    f_calls = 3
    iterations = 0

    while np.abs(fx) > tolerance and iterations < max_iterations:
        iterations += 1
        x = x - fx / df(x)
        f_calls += 1
        #check that Newton update yields x within bracket
        if a < x < b:
            fx = f(x)
        else:
            x = a + (b-a)/2
            fx = f(x)
        f_calls += 1
        #bracket is in [x,b].
        if np.sign(fa)*np.sign(fx) > 0:
            a = x
            fa = fx
        else:
            b = x
            fb = fx
    return x, f_calls

def linesearch(F,X0,d, alpha_max, tol = 1e-13, max_iterations = 500):
    """
    Find the root of the vector function F along the line X0-d
    :param F: 
    :param X0: 
    :param tol: 
    :param max_iterations: 
    :return: 
    """""
    # 1st: Define the restriction of F along this line
    def F_restricted(x, X0, d):
        value = np.sum ( d * F(X0 + x*d))
        return value
    def F_restricted_true(x, X0 = X0, d = d):
        value = np.sum ( d * F(X0 + x*d))
        return value

    #F_restricted(x, X0, d)
    alpha, f_calls = bisection_root(F_restricted_true, 0, alpha_max,tol)

    return alpha, f_calls

def golden_section_min(f, a, b , tolerance = 1e-4):

    #initialize
    tau = (np.sqrt(5) - 1) / 2

    # defining x1 and x2 as below ensures that when updating/restricting the interval at each step, the points relative
    # positions will still be tau and 1-tau rel. to the new interval, hence we only need to call f once per iteration
    x1 = a + (1 - tau) * (b - a)
   # x1 = b - tau*(b-a)
    x2 = a + tau * (b - a)

    f1, f2 = f(x1), f(x2)
    f_calls = 2

    while np.abs(a - b) > tolerance:
        f_calls += 1
        # in this case, the minimum must be in [x1,b]. Therefore, restrict interval by letting a = x1
        # the position of x2 rel. to x1 [the new a] is tau, and so we can let x1=x2 and only calculate the new x2
        if f1 > f2:
            a = x1
            x1, f1 = x2, f2
            x2 = a + tau * (b - a)
            f2 = f(x2)
        #  else f1<f2, and the minimum must be in [a,x2]. Therefore let b = x2. the position of x1 is tau relative to
        # x2 [the new b], and we can thus let x2=x1 and only calculate the new x2
        else:
            b = x2
            x2, f2 = x1, f1
            x1 = a + (1 - tau) * (b - a)
         #   x1 = b - tau * (b - a)
            f1 = f(x1)
    #minimum in [a,x2]
    if f1 < f2:
        return a + (x2-a) / 2, f_calls
    #minimum in [x1,b]
    else:
        return x1 + (b-x1) / 2, f_calls

if 0:
    import math

    def golden_section_min_wik(f, a, b, tolerance = 1e-8):
        invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
        invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2
        (a, b) = (min(a, b), max(a, b))
        h = b - a
        if h <= tolerance:
            return (a, b)

        # Required steps to achieve tolerance
        n = int(math.ceil(math.log(tolerance / h) / math.log(invphi)))

        c = a + invphi2 * h
        d = a + invphi * h
        yc = f(c)
        yd = f(d)

        for k in range(n - 1):
            if yc < yd:  # yc > yd to find the maximum
                b = d
                d = c
                yd = yc
                h = invphi * h
                c = a + invphi2 * h
                yc = f(c)
            else:
                a = c
                c = d
                yc = yd
                h = invphi * h
                d = a + invphi * h
                yd = f(d)

        if yc < yd:
            return (a, d)
        else:
            return (c, b)

    def golden_section_min(f,a,b,tolerance=1e-3):
        tau = (np.sqrt(5)-1)/2
        x0, x1  = b - tau*(b-a), a + tau*(b-a)
        f0, f1  = f(x0), f(x1)

        n_steps = int(np.ceil(math.log2((b-a)/tolerance)/math.log2(1/tau)))

        for i in range(n_steps):
            assert((a<x0)&(x0<x1)&(x1<b))
            if f0<f1:
                x0, x1, b = x1 - tau*(x1-a), x0, x1 # Reduce upper bound\n",
                f0, f1    = f(x0), f0
            else:
                a, x0, x1 = x0, x1,  x0+tau*(b-x0)  # Increase lower bound\n",
                f0, f1    = f1, f(x1)

        if f0<f1: # Minimum is in [a;x1]\n",
            return a+(x1-a)/2,  n_steps + 2
        else:     # Minimum is in [x0;b]\n",
           return x0+(b-x0)/2, n_steps + 2

if 0:
    def BFGS_new(f, gradf, X0, tolerance = 1e-6, max_iterations = 10000, linesearch = False, inverse = False):
        """

        :param f:
        :param gradf:
        :param X0: flat 3n vector!
        :param tolerance:
        :param max_iterations:
        :param linesearch:
        :param inverse:
        :return:
        """
        #initialize
        dimension = np.size(X0.flatten())
        f_calls = 0
        iterations = 0

        B = np.eye(dimension)
        X = X0.astype('float')
        grad = gradf(X)
        dX = -0.01 * grad
        grad_new = gradf(X-dX)
        X_new = X - dX
        dY = grad_new-grad
        f_calls += 2

        while iterations < max_iterations:
            if np.linalg.norm(grad_new) < tolerance:
                return X_new, f_calls, True
            iterations += 1

            if inverse:
                B += np.outer(dX,dX) / np.dot(dX, dX) - np.dot(B @ dY, dY @ B) / np.dot (dY, B @ dY)
                dX = - B @ grad_new
            else:
                B += np.outer(dY,dY) / np.dot(dY, dY) - np.dot( B @ dX, dX @ B) / np.dot(dX, B @ dX)
                dX = qr_pivot_solve(B,-grad_new)

            X = X_new.astype('float')
            grad = grad_new.astype('float')
            if linesearch and np.linalg.norm(grad_new) > 0.5:
                alpha, n_calls = golden_section_min(lambda t: f(X+ t * dX), 0, 1)
                f_calls += n_calls
                X_new = X + alpha * dX
            else:
                alpha = 0.25
                X_new = X + alpha * dX
            grad_new = gradf(X_new)
            f_calls += 1
            dY = grad_new - grad

        return X_new, f_calls, False

def wolfe_linesearch(f,gradf,x,s,gradfx,max_iterations=100): #Takes flat vectors and functions as inputs. gradfx = gradf(x)
    # This function finds a step size satisfying the weak Wolfe conditions
    c1, c2= 1e-4, 0.9                           # Common values for the constants c1 and c2
    a, b, t = 0, np.inf, 1                      # Initialize a,b and t
    fx = f(x)                                   # Store value of f(x)

    if np.linalg.norm(gradfx)>1:
        max_iterations = 20
    else:
        max_iterations = 100

    for k in np.arange(max_iterations):

    #We find t by performing a kind of bisection procedure on the interval [a,b].

        # If 1st Wolfe condition not satisfied, update right end of interval (set b=t) and set t to be the midpoint of
        # the interval [a,b]
        if f(x+t*s) > fx + c1 * t * np.dot(s , gradfx):
            b = t
            t = 1/2 * (a + b)

        # If 2nd Wolfe condition not satisfied, update left end of interval (set a=t) and let t be midpoint of the
        # interval [a,b] (unless b=inf)
        elif np.dot(s,gradf(x+t*s)) < c2 * np.dot (s, gradfx):    #If 2nd Wolfe condition not satisfied, update a and t
            a = t
            if b == np.inf:
                t = 2*a
            else:
                t = 1/2 * (a + b)
        else:
            return t,k+2,gradf(x+t*s)   #Return optimal step size t, number of function calls and new gradient value

        #If Wolfe conditions are not met within max_iterations, return t=1, so f_new = f(x+1*s), and line search
        #failed. Also return number of function calls of f and grad f along with new gradient value
    return 1,k+2,gradf(x+s)

def BFGS(f, gradf, X0, tolerance = 1e-6, max_iterations = 10000, n_reset = 100, linesearch = False, inverse = True):
    """

    :param f:
    :param gradf:
    :param X0: flat 3n vector!
    :param tolerance:
    :param max_iterations:
    :param n_reset:
    :param linesearch:
    :param inverse:
    :return:
    """
    #initialize
    dimension = np.size(X0.flatten())
    iterations = 0
    converged = False

    B = np.eye(dimension)
    X = X0.astype('float')
    grad = gradf(X)
    dX = - 0.001 * grad
    #grad_new = gradf(X-dX)
   # X_new = X - dX
    #dY = grad_new-grad
    f_calls = 1

    while iterations < max_iterations:
        if np.linalg.norm(grad) < tolerance:
            converged = True
            break
        iterations += 1
      #  if iterations % 1000 == 0:
      #      print (iterations)
        if iterations % n_reset == n_reset - 1:
            B = np.eye(dimension)

        if linesearch:
            alpha, n_calls = golden_section_min(lambda t: f(X + t * dX), 0, 1,tolerance=5e-3)
            #alpha, steps, grad_new = wolfe_linesearch(f, gradf, X, dX, grad)
            f_calls += n_calls
        else:
            alpha = 0.25

        X_new = X + alpha * dX
        grad_new = gradf(X_new)
        f_calls += 1
        dY = grad_new - grad

        denominator = np.dot(dY, dX)
        normX, normY = np.sqrt(np.dot(dX, dX)), np.sqrt(np.dot(dY, dY))

        #deal with dY orthogonal to dX
        if denominator / (normX * normY) < 1e-10:
            denominator = normX * normY


        if inverse:
            BdY = B @ dY
            B += np.outer(dX,dX) / denominator - np.outer(BdY, BdY) / np.dot (dY, BdY) #np.outer(BdY, BdY) / np.dot (dY, BdY)
            dX = - B @ grad_new
        else:
            BdX = B @ dX
            B += np.outer(dY, dY) / denominator - np.outer(BdX,BdX) / np.dot(dX, BdX)
            dX = qr_pivot_solve(B,-grad_new)

        X = X_new.astype('float')
        grad = grad_new.astype('float')

    return X, f_calls, converged



# Quest a:

x1 = np.array([0,0,0],dtype='float')
x2 = np.array([14,0,0],dtype='float')
x3 = np.array([7,3.2,0],dtype='float')
X0 = np.block([[x1],[x2],[x3]])

if 0:
    # Potential of 2 particle
    print(potential_plotter(x1,3,11))

    # Potential of 4 particles
    print(potential_plotter(X0,3,11))

# Ques b: test bisection root on to particle system to verify that V(x)=0 when x = SIGMA

z = lambda x: pot_calculator(x1,x)
print(z(SIGMA))

if 0:
    xmin, f_calls = bisection_root(lambda x: pot_calculator(x1,x),2,6)
    print ("root and calls bisection", xmin, f_calls)

    xmin, f_calls = newton_root(lambda r: pot_calculator(x1,r), pot_derivative, x0=2)

    print ("root and calls newton", xmin, f_calls)

    xmin, f_calls = bisection_newton_root(lambda r: pot_calculator(x1,r), pot_derivative,xmin=2, xmax=6, x0=2)

    print ("root and calls combined", xmin, f_calls)

#ques e: Plot gradient
def grad_calculator(x,X0):
    #place the variable particle on top, so that grad[0,0] is the total negative force acting on x1
    X = np.r_['0,2',np.array([x,0,0], dtype = 'float'), X0]
    grad = gradV (X)
    return grad[0,0]


if 0:
    xmin, xmax = 3, 11
    #vectorize grad_calculator for X0 = x1
    grad_calculator_vec = np.vectorize(lambda x: grad_calculator(x,x1))
    grad_root,calls = bisection_root(lambda x: grad_calculator(x,x1),3,11)

    plt.plot(grad_root*np.ones(2),np.linspace(-5,5,2),'k--')
    range = np.linspace(3,11,100)
    grad_values = grad_calculator_vec(range)
    plt.plot(range,grad_values,'g--', label = 'gradient ')
    print(potential_plotter(x1, xmin, xmax))

# quest f: implement linesearch and test

if 0:
    x0 = np.array([4,0,0],dtype='float')
    X0 = np.r_['0,2',x0,X0]
    print(X0)

    alpha, f_calls = linesearch(gradV,X0,-gradV(X0),1)

    print ("t_value that minimizes F as well as f_calls", alpha, f_calls)


# quest g: golden_section. Check that you reobtain alpha from linesearc and find r0
if 1:
    # in f, we found an alpha that minimizes the potential along X0+alpha*d. We check whether alpha is reobained using golden

    V_restricted = lambda t: V(X0 - t * gradV(X0))
    alpha, f_calls = golden_section_min(V_restricted, 0,1)
    print("golden section alpha, f_calls", alpha, "   ", f_calls)


#quest h: bfgs with line

if 1:

    # Collect and index intial particle configurations to initialize for-loop
    collection = [Xstart2, Xstart3, Xstart4, Xstart5, Xstart6, Xstart7, Xstart8, Xstart9, Xopt20]
    index = [2, 3, 4, 5, 6, 7, 8, 9, 20]
    # Collect solutions
    minimum_configurations = []
    calls_total_without = 0
    print("Using the BFGS-minimizer without linesearch to find the minimum energy configurations, we find that:")
    for k in np.arange(len(index)):
        x, f_calls,converged = BFGS(flat_V, flat_gradV, collection[k], linesearch=False)
        X = x.reshape(-1, 3)  # Reshape solution to Nx3 matrix
        minimum_configurations.append(X)  # Collect solution
        calls_total_without += f_calls
        n, n = np.shape(distance(X))
        # Calculate av. interatomic distances by summing all elements and dividing by n^2-n [we subtract vanishing diagonal elements]
        av_distance = np.sum(distance(X)) / (n ** 2 - n)
        av_distance_unit_r0 = av_distance / r0  # Calculate average distance in units of equilibrium distance r0
        # Calculate number of distances within 1% of equilibrium distance. Divide by two to avoid counting each distance twice.
        if index[k] > 8:
            ideal_distances = 0.5 * np.sum(np.abs((distance(X) - r0) / r0) <= 0.02)
        else:
            ideal_distances = 0.5 * np.sum(np.abs((distance(X) - r0) / r0) <= 0.01)

        print(f"For {index[k]} particles: ",
              f"# of distances equal to equilibrium distance within 1 % = {ideal_distances},  ", f" "
                                                                                                 f"\n Average interatomic distance in units of eq. distance = {av_distance_unit_r0:.4f}, ",
              f" # of functions calls = {f_calls}. Converged = {converged}")
    print()
    print(
        "We see that while the BFGS-minimizer does converge for all systems in question, it performance is inconsistent"
        "for N > 3.\n Note that the reasonable average distance for N=4 is obtained because two particles are closer"
        " than the equilibrium distance,\n which is physically unacceptable for a minimum energy configuration.")
    print()
    print("Total # of calls without linesearch = ", calls_total_without)

    # Now we will plot the calculated minimum energy particle configurations for N=3 and N=20 (the highest
    # converged N). While the BFGS converges for all systems in question, its performance is inconsistent,
    # and the reasonable results for N=6 and N=9 might just be luck, as the results for N=5 and N=7 are terrible
    # Despite small average distances for N=4 and N=20, these solutions are poor, as some inter-atomic distances
    # for these solutions are as small as 1/2r0 - which lowers the average but is absurd, as the potential is highly
    # repulsive at this distance

    # We plot the solutions by using James' functions for making nice plots:
    if 0:
        # For 3 particles:
        ax, points = create_base_plot(minimum_configurations[1])
        ax = make_plot_pretty(ax)
        plt.axis('on')
        plt.title("Minimum energy configuration of 3 Argon atoms", fontsize=15)
        plt.show()

        # For 20 particles:
        ax, points = create_base_plot(minimum_configurations[8])
        ax = make_plot_pretty(ax)
        plt.title("Approximate min. energy configuration of 20 Argon atoms", fontsize=15)
        plt.axis("on")
        plt.show()
        print("We have plotted the BFGS-proposed minimum energy configurations for N=3 and N=20.")
        print()

    ### PROBLEM J:
    # We have added a line-search step to the BFGS function, which is found in the function and activated by
    # setting linesearch = True as parameter. I did not have a lot of success with using the golden_min_search as a
    # linesearch step, probably since it only works for unimodal functions (which is quite a restriction).

    # Instead, I have implemented a linesearch function called wolfe_linesearch (found above the BFGS function),
    # which finds an ideal stepsize alpha such that the two weak Wolfe conditions are satisfied. Note that I have
    # not implemented a linesearch that alters the direction of the change in x, as is otherwise common when performing
    # Wolfe linesearch.

    # Now, we simply repeat the task of calculating minimum energy configurations of the various systems, this time with
    # a line search step:

    collection = [Xstart2, Xstart3, Xstart4, Xstart5, Xstart6, Xstart7, Xstart8, Xstart9, Xopt20]
    index = [2, 3, 4, 5, 6, 7, 8, 9, 20]
    minimum_configurations = []
    total_calls_with_linesearch = 0
    print(
        "Using the BFGS-minimizer with Wolfe conditions linesearch to find the minimum energy configurations, we find that:")
    for k in np.arange(len(index)):
        x, f_calls,converged = BFGS(flat_V, flat_gradV, collection[k], linesearch=True)
        X = x.reshape(-1, 3)  # Reshape to Nx3 matrix
        minimum_configurations.append(X)  # Collect solution
        n, n = np.shape(distance(X))
        total_calls_with_linesearch += f_calls

        # Calculate av. interatomic distances by summing all elements and dividing by n^2-n.
        # We subtract the diagonal elements from the number of distances, as these represent distances from particles to themselves.
        av_distance = np.sum(distance(X)) / (n ** 2 - n)
        # Calculate average distances in units of equilibrium distance r0
        av_distance_unit_r0 = av_distance / r0
        # Calculate number of distances within 1% of equilibrium distance. Divide by two to avoid counting each distance twice.
        if index[k] > 8:
            ideal_distances = 0.5 * np.sum(np.abs((distance(X) - r0) / r0) <= 0.02)
        else:
            ideal_distances = 0.5 * np.sum(np.abs((distance(X) - r0) / r0) <= 0.01)

        print(f"For {index[k]} particles: ",
              f"# of distances equal to equilibrium distance within 1 % = {ideal_distances},  ", f" "
                                                                                                 f"\n Average interatomic distance in units of eq. distance = {av_distance_unit_r0:.4f}, ",
              f" # of functions calls = {f_calls}. converged = {converged}")
    print()
    print("We see that implementing a linesearch step increases the robustness of the BFGS minimizier considerably. \n"
          "Now, high quality solutions are obtained all up to N=9. \nBy comparing these results to the theoretical "
          "number of ideal bonds in a minimum energy configuration, we see that the solutions are exact up to"
          "\nN=8. For N=9, the solution has one less ideal bond than the theoretical maximum. By considering the number of function calls,"
          "\n however, we see that this increased robustness comes at a heavy price if function evaluations are slow.")
    print()
    print("Total # of calls with linesearch = ", total_calls_with_linesearch)

    # Finally, to get a sense of our linesearch solutions, we plot them
    for k in np.arange(len(index)):
        print(f"Emin for {index[k]} = ", V(minimum_configurations[k]))
        if index[k] >8:
            cutoff = 1.02 * r0
        else:
            cutoff = 1.01 * r0
        ax, points = create_base_plot(minimum_configurations[k],r0)
        ax = make_plot_pretty(ax)
        plt.title(f"Minimum energy configuration of {index[k]} Argon atoms", fontsize=15)
        plt.show()
