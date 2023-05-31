import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
from LJhelperfunctions import V,EPSILON,SIGMA,distance,gradV,flat_V,flat_gradV
NA = newaxis

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
def create_base_plot(points):
    # Create the figure and 3D axis
    fig, ax = create_3d_plot()
    # translate points so centre of mass is at origin
    m = np.mean(points, axis=0)
    points = np.array(points) - m[np.newaxis, :]
    # Plot points and lines between points that are near to optimal
    ax.scatter(*points.T)
    ax = plot_min_distances(ax, points)
    return ax, points
def make_plot_pretty(ax):
    # Make the plot pretty
    ax = transparent_axis_background(ax)
    ax = make_axis_equal(ax)
    return ax


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

    #Find dimension of "economy size" R, a rxr invertible matrix
    R_index = np.argwhere(np.abs(np.diag(R)) >1e-12)   # Find indices of non-vanishing diagonal entries

    r = 1 + int (max (R_index))                        # Find biggest index of non-vanishing diagonal entry
    R = R[0:r,0:r]                                     # Construct "economy size" R

    #Approach 1: Construct the permutation matrix
    P_matrix = np.zeros([n,n])                         #Construct permutation matrix[permutes columns]
    for i in np.arange(n):
        P_matrix[int(P[i]),i]=1

    #y = Qt @ b                                          # Solve right hand side to obtain mx1 matrix

    x = backward_substitution(R,b_copy[0:r])           # Solve invertible part of system to obtain rx1 matrix solution
    x =  P_matrix @ np.block([x,np.zeros(n-r)])        # Obtain full nx1 solution by permutation of [x,np.zeros(n-r)]

    # Approach 2: Simply permute elements in x as a final step
    """
    x_new = backward_substitution(R, b_copy[0:r])
    x_new = np.block([x_new,np.zeros(n-r)])
    for i in np.arange(n):
        if i != int(P[i]):
            index = int ( np.argwhere(P == i))
            x_new [[i,index]] = x_new [[index,i]]
            P[[i, index]] = P[[index, i]]

    """
    return   x
#### Functions needed in part a:
def potential_calculator(x,X0):                 #takes scalar x and matrix X0 of fixed particle positions as input
    x = np.array([x,0,0],dtype='float')         #construct vector from x
    X = np.r_['0,2',x,X0]                       #construct matrix with all particle positions
    return V(X)                                 #return value of potential energy
def potential_derivative(x):                    #derivative of the Jones potential energy function
    val = 4*EPSILON*(6*np.power(SIGMA,6)/np.power(x,7) - 12 * np.power(SIGMA,12) / np.power(x,13)  )
    return val
def potential_plotter(X0): #Takes matrix X0 of fixed particle positions,
    # This function takes a matrix X0 of fixed particle positions. It calculates and plots the potential energy of these
    # particles along with the particle having position x0 = (x,0,0), where x is varied from 3 to 11

    N=100                                                                   #Number of evaluations points
    range = np.linspace(3,11,N)
    potential_vec = np.vectorize(lambda x: potential_calculator(x, X0))     #Vectorize potential energy function

    plt.plot(range,potential_vec(range),'b-',label = 'Potential energy')    #Plot values
    plt.ylabel("Potential energy (kJ/mol)",fontsize=16)
    plt.xlabel("First coordinate position (Å)", fontsize=16)
    plt.ylim(-2,4)
    plt.xlim(2,12)
    plt.plot([2,12],[0,0],'k-')
### Functions needed in part b-f:
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
def newton_root(f,df,x0,tolerance=1e-13,max_iterations=1000):
    x,fx = x0,f(x0)                                 #Initialize values
    for k in np.arange(1,1+max_iterations):
        x_new = x - fx / df(x)                      #Calculate new approximation to solution of f(x)=0
        fx = f(x_new)                               #Store new function value
        if np.abs(fx)<tolerance:
            return x, 1+2*(k)  #Return x and # of function calls of f if convergent.
        x = x_new                                   #Update value of x
    return None                                     #Return None if convergence is not met
def bisection_newton_root(f,df,a,b,tolerance=1e-12):
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
def linesearch(F,X0,d,alpha_max,tolerance=1e-12):   #Performs a line search to find root of F
    # F is an Nx3 matrix function. We use linesearch to find the alpha that minimizes
    # np.dot(F,d) along the line X0+t*d for t in 0,alpha_max.
    # that is, we find the alpha that minimizes the component of F pointing in direction of d along this line

    ### WHY? This linesearch can be used to find minima of scalar function taking vector values (as potential energy)
    # to find the minimum of a function V along the line X0+t*d, we can find the t such that its derivative along this
    # line is 0, i.e. such that np.dot(gradV(X0+t*d),d) is zero.


    def F_along_gamma(t, F, X0, d):  # Construct scalar function taking values on the line X0+t*d
        val = np.sum(F(X0 + t * d) * d)  # Multiply element-wise and sum to perform dot product
        return val

    #Call bisection_root function to find the root (t value corresponding to) of F along the line segment,
    # along with the number of calls of F
    zero_value, number_of_F_calls = bisection_root(lambda t: F_along_gamma(t,F, X0,d),0,alpha_max)
    return zero_value, number_of_F_calls
### Functions needed in part g-j:
def golden_section_min(f,a,b,tolerance = 1e-3):
    #find the minimum of a unimodal (strictly increasing or decreasing function) on the interval [a,b])

    #This weird looking definition of tau ensures that the ratio between x1, x2 and the end points are fixed
    #, which allows us to update only one value at each step (instead of both).
    tau = (np.sqrt(5)-1)/2                      # Define tau such that the ratio between x1,x2 and the end points are fixed
    x1 = a + (1-tau)*(b-a)                      #Initialize x1 and x2
    x2 = a + tau*(b-a)
    f1, f2 = f(x1), f(x2)                     # initialize f1 and f2
    calls, max_iterations = 2, 1000           # calls count the number of function calls

    while np.abs(b-a) > tolerance and calls < max_iterations:
        calls += 1
        if f1 > f2:                               # If f1>f2, there is a minimum in [x1,b]
            a = x1                              # Update left endpoint
            x1, f1 = x2, f2                       # Flip indices to preserve the ratio between x1,x2 and the end points
            x2 = a + tau*(b-a)                  # Update x2 and f2
            f2 = f(x2)
        else:                                   # If f1<f2, there is a minimum in [a,x2]
            b = x2                              # Update right endpoint
            x2, f2 = x1, f1                       # Flip indices to preserve the ratio between x1,x2 and the end points
            x1 = a + (1-tau)*(b-a)              # Update x1 and f1
            f1 = f(x1)
    return (b+a)/2, calls                           # Return value that sends f to a minimum and number of function calls
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
            return t,k+2   #Return optimal step size t, number of function calls and new gradient value

        #If Wolfe conditions are not met within max_iterations, return t=1, so f_new = f(x+1*s), and line search
        #failed. Also return number of function calls of f and grad f along with new gradient value
    return 1,k+2
def BFGS(f,gradf,X,tolerance = 1e-6, max_iterations = 1e4,linesearch=True): #takes flat vectors and functions

    x = X
    B = np.eye(np.size(x))                      #Initialize approximation to Hessian matrix
    grad = gradf(x)                             #Intialize and store value of gradient
    calls = 1                                   #Count number of function calls

    for k in np.arange(max_iterations):
        s = qr_pivot_solve(B,-grad)             # Find size and direction of change in x

        if linesearch and np.linalg.norm(grad) > 0.1:  # When close to convergence, linesearch in unnecessary
            # To find the ideal step size, I use the function wolfe_linesearch, (defined above), that calculates a
            # stepsize satisfying the weak Wolfe conditions.
            # It takes current gradient value - grad - as input and gives back new gradient value to minimize function
            # calls. It also returns  the stepsize alpha and number of function calls
            alpha, steps = wolfe_linesearch(f, gradf, x, s, grad)
            calls += steps
        elif not linesearch and k < 3:
            alpha = 1e-3
        else:
            alpha = 0.25
            if not linesearch and k < 1:
                alpha = 1e-3

        x_new = x + alpha * s  # Update x
        grad_new = gradf(x_new)

        y = grad_new - grad                #Update y and B
        B = B + np.outer(y,y) / (y @ s) - (B @ np.outer(s,s) @ B) / (s @ B @ s)

        if np.linalg.norm(grad_new) < tolerance:
            return x, calls    #If converged within tolerance, return x and calls

        grad = np.ndarray.copy(grad_new)   # Update gradient and x
        x = np.ndarray.copy(x_new)
    return None                            # If not converged, return None

def BFGS_improved(f, gradf, X0, tolerance = 1e-6, max_iterations = 10000, n_reset = 100, linesearch = False, inverse = True):
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
    f_calls = 1

    while iterations < max_iterations:
        if np.linalg.norm(grad) < tolerance:
            converged = True
            break
        iterations += 1
       # if iterations % 1000 == 0:
      #      print (iterations)
        if iterations % n_reset == 0:
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


print("PROJECT 3")

### PROBLEM A:
# Use the potential energy function V to plot the strength of the potential between two particles at
# x0 = (x,00) and x1=(0,0,0). Plot the values for x in [3,11]

x1 = np.array([0,0,0],dtype='float')
x2 = np.array([14,0,0],dtype='float')
x3 = np.array([7,3.2,0],dtype='float')

#The function potential_calculator(x) can be found above and calculates the potential energy of an arbitrary number
#of argon atoms, where all but one atom is fixed. The position of the remaning atom is (x,0,0). This function can
#be used to solve both the two body and four body system

#The function potential_plotter can be found above. It takes an input of fixed particle positions, calculates
#and plots the potential energy while variying (x,0,0) from x=3 to x=11
potential_plotter(x1)       #x1 is the fixed particle position
plt.title("Potential energy curve for two Argon atoms",fontsize=16)
plt.show()

#With x2 and x3 defined as above, calculate the potential energy of the 4-body system with positions [x0,x1,x2,x4]
#while varying x in x0=[x,0,0] from 3 to 11

#Construct matrix with columns x1, x2 and x3 of fixed particle positions
X0 = np.block([[x1],[x2],[x3]])

#Use potential_plotter to calculate and plot potential energy
potential_plotter(X0)
plt.title("Potential energy curve for 4 Argon atoms",fontsize=16)
plt.show()


### PROBLEM B:
# The besection_root function can be found above.

#We now calculate the interatomic distance at which the potential energy of the two-body system (x0,x1) is 0 along
#with the number of function calls to acchieve the convergence criterion abs(V)<1e-13:

zero_value,steps = bisection_root(lambda x: potential_calculator(x,x1),2,6)
print("\n")
print("Using the bisection method to calculate the interatomic distance at which the potential energy is zero:")
print("Distance = ",zero_value,",  Number of function calls: ",steps)
print("We see that this distance is equal to the parameter sigma, as expected")
print()


#PROBLEM C:
# The newton_root function can be found above.
#Using the Newton-method to calculate the root of the potential energy to acchieve a convergence criterion abs(V)<1e-12:

zero_value,steps = newton_root(lambda x: potential_calculator(x,x1),potential_derivative,2) #inital guess is 2
print("Using the Newton root method to calculate the interatomic distance at which the potential energy is zero:")
print("Distance = ",zero_value,",  Number of function calls: ",steps)
print()


#PROBLEM D:
# The function bisection_newton_root can be found above. It combines the fast convergence of the newton method
# with the guaranteed success of the bisection method (given a bracket), basically by falling back on the bisection method
# if the newton iteration generates a value that is outside the interval (of a given iteration), and otherwise
# using the values from the Newton method

zero_values,steps = bisection_newton_root(lambda x: potential_calculator(x,x1),potential_derivative,2,6)
print("Using the combined bisection/Newton root method to calculate the interatomic distance at which the potential energy is zero:")
print("Distance = ",zero_value,",  Number of function calls: ",steps)
print()


### PROBLEM E:
#Look at the gradient of the 2-particle system from part a) with x in [x,0,0] and plot along with potential energy
# from x=3 to x=10. Why are exactly 2 components non-zero? Why are they equal and opposite?

# The qualitative questions will be answered more in-depth the report.
# Here, we will simply note that because both particles are situated on the x-axis,
# the force (negative gradient) acts along this axis as well. The force of x1 acting on x0, which is -grad[0,0] is
# equal and opposite to the force of x0 acting on x1, which is -grad[1,0] - it thus suffices just to consider one of these.

# Define a function that takes a scalar x and a static particle configuration X0 and calculates the x-component of the
# (negative) force acting on (x,0,0):
def gradient_calculator(x,X0):
    x = np.array([x,0,0],dtype='float')
    X = np.r_['0,2',x,X0]                       #Construct matrix of all particles
    grad = gradV(X)
    return grad[0,0]                            #Return the negative force acting on particle with position x0=(x,0,0)


#Plot the negative force acting on particle with position x0=(x,0,0) from x=3 to x=10:
N=100
range = np.linspace(3,10,N)
gradient_vec = np.vectorize(lambda x: gradient_calculator(x, x1))       #Vectorize gradient calculator

#In this special case, the gradient is simply the derivative of the potential energy
print("We note that the [0,0]'th coordinate of the gradient (which is just the derivative of the potential energy in "
      " this case), is 0 when the potential energy has a minimum.")

#plot gradient values
plt.plot(range,gradient_vec(range),'m--',label='Derivative of potential energy')
#plot potential energy values
potential_plotter(x1)
#plot vertical lines indicating potential energy minimum (minimum at r0=2**(1/6)*sigma)
plt.plot(r0*np.ones(5),np.linspace(-3,5,5),'k--',label='Potential energy minimum')
plt.ylabel('')
plt.title("Potential energy curve and its derivative",fontsize='16')
plt.legend()
plt.show()

#Next, we are going to calculate the gradient for the 4-particle system from problem a at which the potential
# energy curve has a minimum. Using the golden_section_minimum function (introduced in problem g), we calculate the
# right side minimum as
X0 = np.block([[x1],[x2],[x3]])
minimum,steps = golden_section_min(lambda x: potential_calculator(x,X0),8,11)

#Construct matrix of particle positions
x = np.array([minimum,0,0],dtype='float')
X = np.r_['0,2',x,X0]

#Calculate gradient
grad = gradV(X)
print()
print(f"Gradient of the 4-particle system from part a at the minimum {minimum:.4}:")
print(grad)
print("In this case, the gradient is not a minimum - a force is acting on all particles. This is due to the fact that"
      " \n the potential energy curve has been calculated while keeping 3 particles fixed. In general, \n in one wants to"
      " calculate an equilibrium configuration of N particles (such that the gradient is 0), \n one must vary the positions"
      " of all but 1 particle until equilibrium is reached.")
print()


###PROBLEM F:
# The function linesearch can be found above. In the following, we are going to test it by finding the minimum
# of the potential energy along a curve defined by the particles, whose positions have been collected in X0 below:

x0 = np.array([4,0,0])
X0 = np.block([[x0],[X0]])

# We will find a minimum of the potential energy along the line X0 - alpha * gradV(X0) by using the linesearch function
# to find the value of alpha where the gradV(X0-alpha*gradV(X0)) is completely orthogonal to the line.

#use linesearch to find alpha and the number of function calls of the gradient. 1 is the max value of alpha
alpha,steps = linesearch(gradV,X0,-gradV(X0),1)    #if gradV 0 then -gradV is 0

# point where potential energy is minimized
minimum_point = X0 - alpha * gradV(X0)

print("Value of alpha that minimizes potential energy along line segment = ",alpha, ",    # of function calls = ",steps)
print("Value of potential energy when minimized along line segment =", V(minimum_point))
print(f"Value of dot product between gradient at minimum and direction of line segment = {np.sum(  gradV(minimum_point)*gradV(X0)):.3e}")
print("We see that this dot product vanishes, i.e. that the gradient at this point is completely orthogonal to the line"
      " segment, as it must be")
print()


### PROBLEM G:
# The golden_section minimizer can be found above along the other functions.
# First, we are going to use it to find the same alpha that we did in f). In f), we used a linesearch function to find
# a root of the gradient, i.e. a critical point of the potential (in this case a minimum).
# With the golden_section minimizer, we can find the alpha that minimizes the potential along this line directly.

alpha,f_calls = golden_section_min(lambda t: V(X0-t*gradV(X0)),0,1)
print(f"Value of alpha found by golden section minimization = {alpha:.4f} ",  "# of function calls = ",f_calls)
print("This value is in agreement with the one found above up to 4 sig. digits. This can be improved by making the tolerance of the minimizer smaller.")

#Finally, we will use the minimizer to find the equilibrium distance between two argon atoms
r0,f_calls = golden_section_min(lambda x: potential_calculator(x,x1),3,4)
print(f"Equilibrium distance between two Argon atoms = {r0:.3f}" ,"  # of function calls = ",f_calls)
print()


### PROBLEM H:
# The BFGS algorithm can be found above. Note that I have "dampened" the first step by letting x_new = x - 0.25*grad(f(x))
# in order to avoid / lessen the risk of overshooting and ultimately divergence. The algorithm has been constructed by
# approximating the Hessian. In problem J, we will turn on a linefunction step that significantly increases the useful-
# ness of the routine.

# Now, we will test it to find the equilibrum distance between two Argon atoms by using the matrix Xstart2, a matrix
# containing two random particle positions.

x,f_calls= BFGS(flat_V,flat_gradV,Xstart2,linesearch=False)
X=x.reshape(-1,3)

# We use the distance function to calculate the equilibrium distance between the two particles. The resulting 2x2
# distance matrix is symmetric, and we need only consider the element 1,0 or 0,1 to extract this distance:
D = distance(X)
print(f"Equilibrium distance of 2 Argon atoms found with BFGS = {D[0,1]:.6f},","  # of functions calls = ",f_calls)
print("This result is in agreement with our result from g), but with a greater precision and fewer functions calls.")
print()


### PROBLEM I:
# In this section, we are going to use the BFGS minimizer without the linesearch step to calculate the minimum
# energy configurations for 2,3,...9 and 20 particles.

#Collect and index intial particle configurations to initialize for-loop
collection = [Xstart2,Xstart3,Xstart4,Xstart5,Xstart6,Xstart7,Xstart8,Xstart9,Xopt20]
index = [2,3,4,5,6,7,8,9,20]
#Collect solutions
minimum_configurations =[]
calls_total_without = 0
print("Using the BFGS-minimizer without linesearch to find the minimum energy configurations, we find that:")
for k in np.arange(len(index)):
    x,f_calls= BFGS(flat_V,flat_gradV,collection[k],linesearch=False)
    X=x.reshape(-1,3)                                           #Reshape solution to Nx3 matrix
    minimum_configurations.append(X)                            #Collect solution
    calls_total_without += f_calls
    n, n = np.shape(distance(X))
    # Calculate av. interatomic distances by summing all elements and dividing by n^2-n [we subtract vanishing diagonal elements]
    av_distance = np.sum(distance(X))/(n**2-n)
    av_distance_unit_r0 = av_distance / r0      #Calculate average distance in units of equilibrium distance r0
    #Calculate number of distances within 1% of equilibrium distance. Divide by two to avoid counting each distance twice.
    ideal_distances = 0.5 * np.sum(np.abs((distance(X)-r0)/r0)<=0.01)

    print(f"For {index[k]} particles: ", f"# of distances equal to equilibrium distance within 1 % = {ideal_distances},  ", f" "
    f"\n Average interatomic distance in units of eq. distance = {av_distance_unit_r0:.4f}, " , f" # of functions calls = {f_calls}")
print()
print("We see that while the BFGS-minimizer does converge for all systems in question, it performance is inconsistent"
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
#For 3 particles:
ax,points = create_base_plot(minimum_configurations[1])
ax = make_plot_pretty(ax)
plt.axis('on')
plt.title("Minimum energy configuration of 3 Argon atoms", fontsize = 15)
plt.show()

# For 20 particles:
ax,points = create_base_plot(minimum_configurations[8])
ax = make_plot_pretty(ax)
plt.title("Approximate min. energy configuration of 20 Argon atoms", fontsize = 15)
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
Xstart20 = data['Xstart20']
collection = [Xstart2,Xstart3,Xstart4,Xstart5,Xstart6,Xstart7,Xstart8,Xstart9,Xopt20]
index = [2,3,4,5,6,7,8,9,20]
minimum_configurations =[]
total_calls_with_linesearch = 0
print("Using the BFGS-minimizer with Wolfe conditions linesearch to find the minimum energy configurations, we find that:")
for k in np.arange(len(index)):
    x,f_calls= BFGS_improved(flat_V,flat_gradV,collection[k],n_reset=75,inverse=True,linesearch=True)[0:2]
    X=x.reshape(-1,3)                               #Reshape to Nx3 matrix
    minimum_configurations.append(X)                #Collect solution
    n, n = np.shape(distance(X))
    total_calls_with_linesearch += f_calls

    # Calculate av. interatomic distances by summing all elements and dividing by n^2-n.
    # We subtract the diagonal elements from the number of distances, as these represent distances from particles to themselves.
    av_distance = np.sum(distance(X)) / (n ** 2 - n)
    # Calculate average distances in units of equilibrium distance r0
    av_distance_unit_r0 = av_distance / r0
    # Calculate number of distances within 1% of equilibrium distance. Divide by two to avoid counting each distance twice.
    ideal_distances = 0.5 * np.sum(np.abs((distance(X)-r0)/r0)<=0.01)

    print(f"For {index[k]} particles: ", f"# of distances equal to equilibrium distance within 1 % = {ideal_distances},  ", f" "
    f"\n Average interatomic distance in units of eq. distance = {av_distance_unit_r0:.4f}, " , f" # of functions calls = {f_calls}")
print()
print("We see that implementing a linesearch step increases the robustness of the BFGS minimizier considerably. \n"
      "Now, high quality solutions are obtained all up to N=9. \nBy comparing these results to the theoretical "
      "number of ideal bonds in a minimum energy configuration, we see that the solutions are exact up to"
      "\nN=8. For N=9, the solution has one less ideal bond than the theoretical maximum. By considering the number of function calls,"
      "\n however, we see that this increased robustness comes at a heavy price if function evaluations are slow.")
print()
print("Total # of calls with linesearch = ", total_calls_with_linesearch)

#Finally, to get a sense of our linesearch solutions, we plot them
for k in np.arange(len(index)):
    print(f"Emin for {index[k]} = ", V(minimum_configurations[k]))
    ax,points = create_base_plot(minimum_configurations[k])
    ax = make_plot_pretty(ax)
    plt.title(f"Minimum energy configuration of {index[k]} Argon atoms", fontsize = 15)
    plt.show()

print(V(minimum_configurations[7]))