import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
from LJhelperfunctions import V,EPSILON,SIGMA,distance,gradV,flat_V,flat_gradV
NA = newaxis

##Helt styr på NA. Apply og forstå funktioner
##Styr på egne funktioner ad 2 omgange
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
    # matrix chosen so that at each step, the column of the remaining matrix R[k::,k::] is switched with the
    # k'th column. This collects all j linearly independent columns to the left, resultning in the factorization
    # A = Q [(R, S), 0 , 0], where the reduced jxj matrix R is upper triangular and invertible, and the
    # jx(n-j) matrix S contains the singular part, i.e. spans the kernel of A. The solutions now take the form
    # Ax = [c_image, c_kernel, c_overdetermined]. The reduced system
    # Q @ R[0:j,0:j] @ P^T @ x[0:j] = c_image can now be solved for x[0:j] - the invertible part of the system.
    # with b an mx1 vector, we can solve for P^T @ x[0:j] = Q^T @ b [0:j] --> x = P @ (x[0,j],zeros(m.j),
    # which is the full solution. [0's as solutions to the last m-n equations is simply because can maximally span
    # n dimensions. Setting the n-j components mapping to the kernel to 0 is a particular solution, but we only
    # care about the invertible part of the system.

    # if A has full column rank, then upper part of R, namely R[0:n,:] is invertible.
    # Since orthogonal transformations preserve norms, they preserve the (norm defined) solution to the least squares
    # problem Ax ~ b, meaning that QRx = [c1, c2], where Rx = Q^t c1 = b, which can be solved by back substitution.
    ###

    ##OBS: It is unclear why the column in R[k::,k::] with the biggest 2-norm ensures that the corresponding full
    # column is linearly independent from the columns in R[:,0:k] if k<rank(A). TBD

    m,n = np.shape(A)                                       # A an m x n matrix

    Qt = np.identity(m)                                     #Q_transpose
    R = np.ndarray.copy(np.array(A,dtype='float'))
    P = np.linspace(0,n-1,n)                                #List keeping track of permutations

    for k in np.arange(n-1):
                                                            #Perform pivoting:
        max_index,max_norm = max_norm_index(R,k)            #Use max_norm_index function (defined above) to extract
                                                            #index corresponding to column of R[k::,k::] with biggest 2-norm
        R[:,[k,max_index]] = R[:,[max_index,k]]             # Pivot columns
        P [[k,max_index]] = P[[max_index,k]]

        a = R[k:m,k]                                        # Carry out QR-calculations as usual
        alpha = - np.copysign(1,a[0]) * np.linalg.norm(a)

        v = np.ndarray.copy(a)
        v[0] = v[0]-alpha                                   #construct Householder vector
        v = v / np.linalg.norm(v)                           #normalize

        R[k:m,k:n] = apply_reflection(R[k:m,k:n],v)     #Calculate values on entire sub-matrix to avoid for-loop
        Qt[k:m,:] = apply_reflection(Qt[k:m,:],v)       #On Qt, we apply the reflection on all columns, as
                                                        #columns Qt[k:m,i] do not necesarily vanish for i<k

    P_matrix = np.zeros([n,n])                         #Construct permutation matrix[permutes columns]
    for i in np.arange(n):
        P_matrix[int(P[i]),i]=1

    #Find dimension of "economy size" R, a rxr invertible matrix
    R_index = np.argwhere(np.abs(np.diag(R)) >1e-12)   # Find indices of non-vanishing diagonal entries

    r = 1 + int (max (R_index))                        # Find biggest index of non-vanishing diagonal entry
    R = R[0:r,0:r]                                     # Construct "economy size" R

    y = Qt @ b                                          # Solve right hand side to obtain mx1 matrix

    x = backward_substitution(R,y[0:r])                # Solve invertible part of system to obtain rx1 matrix solution
    x =  P_matrix @ np.block([x,np.zeros(n-r)])        # Obtain full nx1 solution by permutation of [x,np.zeros(n-r)]
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
            k, fm = k + 2, f(m)         # Store fm and update k for calling f(m) and df(m) or df(x)
            if np.abs(fm) < tolerance:
                return m,k     #Return if convergence criterion met

            #If abs(fm)<abs(fx), then m is closer to the solution (at least if there is only one root in [a,b]),
            #and we therefore use m to calculate new value of x via Newton's root method:
            if np.abs(fx) > np.abs(fm):               #If fm is closest to 0, calculate x_new as newton root from m
                x_new = m - fm/df(m)

            #If x, on the other hand, is closest to the solution, we use this value as in Newton's root method
            else:
                x_new = x - fx / df(x)                #If fx is closest to 0, calculate x_new as newton root from x

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

def linesearch_function(F,X0,d,alpha_max,tolerance=1e-12):   #Performs a line search to find root of F
    # F is an NxN matrix function. We use linesearch to find the alpha that minimizes
    # np.dot(F,d) along the line X0+t*d for t in 0,alpha_max.
    # that is, we find the alpha that minimizes the component of F pointing in direction of d along this line

    ### WHY? This linesearch can be used to find minima of scalar function taking vector values (as potential energy)
    # to find the minimum of a function V along the line X0+t*d, we can find the t such that its derivative along this
    # line is 0, i.e. such that np.dot(gradV(X0+t*d),d) is zero.


    def F_along_gamma(t, F, X0, d):  # Construct scalar function taking values on the line X0+t*d
        val = np.sum(F(X0 + t * d) * d)  # Multiply element-wise and sum to perform dot product
        return val

    #Call bisection_root function to find the root (t value corresponding t) of F along the line segment,
    # along with the number of calls of F
    zero_value, number_of_F_calls = bisection_root(lambda t: F_along_gamma(t,F, X0,d),0,alpha_max,tolerance)
    return zero_value, number_of_F_calls

### Functions needed in part g-j:
def golden_section_min(f,a,b,tolerance = 1e-3):
    #find the minimum of a unimodal (strictly increasing or decreasing function) on the interval [a,b])

    #This weird looking definition of tau ensures that the ration between x1, x2 and the end points are fixed
    #, which allows us to update only one value at each step (instead of both).
    tau = (np.sqrt(5)-1)/2                      # Define tau such that the ratio between x1,x2 and the end points are fixed
    x1 = a + (1-tau)*(b-a)                      #Initialize x1 and x2
    x2 = a + tau*(b-a)
    f1, f2 = f(x1), f(x2)                     # initialize f1 and f2
    calls, max_iterations = 2, 100          # calls count the number of function calls

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

    if np.linalg.norm(gradfx) > 1:
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
def BFGS(f,gradf,X,tolerance = 5e-3, max_iterations = 500,linesearch=True): #takes flat vectors and functions

    x = X
    B = np.eye(np.size(x))                      #Initialize approximation to Hessian matrix
    grad = gradf(x)                             #Intialize and store value of gradient
    calls = 1                                   #Count number of function calls

    for k in np.arange(max_iterations):
        s = qr_pivot_solve(B, -grad)             # Find size and direction of change in x

        if linesearch and np.linalg.norm(grad) > 1.85: # When close to convergence, linesearch in unnecessary
            # To find the ideal step size, I use the function wolfe_linesearch, (defined above), that calculates a
            # stepsize satisfying the weak Wolfe conditions.
            # It takes current gradient value - grad - as input and gives back new gradient value to minimize function
            # calls. It also returns  the stepsize alpha and number of function calls

            wolfe, golden = False, True
            if wolfe:
                alpha, steps, grad_new = wolfe_linesearch(f, gradf, x, s, grad)
                x_new = x + alpha * s  # Update x
                calls += steps
            ###OBS:: With tolerance=5e-3 in BFGS, tol = 0.3 i golden_section, goldene_section_max_it = 100
            # and linesearch only when norm(grad)>2, we
            ### get a proper performance up to and including 5 atoms - with perfect results up to 4.
            # curiously, the results for N=20 are surprisingly good also
            if golden:
                alpha, steps = golden_section_min(lambda t: f(x+t*s),0,1,0.3)
                x_new = x + alpha * s  # Update x
                grad_new = gradf(x_new)
                calls += steps + 1
        else:
            alpha = 1
            if not linesearch and k < 1:
                x_new = x - 0.25 * grad          # If k=0, "dampen" change in x to avoid overshooting
                grad_new = gradf( x_new)        # Store new gradient value
                calls += 1
            else:
                x_new = x + alpha * s          # Update x
                grad_new = gradf( x_new)        # Store new gradient value
                calls += 1

        y = grad_new - grad                #Update y and B
        B = B + np.outer(y, y) / (y @ s) - (B @ np.outer(s, s) @ B) / (s @ B @ s)

        if np.linalg.norm(grad_new) < tolerance:
            return x, calls    #If converged within tolerance, return x and calls

        grad = np.ndarray.copy(grad_new)   # Update gradient and x
        x = np.ndarray.copy(x_new)
    return None                            # If not converged, return None


x1 = np.array([0, 0, 0], dtype = 'float')
x2 = np.array([14, 0, 0], dtype = 'float')
x3 = np.array([7, 3.2, 0], dtype = 'float')
X0 = np.block([[x1], [x2], [x3]])
x0 = np.array([4, 0, 0])
X0 = np.block([[x0], [X0]])

### PROBLEM I:
# In this section, we are going to use the BFGS minimizer without the linesearch step to calculate the minimum
# energy configurations for 2,3,...9 and 20 particles.

#Collect and index intial particle configurations to initialize for-loop
collection = [Xstart2,Xstart3,Xstart4,Xstart5,Xstart6,Xstart7,Xstart8,Xstart9,Xopt20]
index = [2,3,4,5,6,7,8,9,20]
#Collect solutions
minimum_configurations =[]
lineout=False
if lineout:
    print("Using the BFGS-minimizer without linesearch to find the minimum energy configurations, we find that:")
    calls_sum = 0
    for k in np.arange(len(index)):
        x,f_calls= BFGS(flat_V,flat_gradV,collection[k],linesearch=False)
        X=x.reshape(-1,3)                                           #Reshape solution to Nx3 matrix
        minimum_configurations.append(X)                            #Collect solution

        n, n = np.shape(distance(X))
        # Calculate av. interatomic distances by summing all elements and dividing by n^2-n [we subtract vanishing diagonal elements]
        av_distance = np.sum(distance(X))/(n**2-n)
        av_distance_unit_r0 = av_distance / r0      #Calculate average distance in units of equilibrium distance r0
        #Calculate number of distances within 1% of equilibrium distance. Divide by two to avoid counting each distance twice.
        ideal_distances = 0.5 * np.sum(np.abs((distance(X)-r0)/r0)<=0.01)

        # fill diagonal to avoid false positives
        Dist_modified = np.ndarray.copy(distance(X) / r0)
        np.fill_diagonal(Dist_modified, 1)
        distances_less_than = np.sum(Dist_modified < 0.99)



        print(f"For {index[k]} particles: ", f"# of distances equal to equilibrium distance within 1 % = {ideal_distances},  ", f" "
        f"\n Average interatomic distance in units of eq. distance = {av_distance_unit_r0:.4f}, " , f" # of functions calls = {f_calls}")
        calls_sum += f_calls
        print(" # of distances less than 0.99r0 = ", distances_less_than)
    print("\n Calls without line = ", calls_sum)


    print()
    print("We see that while the BFGS-minimizer does converge for all systems in question, it performance is inconsistent"
          "for N > 3.\n Note that the reasonable average distance for N=4 is obtained because two particles are closer"
          " than the equilibrium distance,\n which is physically unacceptable for a minimum energy configuration.")
    print()

# Now we will plot the calculated minimum energy particle configurations for N=3 and N=20 (the highest
# converged N). While the BFGS converges for all systems in question, its performance is inconsistent,
# and the reasonable results for N=6 and N=9 might just be luck, as the results for N=5 and N=7 are terrible
# Despite small average distances for N=4 and N=20, these solutions are poor, as some inter-atomic distances
# for these solutions are as small as 1/2r0 - which lowers the average but is absurd, as the potential is highly
# repulsive at this distance

# We plot the solutions by using James' functions for making nice plots:
#For 3 particles:

"""
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
"""

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

collection = [Xstart2,Xstart3,Xstart4,Xstart5,Xstart6,Xstart7,Xstart8,Xstart9,Xopt20]
index = [2,3,4,5,6,7,8,9,20]
minimum_configurations =[]
calls = 0
print("Using the BFGS-minimizer with Wolfe conditions linesearch to find the minimum energy configurations, we find that:")
for k in np.arange(len(index)):
    x,f_calls= BFGS(flat_V,flat_gradV,collection[k],linesearch=True)
    X=x.reshape(-1,3)                               #Reshape to Nx3 matrix
    minimum_configurations.append(X)                #Collect solution
    n, n = np.shape(distance(X))
    # Calculate av. interatomic distances by summing all elements and dividing by n^2-n.
    # We subtract the diagonal elements from the number of distances, as these represent distances from particles to themselves.
    av_distance = np.sum(distance(X)) / (n ** 2 - n)
    # Calculate average distances in units of equilibrium distance r0
    av_distance_unit_r0 = av_distance / r0
    # Calculate number of distances within 1% of equilibrium distance. Divide by two to avoid counting each distance twice.
    ideal_distances = 0.5 * np.sum(np.abs((distance(X)-r0)/r0)<=0.01)

    # fill diagonal to avoid false positives
    Dist_modified = np.ndarray.copy(distance(X) / r0)
    np.fill_diagonal(Dist_modified,1)
    distances_less_than = np.sum(Dist_modified < 0.95)

    print(f"For {index[k]} particles: ", f"# of distances equal to equilibrium distance within 1 % = {ideal_distances},  ", f" "
    f"\n Average interatomic distance in units of eq. distance = {av_distance_unit_r0:.4f}, " , f" # of functions calls = {f_calls}")
    print(" # of distances less than 0.95r0 = ", distances_less_than)

    calls += f_calls
print("\n #calls with line = ", calls)
print()
print("We see that implementing a linesearch step increases the robustness of the BFGS minimizier considerably. \n"
      "Now, high quality solutions are obtained all up to N=9. \nBy comparing these results to the theoretical "
      "number of ideal bonds in a minimum energy configuration, we see that the solutions are exact up to"
      "\nN=8. For N=9, the solution has one less ideal bond than the theoretical maximum. By considering the number of function calls,"
      "\n however, we see that this increased robustness comes at a heavy price if function evaluations are slow.")
print()


#Finally, to get a sense of our linesearch solutions, we plot them
for k in np.arange(len(index)):
    ax,points = create_base_plot(minimum_configurations[k])
    ax = make_plot_pretty(ax)
    plt.title(f"Minimum energy configuration of {index[k]} Argon atoms", fontsize = 15)
    plt.show()

