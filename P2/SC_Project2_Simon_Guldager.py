import numpy as np
from chladni_show import show_nodes,show_waves,show_all_wavefunction_nodes,basis_set

np.set_printoptions(precision = 14, suppress = True)        #set values <e-14 to 0 when printing to avoid clutter
                                                            # in matrices
#TEST MATRICES:

# A1-A3 should work with any implementation
A1   = np.array([[1,3],[3,1]],dtype='float')
eigvals1 = [4,-2]

A2   = np.array([[3,1],[1,3]],dtype='float')
eigvals2 = [4,2]

A3   = np.array([[1,2,3],[4,3.141592653589793,6],[7,8,2.718281828459045]],dtype='float')
eigvals3 = [12.298958390970709, -4.4805737703355,  -0.9585101385863923]

# A4-A5 require the method to be robust for singular matrices
A4   = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype='float')
eigvals4 = [16.1168439698070429897759172023, -1.11684396980704298977591720233, 0]

A5   = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
eigvals5 = [68.6420807370024007587203237318, -3.64208073700240075872032373182, 0, 0, 0]

# A6 has eigenvalue with multiplicity and is singular
A6  = np.array(
    [[1.962138439537238,0.03219117137713706,0.083862817159563,-0.155700691654753,0.0707033370776169],
       [0.03219117137713706, 0.8407278248542023, 0.689810816078236, 0.23401692081963357, -0.6655765501236198],
       [0.0838628171595628, 0.689810816078236,   1.3024568091833602, 0.2765334214968566, 0.25051808693319155],
       [-0.1557006916547532, 0.23401692081963357, 0.2765334214968566, 1.3505754332321778, 0.3451234157557794],
       [0.07070333707761689, -0.6655765501236198, 0.25051808693319155, 0.3451234157557794, 1.5441014931930226]])
eigvals6 = [2,2,2,1,0]

K = np.load("Chladni-Kmat.npy")     #Matrix of interest


def max_norm_index(A,k):
    ##Find the index and value of the column in submatrix A[k::, k::] having the greatest Euclidean norm
    m, n = np.shape(A)
    norm_list = np.empty(n-k)
    for i in range(n-k):
        norm_list[i] = np.linalg.norm(A[k::, k+i])
    index = np.argmax(norm_list)
    #return the index of the column in A[k::,k::] having the greatest Euclidean norm along with that norm.
    return index + k, norm_list[index]

def norm_inf (A):
    n = int(np.size(A[0,:]))
    max = sum(np.abs(A[0, :]))           # Set absolute sum of 1st row = infinity norm of A
    for k in range(1, n):
        if sum(np.abs(A[k, :])) > max:   # Search for rows with a bigger absolute sum
            max = sum(np.abs(A[k, :]))   # Update value of infinity norm if a bigger absolute sum is found
    return max

def random_vector(n):               #generate an nx1 matrix, whose entry values are randomly chosen in the range [-1,1]
    x0 = np.random.rand(n)          #nx1 matrix with elements taking random values in [0,1]
    x0 = -1 + 2 * x0                 #Rescale to obtain an mx1 matrix with elements taking random values in [-1,1]
    x0 = x0 / np.linalg.norm(x0,np.inf)
    return x0

def order_lists(eigvectors,eigvalues,rayleigh_resid,iterations,dim):    # order lists from smallest to biggest eigenvalue
    for i in range(dim):
        min_index = np.argmin(eigvalues[i::])  # Find index of smallest eigenvalue in the remaining list

        if i != int(min_index + i):  # Swap values to obtain ordered lists
            eigvalues[[i, int(min_index + i)]] = eigvalues[[int(min_index + i), i]]
            eigvectors[[i, min_index + i], :] = eigvectors[[min_index + i, i], :]
            iterations[[i, min_index + i]] = iterations[[min_index + i, i]]
            rayleigh_resid[[i, min_index + i]] = rayleigh_resid[[min_index + i, i]]

    return eigvectors,eigvalues,rayleigh_resid,iterations

def backward_substitution(A,b):
    n,n = np.shape(A)
    y = np.zeros(n)                                          # Initial values of y set to 0
    for k in range(n-1,-1,-1):
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

    ## NB: Regarding column piovting: We are solving PA = QR st A = QT P.T. Therefore, the transformation in this
    # algorithm sendt QR P.T --> R, i.e. it has the effect of applying Q.T on the left and P on the right.
    # Therefore, when initializing the permutation matrix as the indentity and performing only the column pivoting
    # part on it, we end up with P
    # We thus solve QR(P.Tx) = b --> P.Tx = back_sub (R,Q.T b) --> x = P Q.T b

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
 #   Qt = np.identity(m)                                     #Q_transpose
    R = np.ndarray.copy(np.array(A,dtype='float'))
    P = np.linspace(0,n-1,n)                                #List keeping track of permutations

    for k in range(n-1):
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
      #  Qt[k:m,:] = apply_reflection(Qt[k:m,:],v)       #On Qt, we apply the reflection on all columns, as
        b_copy[k:m] = b_copy[k:m] - 2 * np.dot(v,b_copy[k:m]) * v
                                                        #columns Qt[k:m,i] do not necesarily vanish for i<k


    #Find dimension of "economy size" R, a rxr invertible matrix
    R_index = np.argwhere(np.abs(np.diag(R)) >1e-12)   # Find indices of non-vanishing diagonal entries

    r = 1 + int (max (R_index))                        # Find biggest index of non-vanishing diagonal entry
    R = R[0:r,0:r]                                     # Construct "economy size" R


       #Approach 1: build permutation matrix
    P_matrix = np.zeros([n, n])  # Construct permutation matrix[permutes columns]
    for i in range(n):
        P_matrix[int(P[i]), i] = 1

    # Const = Qt.T @ R @ P_matrix.T
    # print("A = QRP", norm_inf(A-Const))
    x = backward_substitution(R,b_copy[0:r])           # Solve invertible part of system to obtain rx1 matrix solution
    x =  P_matrix @ np.block([x,np.zeros(n-r)])        # Obtain full nx1 solution by permutation of [x,np.zeros(n-r)]

    #Approach 2: Simply permute elements in x as a final step
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

def gershgorin(A):                                     # Calculate Gershgorin centers and radii of matrix A
    n,n = np.shape(A)                                  # A an nxn matrix
    l = []
    for k in range(n):
        row_sum = np.sum(abs(A[k,:]))                  #Calculate row sum of absolute values
        column_sum = np.sum(abs(A[:,k]))               #Calculate column sum of absolute values
        rad = min(row_sum,column_sum)                  #Find the radius as minimum
        l.append((A[k,k],rad))                         #Return centers and radii in the format (center,radius)
    return l

def rayleigh_qt(A,x):                         #Calculate Rayleigh coefficient, ie. approx. eig.value of A@x = eig_val*x
    #The Rayleigh quotient, given an approximate eigenvector, gives a better approximation to the eigenvalue that
    # that yielded by inverse power iteration. Conversely, inverse power iteration converges quickly if an app.
    # eigenvector is used as shift. Therefore, the two methods can be combined to yield rayleigh quotient iteration
    # aka shifted inverse power iteration

    eig_val = ( x @ A @ x) / (x @ x )
    return eig_val

def power_iterate(A,x0):
    """
    Find the largest eigenvalue and corresponding eigenvector of a nxn matrix A

    """

    x = np.array(x0,dtype='float')            #Initial eigenvector guess
    y = 2 * x                              #Arbitrary value to ensure that convergence criterion is not met intially
    sensitivity = 1e-10                      # sensitivity of convergence criterion
    k = 0; max_iterations = 1000

    while np.linalg.norm(y-x,np.inf) > sensitivity and k < max_iterations:
        y = np.ndarray.copy(x)                # Make a copy of x = A @x before updating value of x
        x = x / np.linalg.norm(x,np.inf)      # Normalize
        x = A @ x
        k += 1

    # At this point, both y and x will have 0's everwhere except at the k'th entry, whose absolute value is the eigenvalue
    # If sign(eigenvalue)<0, the sign of the k'th entry of x and y will by opposite. To find the sign of the eigenvalue,
    # we therefore extract the sign of the biggest entry of x:
    index_max_absolute_value = np.argmax(np.abs(x))
    sign_x = np.sign(x[index_max_absolute_value])

    # The relative sign is then found by comparing the sign of the k'th entry of x with that of the k'th entry of y
    relative_sign = sign_x * np.sign(y[index_max_absolute_value])
    return x/np.linalg.norm(x,np.inf), relative_sign * np.linalg.norm(y,np.inf), k

def inverse_iteration (A,x0):
    """
    Similarly to power iteration, the numerically smallest (non-zero) eigenvalue of a matrix can be calculated by applying
    inv(A) iteratively to a random initial vector. At each step, we thus need to update x as
    x_new = inv(A) @ x. To avoid calculating inv(A), we can instead solve the equiv. system
    A @ x_new = x using LU-decomp or QR.
    NB: Since the QR-solver is robust for singular matrices, this method works even when A is singular, as the
    least squares solution is simply calculated instead
    NB: It cannot find the eigenvalue 0 of a matrix, as it would corresponding to the eigenvalue inf for inv(A)
    """

    #Initial guess
    x = np.array(x0,dtype='float')
    #Initiliaze x_new to avoid meeting convergence criterion initially
    x_new = 2 * x

    iterations = 0
    max_iterations = 1000
    tolerance = 1e-10

    while (np.linalg.norm(qr_pivot_solve(A, x)-x_new,np.inf) > tolerance and iterations < max_iterations):

        x_new = np.copy(x)
        x = x / np.linalg.norm(x, np.inf)
        x = qr_pivot_solve(A, x)

        iterations += 1

   # print("x = ", x, "\n \n")
   # print("x = ", qr_pivot_solve(A, x), "\n \n")
   # print("x_new = " , x_new, "\n \n")

    # At this point, both y and x will have 0's everwhere except at the k'th entry, whose absolute value is the eigenvalue
    # If sign(eigenvalue)<0, the sign of the k'th entry of x and y will by opposite. To find the sign of the eigenvalue,
    # we therefore extract the sign of the biggest entry of x:
    index_max_absolute_value = np.argmax(np.abs(x))
    sign_x = np.sign(x[index_max_absolute_value])

    # The relative sign is then found by comparing the sign of the k'th entry of x with that of the k'th entry of y
    relative_sign = sign_x * np.sign(x_new[index_max_absolute_value])
    return x / np.linalg.norm(x, np.inf), 1/(relative_sign * np.linalg.norm(x_new, np.inf)), iterations

def rayleigh_iterate(A,x0, shift0):             #Obtain eigenvalue and eigenvector of A from random vector x0 and
                                                # a guess of the eigenvalue
    """
     Combining inverse power iteration, which yields the smallest eigenvalue and corresponding eigenvector
     of an invertible nxn matrix A [THIS METHOD WORKS FOR SINGULAR MATRICES, AS THE QR-SOLVER SIMPLY SOLVE BY LEAST
    SQUARE IF THE MATRIX IS SINGULAR], and extends it to any eigenvalue by shifting it with the rayleigh coefficient
     of the given eigenvector, which gives a good approximation to the eig.val. This eig.val is then used to solve
     (A-I*eig.val)y = x for y, which in turn can be used to calc. a new eig.val etc.
    """

    n, n = np.shape(A)                           #A an nxn matrix
    x = np.array(x0, dtype='float')             #Initial eigenvector guess

    # Initialize shift
    shift = shift0

    sensitivity = 1e-8  # Sensitivity of convergence criterion
    max_iterations = 100
    k = 0

    for i in range(3):                          #Perform inverse iteration 3 times to obtain approximate eig.vector
        # Here one could obtain the QRP factorization to cheapen the calculation of step 2 and 3
        x = qr_pivot_solve(A-np.eye(n)*shift,x)         # Find solution to shifted system
        x = x / np.linalg.norm(x,np.inf)                    # Normalize with inf-norm
        k += 1

    #continue until the norm of (A-I*eig_val)x less than sensitivity
    while np.linalg.norm( (A - shift*np.eye(n)) @ x ) > sensitivity and k < max_iterations:

        x = qr_pivot_solve(A-np.eye(n)*shift,x)             # Find solution to shifted system
        x = x / np.linalg.norm(x,np.inf)                    # Normalize with inf_norm

        shift = rayleigh_qt(A,x)                            # Update approx. eig.value
        k += 1
    return x/np.linalg.norm(x), shift, k                    #Return norm-2 normalized eig.vec, eigvalue and # of iterations

def find_eigenvectors_and_eigenfunctions_non_defective_slow(A, error_tolerance_eigval = 1e-6):
    matrix = np.ndarray.copy(np.array(A,dtype='float'))
    dim, dim = np.shape(matrix)
    eigv_list = []

    i = 0
    max_iterations = 5 * dim

    while np.size(eigv_list) < dim and i < max_iterations:  # continue until all eigenvalues are found
        i += 1
        if i <= dim:  # Search for eigenvalues using the Gershgorin centers,
            # i.e. the diagonal entries, as guesses for eigenvalues
            x, eigval, k = rayleigh_iterate(matrix, random_vector(dim), matrix[int(i%dim), int(i%dim)])
            if i == 1:
                # Initialize lists containing the eigenvectors, eigenvalues, iterations k and rayleigh residuals
                x_list = x
                eigv_list = eigval
                k_list = k
                rayleigh_residual_list = np.linalg.norm(matrix @ x - eigval * x)
        else:  # If all eigenvalues not found after using all Gershgorin centers as guesses,
            # change guesses to averages between adjacent centers

            average_between_centers = 0.5 * (
                        matrix[int(i % dim), int(i % dim)] + matrix[int((i + 1) % dim), int((i + 1) % dim)])
            x, eigval, k = rayleigh_iterate(matrix, random_vector(dim), average_between_centers)

        if np.all(np.abs(eigv_list - eigval) > error_tolerance_eigval):  # make sure that the eigenvalue has not already been found
            x_list = np.r_['0,2', x_list, x]  # add eigenvector as a new row [0 means concatenate along rows, 2 means new row]
            eigv_list = np.r_[eigv_list, eigval]  # add to list
            k_list = np.r_[k_list, k]
            rayleigh_residual_list = np.r_[rayleigh_residual_list, np.linalg.norm(matrix @ x - eigval * x)]


    x_list, eigv_list, rayleigh_residual_list, k_list = order_lists(x_list, eigv_list, rayleigh_residual_list, k_list,
                                                                    dim)
    return x_list, eigv_list, rayleigh_residual_list, k_list


def find_eigenvectors_and_eigenfunctions_non_defective(A, error_tolerance_eigval = 1e-6,amplification = 1500):
    matrix = np.ndarray.copy(np.array(A,dtype='float'))
    dim, dim = np.shape(matrix)
    eigv_list = []

    #create list of perturbations that can be made to the original guesses if no new solution found.
    # Create it such that it takes negative and positive values
    guess_perturbation = np.linspace(1,10,10)
    for j in range(np.size(guess_perturbation)):
        if j%2 != 0:
            guess_perturbation[j] = - guess_perturbation[j]
    guess_perturbation *= amplification

    #number of times we have tried to change a given initial guess with a perturbation
    n_failed_guesses = 0
    i = 0
    max_iterations = dim
    try_new_guess = False
    #variable that keeps track of iterations of failed attempts to find new solutions
    additional_rayleigh_iterations = 0

    while np.size(eigv_list) < dim and i < max_iterations:  # continue until all eigenvalues are found
        # Initialize lists containing the eigenvectors, eigenvalues, iterations k and rayleigh residuals
        if i == 0:
            x, eigval, k = rayleigh_iterate(matrix, random_vector(dim), matrix[int(i), int(i)])
            x_list = x
            eigv_list = eigval
            k_list = k
            rayleigh_residual_list = np.linalg.norm(matrix @ x - eigval * x)
            i += 1
        # Try with the Gerschgorin center as a first guess
        else:
            if try_new_guess == False:
                x, eigval, k = rayleigh_iterate(matrix, random_vector(dim), matrix[int(i), int(i)])
            else:
                x, eigval, k = rayleigh_iterate(matrix, random_vector(dim), matrix[int(i), int(i)]+guess_perturbation[n_failed_guesses])
            #If eigenvalue is already in list, try new guess or continue if all guesses have failed:
            if np.all(np.abs(eigv_list - eigval) < error_tolerance_eigval):
                additional_rayleigh_iterations += k                 # keep track of iterations used on failed attempts
                # If the Gerschgorin center didn't yield a new solution, add a perturbation to the guess and try again
                if try_new_guess == False:
                    try_new_guess = True
                    continue
                # If the perturbation didn't work, try a new one
                else:
                    if n_failed_guesses < np.size(guess_perturbation)-1:
                        n_failed_guesses += 1
                        continue
                # If none of the perturbations gave a new solution, move on to next center and reset
                    else:
                        n_failed_guesses = 0
                        try_new_guess = False
                        i += 1
                        continue
            # We have found a new solution, and we collect it.
            else:
                x_list = np.r_['0,2', x_list, x]  # add eigenvector as a new row [0 means concatenate along rows, 2 means new row]
                eigv_list = np.r_[eigv_list, eigval]  # add to list
                k_list = np.r_[k_list, k+additional_rayleigh_iterations]
                rayleigh_residual_list = np.r_[rayleigh_residual_list, np.linalg.norm(matrix @ x - eigval * x)]
                i +=1
                additional_rayleigh_iterations = 0
                #Reset so as to use the Gerschgorin center as a first guess
                if try_new_guess == True:
                    try_new_guess = False
                    n_failed_guesses = 0


    x_list, eigv_list, rayleigh_residual_list, k_list = order_lists(x_list, eigv_list, rayleigh_residual_list, k_list,
                                                                    dim)
    return x_list, eigv_list, rayleigh_residual_list, k_list




def print_eigenvectors_and_eigenfunctions(x_list, eigv_list, rayleigh_residual_list, k_list):
    dim = len(x_list)
    for l in range(dim):
        print(f"Eigenvalue = {eigv_list[l]:.10f} ", f'Rayleigh residual = {rayleigh_residual_list[l]:.5e}' , f' # of iterations = {k_list[l]}'
                                                                                                        f'')




#def main():
#    pass

#if __name__=='__main__':
 #   main()

print("PROJECT 2","\n")
# Problem a: The Gershgorin function can be found above. To localize the disks, in which the eigenvalues of
# K are localized, we simply apply the aforementioned function

print("Gershgorin disk centers and radii for the matrix K, in the format (center,radius): ")
print()
print(gershgorin(K))
print("\n")

# Problem b: The functions rayleigh_qt and power_iterate can be found above.
# Now, we are going to test the power_iterate function by finding the largest eigenvalues of the test matricees,
# along with their Rayleigh residual and the number of iterations used

print("Using the power iteration functions to find the biggest eigenvalue and corresponding Rayleigh residual"
      " and number of iterations of the test matrices: ")
print()

g=1         #numbering the matrices
for A in [A1,A2,A3,A4,A5,A6]:
                                        # The function random_vector(n) can be found above and generates an nx1 matrix
                                        # with all entries taking random values in the interval [-1,1]
    eigvec, eigval, k = power_iterate(A,random_vector(np.size(A[:,0])))
    rayleigh_residual = np.linalg.norm (A @ eigvec - eigval * eigvec)

    print(f'Biggest eigenvalue of matrix A{g} = {eigval:.8f},', f' Reyleigh residual = {rayleigh_residual:.4e},', f' # of iterations used: ',k)
    g = g + 1


# Problem b, part 4): We are now going to find the largest eigenvalue of K and visualize the corresponding eigen-
# function and nodes by using show_waves and show_nodes

eigvec, eigval, k = power_iterate(K,random_vector(np.size(K[:,0])))


show_waves(eigvec,basis_set)
show_nodes(eigvec,basis_set)


# Problem c: The Rayleigh quotient iteration function can be found above. It has been implemented with
# qr_pivot_solve, a householder QR-solver with pivoting that is robust for singular matrices.

# In the following, I will find the eigenvalues, Rayleigh residuals and number of iterations used for test matrix A4.

# The program is able to find eigenvalues and eigenvectors of A5 and A6 as well, but these matrices
# have eigenvalues with multiplicity, and the function cannot determine the multiplicity of each eigenvalue.
# To do that, one would have to, while calculating eigenvalues and eigenvectors, check the linear dependence of
# eigenvectors corresponding to the same eigenvalue, only collecting linearly independent eigenvectors.
# In this manner, all eigenvectors, eigenvalues and their multiplicities can be determined.

print()
print("Now, the qr_solver used in the rayleigh iteration function is robust for singular matrices. However, since"
      " the test matrices A5 and A6 \neach have an eigenvalue with multiplicity, simply using the rayleigh iteration"
      " cannot determine the multiplicity of each eigenvalue. \nTherefore, I will report the results of applying the"
      " function on test matrix A4, which does not have eigenvalues with multiplicity. ")

# The program written below collects and orders the eigenvalues, eigenvectors, Rayleigh residuals and number of iterations
# of a matrix with non-degenerate eigenvalues. It is written in a way that makes it possible to reuse it for determining
# all eigenvalues and eigenvectors of the matrix K


print("\nEigenvalues of the matrix A4 along with the corresponding eigenvectors, Rayleigh residuals and number of iterations used: \n")
x_list,eigv_list,rayleigh_residual_list,k_list = find_eigenvectors_and_eigenfunctions_non_defective(A4,amplification=1)
print_eigenvectors_and_eigenfunctions(x_list,eigv_list,rayleigh_residual_list,k_list)


# Problem d: The answer to part 1) can be found in the report.
# In the following, we are going to calculate all eigenvalues and eigenvectors of the matrix K, along with the
# corresponding Rayleigh residuals and number of iterations for good measure.

print("\nEigenvalues of the matrix K along with the corresponding eigenvectors, Rayleigh residuals and number of iterations used: \n")
x_list,eigv_list,rayleigh_residual_list,k_list = find_eigenvectors_and_eigenfunctions_non_defective(K)
print_eigenvectors_and_eigenfunctions(x_list,eigv_list,rayleigh_residual_list,k_list)
dim, dim = np.shape(K)
#print(x_list[dim-1])

#Showing nodes for the eigenfunction with the lowest eigenvalue:
show_nodes(x_list[0, :], basis_set)

# Part 3: Since x_list already contain all eigenvectors (as row vectors) in order of ascending eigenvalues, the
# transformation matrix T is simply the transpose of x_list.

# In the following, we verify that K = T @ diag @ inv(T), where diag is a diagonal matrix whose entries are
# eigenvalues of K
dim, dim = np.shape(K)

print(x_list[0])
T = np.transpose(x_list)
diag = eigv_list * np.eye(dim)          # diagonal matrix whose entries are eigenvalues of K

inf_norm_difference = np.linalg.norm(T @ diag @ np.linalg.inv(T)-K,np.inf)

print(f"Let T denote the transformation matrix whose columns are eigenvectors of K. Let diag denote the"
      " diagonal matrix whose entries are eigenvalues of K. We confirm that K = T @ diag @ inv(T) by calculating the "
      f" infinity norm of their difference, which has the value {inf_norm_difference:.4e}, and so each element of K"
      f" and T @ diag @ inv(T) differ by this value at most")
print("\n")
print("\n")

#Finally, we visualize the nodes of all eigenfunctions of K:
show_all_wavefunction_nodes(np.transpose(x_list), eigv_list, basis_set)
