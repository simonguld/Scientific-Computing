import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision = 14, suppress = True)        #set values <e-14 to 0 when printing to avoid clutter
                                                            # in matrices
Amat = np.array([
    [22.13831203, 0.16279204, 0.02353879, 0.02507880,-0.02243145,-0.02951967,-0.02401863],
    [0.16279204, 29.41831006, 0.02191543,-0.06341569, 0.02192010, 0.03284020, 0.03014052],
    [0.02353879,  0.02191543, 1.60947260,-0.01788177, 0.07075279, 0.03659182, 0.06105488],
    [0.02507880, -0.06341569,-0.01788177, 9.36187184,-0.07751218, 0.00541094,-0.10660903],
    [-0.02243145, 0.02192010, 0.07075279,-0.07751218, 0.71033323, 0.10958126, 0.12061597],
    [-0.02951967, 0.03284020, 0.03659182, 0.00541094, 0.10958126, 8.38326265, 0.06673979],
    [-0.02401863, 0.03014052, 0.06105488,-0.10660903, 0.12061597, 0.06673979, 1.15733569]])
Bmat = np.array([
    [-0.03423002, 0.09822473,-0.00832308,-0.02524951,-0.00015116, 0.05321264, 0.01834117],
    [ 0.09822473,-0.51929354,-0.02050445, 0.10769768,-0.02394699,-0.04550922,-0.02907560],
    [-0.00832308,-0.02050445,-0.11285991, 0.04843759,-0.06732213,-0.08106876,-0.13042524],
    [-0.02524951, 0.10769768, 0.04843759,-0.10760461, 0.09008724, 0.05284246, 0.10728227],
    [-0.00015116,-0.02394699,-0.06732213, 0.09008724,-0.07596617,-0.02290627,-0.12421902],
    [ 0.05321264,-0.04550922,-0.08106876, 0.05284246,-0.02290627,-0.07399581,-0.07509467],
    [ 0.01834117,-0.02907560,-0.13042524, 0.10728227,-0.12421902,-0.07509467,-0.16777868]])

yvec= np.array([-0.05677315,-0.00902581, 0.16002152, 0.07001784, 0.67801388,-0.10904168, 0.90505180])
omega_values = np.array([0.800,1.146,1.400])
omega_uncertainty = 0.5e-3
n_omega = int(np.size(omega_values))

E = np.block ([[Amat,Bmat],[Bmat,Amat]])                            # constructing E and S from submatrices
S = np.block([[np.identity(7),np.zeros([7,7])],[np.zeros([7,7]),-np.identity(7)]])
z = np.block ([yvec,-yvec])

def norm_inf (A):
    n = int(np.size(A[0,:]))
    max = sum(np.abs(A[0, :]))           # Set absolute sum of 1st row = infinity norm of A
    for k in range(1, n):
        if sum(np.abs(A[k, :])) > max:   # Search for rows with a bigger absolute sum
            max = sum(np.abs(A[k, :]))   # Update value of infinity norm if a bigger absolute sum is found
    return max

def condition_number_inf (A):
    return norm_inf(A) * norm_inf (np.linalg.inv(A))

def LU_decomposition(A):
    ###
    # In LU-decomposition without pivoting, we are decomposing a square matrix A such that A = LU, where
    # L is a lower triangular matrix, U is an upper
    # Even if A is invertible, such a composition might not exist
    # It breaks down for vanishing pivots and might overflow for small pivots
    # The algorithm is improved by incorporating pivoting --> see function LU_decomposition_with_pivoting

    # We can solve Ax = b by noting that Ax = = LUx = b, and so we solve
    # 1: Ly = b by forward substitution
    # 2: Ux = y by backward substitution

    ###

    m, n = np.shape(A)
    if m != n:
        print("Matrix not square!")
        return

    L = np.identity(n)
    A_copy = np.ndarray.copy(A)
    for k in range(n-1):
        if A_copy[k,k]==0:                                       #Abort if diagonal element vanishes
            print("\n","Vanishing diagonal entries!","\n")
            return

        # FOR BOTH APPROACHES: It is Gaussian elimination in both cases.
        # Since we know that the k'th column will be annihilated by these operations, we don't bother calculating it.
        # What's interesting is thus only what happens to the remaining submatrix A[k+1:n,k+1:n].

        # Approach 1: Column approach: Calc the entire k'th column of L. It has entries L[i,j] = A[i,j] / A[j,j].
        # Go on to calculate the new columns of A from k+1 to n.
        """
        L[k + 1:n, k] = A_copy[k + 1:n, k] / A_copy[k, k]  # Apply on entire row at once to avoid for-loop
        for j in range(k + 1, n):  # Apply algorithm, updating L and A simultaneously
            # L is ind. of j!
            A_copy[k + 1:n, j] = A_copy[k + 1:n, j] - L[k + 1:n, k] * A_copy[k, j]
        """
        # EQUIVALENTLY / Approach 2:
        # For each row in the submatrix A[k+1:n,k+1:n], subtract row k multiplied bt L[j,k] - this ensures that
        # entries A[k:n,k] are all annihilated.

        for j in range(k + 1, n):  # Apply algorithm, updating L and A simultaneously
            # L is ind. of j!
            L[j, k] = A_copy[j, k] / A_copy[k, k]  # Apply on entire row at once to avoid for-loop
            A_copy[j, k + 1:n] = A_copy[j, k + 1:n] - L[j, k] * A_copy[k, k + 1:n]


    U = np.triu(A_copy)                                          #Obtain U as the upper tringular part of matrix A
    return L,U

def LU_decomposition_with_pivoting(A):
    ###
    # In LU-decomposition with pivoting, we are decomposing a square matrix A such that PA = LU, where
    # L is a lower triangular matrix, U is an upper, and P is a permutation matrix.
    # If A is invertible, such a composition exists and is unique.
    # It has the advantage of working in case of vanishing diagonal elements and reduces overflow by avoiding
    # division with small pivots.

    # We can solve Ax = b by noting that PAx = = LUx = Pb, and so we solve
    # 1: Ly = Pb by forward substitution
    # 2: Ux = y by backward substitution

    ###

    m, n = np.shape(A)
    if m != n:
        print("Matrix not square!")
        return

    # initialize permutation matrix, L and copy A
    perm_index = 1
    # Collect permutations in the list P
    if perm_index:
        P = np.linspace(0, n - 1, n)
    # Alternatively the more expensive matrix approach
    else:
        P = np.identity(n)

    L = np.identity(n)
    # Copy to avoid altering A, and make sure that copy type is float
    A_copy = np.ndarray.copy(np.array(A, dtype='float'))

    for k in range(n - 1):
        # Perform pivoting:
        # for each column, find index corresponding to entry with largest absolute value
        max_index = k + np.argmax(np.abs(A_copy[k:n, k]))  # add k as argmax returns index starting from k
        # Swap rows k and max_index
        if k != max_index:
            A_copy[[k, max_index], k:n] = A_copy[[max_index, k], k:n]
            if perm_index:  # swap list elements
                P[[k, max_index]] = P[[max_index, k]]
            else:  # swap rows
                P[[k, max_index], :] = P[[max_index, k], :]
            if k > 0:
                # do not swap for first column. Swap the left - already calculated part of L from 0:k
                # this way, L is calculated starting from I at each step, and then we are swapping the already
                # calculated part of L, to take into account the swapping after the fact
                L[[k, max_index], 0:k] = L[[max_index, k], 0:k]

        tol = 1e-12
        if np.abs(A_copy[k, k]) < tol:  # Move on if biggest entry is 0
            continue

        # FOR BOTH APPROACHES: It is Gaussian elimination in both cases.
        # Since we know that the k'th column will be annihilated by these operations, we don't bother calculating it.
        # What's interesting is thus only what happens to the remaining submatrix A[k+1:n,k+1:n].

        # Approach 1: Column approach: Calc the entire k'th column of L. It has entries L[i,j] = A[i,j] / A[j,j].
        # Go on to calculate the new columns of A from k+1 to n.
        """
        L[k + 1:n, k] = A_copy[k + 1:n, k] / A_copy[k, k]  # Apply on entire row at once to avoid for-loop
        for j in range(k + 1, n):  # Apply algorithm, updating L and A simultaneously
            # L is ind. of j!
            A_copy[k + 1:n, j] = A_copy[k + 1:n, j] - L[k + 1:n, k] * A_copy[k, j]
        """
        # EQUIVALENTLY / Approach 2:
        # For each row in the submatrix A[k+1:n,k+1:n], subtract row k multiplied bt L[j,k] - this ensures that
        # entries A[k:n,k] are all annihilated.

        for j in range(k + 1, n):  # Apply algorithm, updating L and A simultaneously
            # L is ind. of j!
            L[j, k] = A_copy[j, k] / A_copy[k, k]  # Apply on entire row at once to avoid for-loop
            A_copy[j, k + 1:n] = A_copy[j, k + 1:n] - L[j, k] * A_copy[k, k + 1:n]


    if perm_index:
        print("P = ", P)
        # construct permutation matrix
        permutation_matrix = np.zeros([n, n])
        for i in range(n):
            permutation_matrix[i, int(P[i])] = 1
        U = np.triu(A_copy)  # Obtain U as the upper tringular part of matrix A
        return permutation_matrix, L, U
    else:
        U = np.triu(A_copy)  # Obtain U as the upper tringular part of matrix A
    return P, L, U

def forward_substitution(A,b):
    n = int(np.size(b))
    y = np.zeros(n)                                         # Initial values of y set to 0
    for k in range(n):
        if A[k,k] == 0:                                       #Abort if diagonal element vanishes
            print("\n","Vanishing diagonal entries!","\n")
            return
        else:                                               #Calculate values using a dot product to avoid a for-loop.
            y[k] = ( b[k] - np.dot(A[k,:],y) ) / A[k,k]     #By setting the initial values of y to zero, the full
                                                            # dot product can be used every time, thus avoiding
                                                            # complicated indexing
        ## Alternativt y[k] = ( b[k] - np.sum(A[k,:]*y) ) / A[k,k]
    return y

def backward_substitution(A,b):
    n = int(np.size(b))
    y = np.zeros(n)                                          # Initial values of y set to 0
    for k in range(n-1, -1, -1):
        if A[k,k] == 0:                                        #Abort if diagonal element vanishes
            print("\n","Vanishing diagonal entries!","\n")
            return
        else:                                               #Calculate values using a dot product to avoid a for-loop.
             y[k] = ( b[k]-np.dot(A[k,:],y) ) / A[k,k]      #By setting the initial values of y to zero, the full
                                                            # dot product can be used every time, thus avoiding
                                                            # complicated indexing
    return y

def polarizability(w):
    L, U = LU_decomposition(E-w*S)                          # Obtain LU decomposition
    y = forward_substitution(L, z)                          # Solve Ly = z with y = Ux by forward substitution
    x = backward_substitution(U, y)                         # Solve Ux = y by backwards substitution
    return np.dot(z,x)

def apply_reflection(A, v):             #v an mx1 vector, A an mxn matrix
    c = - 2 * np.dot(v, A)              #Calculate the 1xn row vector transpose(v)*A
    A = A + c * v[:, np.newaxis]          # Perform reflection of A by creating a mxn matrix v[:,np.newaxis],
                                        # i.e a matrix V whose columns are all v, then multiplying each column V[i]
                                        # with the scalar c[i], and adding it to A
    return A

def householder_QR_factorization_slow(A):    # A an m x n matrix
    ###
    # Transform an mxn matrix A to QR, where Q is orthogonal and R[0:n,:] is upper triangular.
    # if A has full column rank, then upper part of R, namely R[0:n,:] is invertible.
    # Since orthogonal transformations preserve norms, they preserve the (norm defined) solution to the least squares
    # problem Ax ~ b, meaning that QRx = [c1, c2], where Rx = Q^t c1 = b, which can be solved by back substitution.
    ###

    m = np.size(A[:, 0])                 # numer of rows
    n = np.size(A[0, :])                 # number of columns

    Q = np.identity(m)
    R = np.ndarray.copy(np.array(A, dtype='float'))

    for k in range(n): ## Here we assume that m>n. Otherwise let k in range (min(n, m-1))
        # Initialize a as the bottom k:m part of column k. Applying Householder transformation will
        # annihilate all entries below entry (k,k)
        a = R[k:m, k]
        #make sure alpha has opposite sign of a[0]. use copysign instead og sign to avoid 0 when a = 0
        alpha = - np.copysign(1, a[0]) * np.linalg.norm(a)

        #obs: in algorithm 3.1, we check whether norm(v)=0, but one can easily show that norm(a)=0 (so alpha=0)
        # iff norm(v)=0 [cuz of the sign chosen for alpha]. the 2 conditions are equivalent.
        if alpha == 0:                                      # if alpha = 0, move on to next row
            print("Vanishing alpha, matrix rank deficient!","\n")
            continue
        else:
            v = np.ndarray.copy(np.array(a,dtype='float'))
            v[0] = v[0]-alpha           #construct v
            v = v / np.linalg.norm(v)   #normalize to avoid over/underflow

            ## Explanation of reflection (compare with alg. 3.1 on page 124):
            """
            for j in range(k,n):
                a_j -= gamma_j * v  equiv to R[k:m,j] -= 2 * np.dot(v,R[k:m,j])*v
            vectorizing loop gives us
            R[k:m,j:n] -= 2 * np.dot (v,R[k:m,j:n]) * v, but
            c = np.dot (v,R[k:m,j:n]) is an 1xn row vector v^T R[k:m,j:n] with scalar entries i c[i] = v^T * R[k:m,i], 
            so we have
             R[k:m,j:n] -= 2 [ c[k] * v, c[k+1] * v , ... , c[n] * v]
             Now, in apply_reflection, we use new axis to construct the mxn matrix V = [v,v,...,v] = v[:,np.newaxis].
             Since * yields elementwise multiplication in python, we simply find that
             R[k:m,j:n] -= 2* c* v[:,np.newaxis],
             which is exactly what the function apply_reflection does
            """

            R[k:m, k:n] = apply_reflection(R[k:m, k:n], v)     #Calculate values on entire sub-matrix to avoid for-loop
            Q[k:m, :] = apply_reflection(Q[k:m, :], v)         #On Q, we apply the reflection on all columns, as
                                                            #columns Q[k:m,i] do not necesarily vanish for i<k
    Q = np.transpose(Q)                                     #Algorithm yiels Q^T. Obtain Q by transposing
    return Q, R


def householder_QR_factorization(A):    # A an m x n matrix
    ###
    # In this optimized version of the Householder algorithm, we avoid calculating Q and simply collect the reflection
    # vector v. The are built into the nxn R matrix by simply collecting them under the diagonal.
    # Since the vector v have length m-k, the matrix VR will be of length m+1,n
    # By applying the reflection vectors iteratively on an object gives the same effect as multiplying with Q.T, which
    # we need to transform the right hand side. However, it is cheaper to apply these on the 1xm matrix b when needed
    # than to apply them to an mxm matrix to obtain Q.T explicitly.
    ###

    m, n = np.shape(A)                                   # numer of rows

    VR = np.ndarray.copy(np.array(A, dtype='float'))     # initialize R

    # add a row of zeros to make space for the reflection vectors below the diagonal
    VR = np.r_['0,2', VR, np.zeros(n)]

    for k in range(n): ## Here we assume that m>n. Otherwise let k in range (min(n, m-1))
        # Initialize a as the bottom k:m part of column k. Applying Householder transformation will
        # annihilate all entries below entry (k,k)
        a = VR[k:m, k]
        #make sure alpha has opposite sign of a[0]. use copysign instead og sign to avoid 0 when a = 0
        alpha = - np.copysign(1, a[0]) * np.linalg.norm(a)

        #obs: One can easily show that norm(a)=0 (so alpha=0)
        # iff norm(v)=0 [cuz of the sign chosen for alpha]. the 2 conditions are equivalent.
        if alpha == 0:                                      # if alpha = 0, move on to next row
            print("Vanishing alpha, matrix rank deficient!","\n")
            continue
        else:
            v = np.ndarray.copy(np.array(a,dtype='float'))
            v[0] = v[0]-alpha           #construct v
            v = v / np.linalg.norm(v)   #normalize to avoid over/underflow

            VR[k:m, k:n] = apply_reflection(VR[k:m, k:n], v)     #Calculate values on entire sub-matrix to avoid for-loop
            VR[k + 1::, k] = v                                  #Collect reflection vector v_k below the diagonal

    return VR

def least_squares_slow(A,b):                                     # A an mxn matrix
    ###
    # Transform an mxn matrix A to QR, where Q is orthogonal and R[0:n,:] is upper triangular.
    # if A has full column rank, then upper part of R, namely R[0:n,:] is invertible.
    # Since orthogonal transformations preserve norms, they preserve the (norm defined) solution to the least squares
    # problem Ax ~ b, meaning that QRx = [c1, c2], where R[0:n,:]x = Q^t c1 = b, which can be solved by back substitution.
    ###

    n = np.size(A[0, :])                                     # number of columns
    Q, R = householder_QR_factorization_slow(A)                   # Obtain Q,R by householder factorization
    Q_transpose = np.transpose(Q)                           # Obtain Q_transpose
    b_augmented = np.dot(Q_transpose, b)                     # Obtain 1xm vector Q_transpose*b

                                                            #Find least squares solution to the quadratic
                                                            #upper part of matrix equation
    print("rold =", R[0:n,:], "\n \n")
    print("bold =", b_augmented[0:n], "\n \n")
    print("np_oldnorms", np.linalg.norm(Q), np.linalg.norm(R))
    x_least_squares = backward_substitution(R[0:n, :], b_augmented[0:n])
    return x_least_squares

def least_squares_fast(A,b):                                     # A an mxn matrix
    ###
    # This version of the least squares algorithm takes as input the matrix VR containing R as well as the reflection
    # vectors. To solve the system R @ x=Q.T @ b.
    # In this version, we apply the reflection vectors directly to B to obtain Q.T @ b.
    ###

    m, n = np.shape(A)
    b_copy = np.copy(b)
    VR = householder_QR_factorization(A)                   # Obtain VR by householder factorization

    for k in range(n):
        v = np.ndarray.copy(np.array(VR[k+1::,k], dtype='float')) # Extract reflection vector v_k
        if np.linalg.norm(v) < 1e-14:
            continue
        b_copy[k:m] = b_copy[k:m] - 2 * np.dot(v,b_copy[k:m]) * v   # apply reflection vectors

    #Extract R matrix from VR
    R = np.triu(VR[0:n,:])

    #Solve invertible part of system by least squares
    x_least_squares = backward_substitution(R, b_copy[0:n])
    return x_least_squares

def polynomial_approximation(x,omega):  #x is the coefficients of the polynomial approximation
    n = np.size(x)                      #the length of x decides the number of coefficients
    poly = 0
    for k in range(n):                  #construct polynomial approximation as a function of omega
        poly += x[k]*np.power(omega, 2*k)
    return poly
def rational_approximation(a,b,omega):
    poly1 = 0
    poly2 = 1
    for n in range(np.size(a)):             #Construct polynomial in the numerator
        poly1 += a[n]*np.power(omega, n)
    for n in range(np.size(b)):         #Construct polynomial in the denominator
        poly2 += b[n]*np.power(omega, n+1)
    rational = poly1 / poly2                #Construct rational approximation as a function of omega
    return rational

# Problem e:
# 1) We are going to plot the polarizability for frequencies in the range [1.2,4]. We use the numpy function
# vectorize in order to apply the function to all omega values at once. These values are stored in the vector
# polarizability_values

N = 1000
print("NEW")
omega_range = np.linspace(0.7,1.5,N)
polarizability_vectorized = np.vectorize(polarizability)
polarizability_values=polarizability_vectorized(omega_range)


# Since the polarizability has a singularity at around omega = 1.6069, and polynomials don't approximate singulari-
# ties very well, we shall restrict the approximation interval to the left of this frequency. Specifically, we
# choose omega_p = 1.57838

omega_p_index = 500               #index corresponding to omega_p
omega_p = omega_range[omega_p_index]

print("\n")
print(f'The end point of the approximation interval, omega_p, is chosen to be omega_p = {omega_p}')
print()

# In the following, we will construct the matrix V needed for using least sqauares to find the coefficients of
# the polynomial approximations

n_coefficient_1 = 5                     # number of coefficients in approximation 1 (8'th order polynomial)
n_coefficients_2 = 7                    # number of coefficients in approximation 2 (12'th order polynomial)
W=np.ones([1,omega_p_index])            # Construct the first row of the transpose of the target matrix V


for n in range(1,n_coefficients_2):     #Construct succesive rows of increasing power in omega
    W = np.block([[W],[np.power(omega_range[0:omega_p_index],2*n)]])
    V = np.transpose(W)                 #Transpose to obtain target matrix

p_matrix1 = V[:, 0:n_coefficient_1]  #Construct target matrix of approximation 1 by excluding omega to the 10'th and 12'th power
p_matrix2 = V                       #relabel target matrix of approximation 2

# We are now ready to use least squares to find coefficients of the polynomial approximations,
# i.e. the solutions to the matrix equations Vx = polarizability[0:omega_p_index]

x_p1 = least_squares_fast(p_matrix1, polarizability_values[0:omega_p_index])   # coefficients for n=4 approximation
x_p2 = least_squares_fast(p_matrix2, polarizability_values[0:omega_p_index])  # coefficients for n=6 approximation

#Using the function polynomial_approximation, which can be found alongside the other functions, we are now ready
#to calculate the values of the two polynomial approximations on the interval:

poly_values_1 = polynomial_approximation(x_p1, omega_range[0:omega_p_index])
poly_values_2 = polynomial_approximation(x_p2, omega_range[0:omega_p_index])

#We now calculate the relative error of the polynomial approximations relative to the polarizability along with the
#number of significant digits of the approximations for each frequency:

rel_error1 = np.abs(poly_values_1-polarizability_values[0:omega_p_index])/np.abs(polarizability_values[0:omega_p_index])
sig_digits1 = - np.log10(rel_error1)
average_sig_digits1 = np.average(sig_digits1)

rel_error2 = np.abs(poly_values_2-polarizability_values[0:omega_p_index])/np.abs(polarizability_values[0:omega_p_index])
sig_digits2 = - np.log10(rel_error2)
average_sig_digits2 = np.average(sig_digits2)

print(f'The average number of significant digits for the n=4 approximation on [1.2,omega_p] = {average_sig_digits1:.3}')
print(f'The average number of significant digits for the n=6 approximation on [1.2,omega_p] = {average_sig_digits2:.3}')

# Finally, we plot the significant digits of each approximation along with their average values:

print()
print(f'We note that the significant digits of each approximations decrease rapidly as we approach the singularity')
plt.figure()

plt.plot(omega_range[0:omega_p_index],sig_digits1,'b-',label='Significant digits for n = 4')
plt.plot(omega_range[0:omega_p_index],np.ones(omega_p_index)*average_sig_digits1,'b--',label='Average for n = 4')

plt.plot(omega_range[0:omega_p_index],sig_digits2,'m',label='Significant digits for n = 6')
plt.plot(omega_range[0:omega_p_index],np.ones(omega_p_index)*average_sig_digits2,'m--',label='Average for n = 6')

plt.legend(prop={'size': 10})
plt.xlim(0.7,omega_p)
plt.ylim(-1,4)
plt.xlabel("Frequency",fontsize = 16)
plt.ylabel("Significant digits", fontsize = 16)
plt.title("Accuracy of polynomial approximations",fontsize=17)
plt.show()

print("\n")
m1 = 2
m2 = 4

WQ=np.ones([1,N])            # Construct the first row of the transpose of the target matrix VQ

for n in range(1,m2+1):     #Construct succesive rows of increasing power in omega
    WQ = np.block([[WQ],[np.power(omega_range,n)]])
for n in range(1,m2+1):     #Construct successive rows of polarizability * increasing powers of omega
    WQ = np.block([[WQ], [-polarizability_values*np.power(omega_range, n)]])

VQ = np.transpose(WQ)                 #Transpose to obtain target matrix

q_matrix1 = np.block([VQ[:,0:m1+1],VQ[:,m2+1:m1+m2+1]])  #Extract part of VQ to obtain target matrix for m=2
q_matrix2 = VQ                                           #Relabel target matrix

# We are now ready to use least squares to find coefficients of the rational approximations,
# i.e. the solutions to the matrix equations VQx = polarizability_values

x_q1 = least_squares_fast(q_matrix1, polarizability_values)  #coefficients for approximation m=2
a_values_q1 = x_q1[0:m1+1]                              #collect coefficients belonging to polynomial in numerator
b_values_q1 = x_q1[m1+1::]                              #collect coefficients belonging to polynomial in denominator

x_q2 = least_squares_fast(q_matrix2, polarizability_values)  #repeat for m=4
a_values_q2 = x_q2[0:m2+1]
b_values_q2 = x_q2[m2+1::]


#Using the function rational_approximation, which can be found alongside the other functions, we are now ready
#to calculate the values of the two rational approximations on the interval [1.2,4]:

q_values_1 = rational_approximation(a_values_q1,b_values_q1, omega_range)
q_values_2 = rational_approximation(a_values_q2,b_values_q2, omega_range)

#We now calculate the relative error of the rational approximations relative to the polarizability along with the
#number of significant digits of the approximations for each frequency:

q_rel_error1 = np.abs(q_values_1-polarizability_values)/np.abs(polarizability_values)
q_sig_digits1 = - np.log10(q_rel_error1)
q_average_sig_digits1 = np.average(q_sig_digits1)

q_rel_error2 = np.abs(q_values_2-polarizability_values)/np.abs(polarizability_values)
q_sig_digits2 = - np.log10(q_rel_error2)
q_average_sig_digits2 = np.average(q_sig_digits2)

print(f'The average number of significant digits for the rational m=2 approximation on [1.2,4] = {q_average_sig_digits1:.3}')
print(f'The average number of significant digits for the m=4 approximation on [1.2,4] = {q_average_sig_digits2:.3}')


# Finally, we plot the significant digits of each approximation along with their average values:

print()
print(f'We note that we lose all significant digits of the m=2 approximation around the singularity, whereas this is'
      f' not the case for m=4. Evidently, m=2 is not a sufficiently high order approximation to correctly reproduce'
      f' the singularity')

plt.figure()

plt.plot(omega_range,q_sig_digits1,'b-',label='Significant digits for m = 2')
plt.plot(omega_range,np.ones(N)*q_average_sig_digits1,'b--',label='Average for m = 2')

plt.plot(omega_range,q_sig_digits2,'m',label='Significant digits for m = 4')
plt.plot(omega_range,np.ones(N)*q_average_sig_digits2,'m--',label='Average for m = 4')

plt.legend(prop={'size': 10})
plt.xlim(0.7,1.4)
plt.ylim(-2,8)
plt.xlabel("Frequency",fontsize = 16)
plt.ylabel("Significant digits", fontsize = 16)
plt.title("Accuracy of rational approximations",fontsize=17)
plt.show()

# Part 3): In this final part of the project, we will modify the rational approximation such that it can reproduce
# the polarizability correctly on [-4,4], an interval on which the polarizability has several singularities.

# We will use 1000 equally spaced points on this interval. m=4 is not sufficient to reproduce the singularities, and
# so we choose m=8 in our rational approximation of the polarizability on [-4,4]
# the method is identical to the previous section

N=3000
omega_range=np.linspace(-4,4,N)                                 #Redefining omega_range
polarizability_values = polarizability_vectorized(omega_range)  #Redefining polarizability_values

m2=8

WQ=np.ones([1,N])            # Construct the first row of the transpose of the target matrix VQ

for n in range(1,m2+1):     #Construct transpose of target matrix
    WQ = np.block([[WQ],[np.power(omega_range,n)]])
for n in range(1,m2+1):
    WQ = np.block([[WQ], [-polarizability_values*np.power(omega_range, n)]])

VQ = np.transpose(WQ)                 #Transpose to obtain target matrix
q_matrix2 = VQ                        #Relabel target matrix

# Least squares

x_q2 = least_squares_fast(q_matrix2, polarizability_values)
a_values_q2 = x_q2[0:m2+1]                              #collecting values for pol. in numerator
b_values_q2 = x_q2[m2+1::]                              # collecting values for pol. in denominator


#Calculating values of the rational approximation for m=8

q_values_2 = rational_approximation(a_values_q2,b_values_q2, omega_range)

#We now calculate the relative error of the rational approximation relative to the polarizability along with the
#number of significant digits for each frequency:

q_rel_error2 = np.abs(q_values_2-polarizability_values)/np.abs(polarizability_values)
q_sig_digits2 = - np.log10(q_rel_error2)
q_average_sig_digits2 = np.average(q_sig_digits2)

print("\n")
print(f'The average number of significant digits for the m=8 approximation on [-4,4] = {q_average_sig_digits2:.3}')

# plot the approximation alongside the polarizability
plt.figure()

plt.plot(omega_range,polarizability_values,'r-',label='Polarizability',linewidth='3')
plt.plot(omega_range,q_values_2,'b-',label='Rational approximation for m=8')

plt.legend(prop={'size': 10})
plt.xlim(-4,4)
plt.ylim(-20,20)
plt.xlabel("Frequency",fontsize = 16)
plt.ylabel("Polarizability", fontsize = 16)
plt.title("Polarizability of water",fontsize=17)
plt.show()


# Finally, we plot the significant digits of the m=8 approximation along with its average values:
plt.figure()

plt.plot(omega_range,q_sig_digits2,'m',label='Significant digits for m = 8')
plt.plot(omega_range,np.ones(N)*q_average_sig_digits2,'m--',label='Average for m = 8')

plt.legend(prop={'size': 12})
plt.xlim(-4,4)
plt.ylim(3,11)
plt.xlabel("Frequency",fontsize = 16)
plt.ylabel("Significant digits", fontsize = 16)
plt.title("Accuracy of rational approximation",fontsize=17)
plt.show()


