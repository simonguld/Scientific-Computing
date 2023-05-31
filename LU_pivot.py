import numpy as np
from scipy.linalg import lu

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
        print ("Matrix not square!")
        return

    #initialize permutation matrix, L and copy A
    perm_index = 1
    #Collect permutations in the list P
    if perm_index:
        P = np.linspace(0,n-1,n)
    #Alternatively the more expensive matrix approach
    else:
        P = np.identity(n)

    L = np.identity(n)
    #Copy to avoid altering A, and make sure that copy type is float
    A_copy = np.ndarray.copy(np.array(A,dtype='float'))

    for k in range(n-1):
        # Perform pivoting:
            #for each column, find index corresponding to entry with largest absolute value
        max_index = k+np.argmax(np.abs(A_copy[k:n, k])) #add k as argmax returns index starting from k
        #Swap rows k and max_index
        if k != max_index:
            A_copy[[k,max_index],k:n] = A_copy[[max_index,k],k:n]
            if perm_index: #swap list elements
                P[[k,max_index]] = P[[max_index,k]]
            else: #swap rows
                P[[k, max_index], :] = P[[max_index, k], :]
            if k > 0:
                #do not swap for first column. Swap the left - already calculated part of L from 0:k
                #this way, L is calculated starting from I at each step, and then we are swapping the already
                #calculated part of L, to take into account the swapping after the fact
                L[[k, max_index], 0:k] = L[[max_index, k], 0:k]

        tol = 1e-12
        if np.abs(A_copy[k,k])<tol:                                     #Move on if biggest entry is 0
            continue


        L[k + 1:n, k] = A_copy[k + 1:n, k] / A_copy[k, k]  # Apply on entire row at once to avoid for-loop
        for j in range(k + 1, n):  # Apply algorithm, updating L and A simultaneously
            # L is ind. of j!
            A_copy[k + 1:n, j] = A_copy[k + 1:n, j] - L[k + 1:n, k] * A_copy[k, j]

        #EQUIVALENTLY
        """
                    for j in range(k + 1, n):  # Apply algorithm, updating L and A simultaneously
            # L is ind. of j!
            L[j, k] = A_copy[j, k] / A_copy[k, k]  # Apply on entire row at once to avoid for-loop
            A_copy[j, k:n] = A_copy[j, k:n] - L[j, k] * A_copy[k, k:n]
        """
    if perm_index:
        print ("P = ", P)
        #calculate permuation matrix
        permutation_matrix = np.zeros([n,n])
        for i in range(n):
            permutation_matrix [i,int(P[i])]=1
        U = np.triu(A_copy)  # Obtain U as the upper tringular part of matrix A
        return permutation_matrix, L, U
    else:
        U = np.triu(A_copy)                                          #Obtain U as the upper tringular part of matrix A
        return P, L, U

def forward_substitution(A,b):
    n = int(np.sqrt(np.size(A)))
    y = np.zeros(n)                                         # Initial values of y set to 0
    for k in range(n):
        if A[k,k]==0:                                       #Abort if diagonal element vanishes
            print("\n","Vanishing diagonal entries!","\n")
            break
        else:                                               #Calculate values using a dot product to avoid a for-loop.
            y[k] = ( b[k] - np.dot(A[k,:],y) ) / A[k,k]     #By setting the initial values of y to zero, the full
                                                            # dot product can be used every time, thus avoiding
                                                            # complicated indexing
    return y

def backward_substitution(A,b):
    n = int(np.sqrt(np.size(A)))
    y = np.zeros(n)                                          # Initial values of y set to 0
    for k in range(n-1,-1,-1):
        if A[k,k]==0:                                        #Abort if diagonal element vanishes
            print("\n","Vanishing diagonal entries!","\n")
            break
        else:                                               #Calculate values using a dot product to avoid a for-loop.
             y[k] = ( b[k]-np.dot(A[k,:],y) ) / A[k,k]      #By setting the initial values of y to zero, the full
                                                            # dot product can be used every time, thus avoiding
                                                            # complicated indexing
    return y

def LU_decomposition(A):
    m, n = np.shape(A)
    if m != n:
        print("Matrix not square!")
        return

    L = np.identity(n)
    A_copy = np.ndarray.copy(np.array(A,dtype='float'))
    for k in range(n-1):
        if A_copy[k,k]==0:                                       #Abort if diagonal element vanishes
            print("\n","Vanishing diagonal entries!","\n")
            return
        for j in range(k+1,n):                              #Apply algorithm, updating L and A simultaneously
            # L is ind. of j!
                L[k+1:n, k] = A_copy[k+1:n, k] / A_copy[k, k]         #Apply on entire row at once to avoid for-loop
                A_copy[k+1:n,j] = A_copy[k+1:n,j] - L[k+1:n,k]*A_copy[k,j]
    U = np.triu(A_copy)                                          #Obtain U as the upper tringular part of matrix A
    return L,U

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
omega_values = np.array([1.300,1.607,2.700])
omega_uncertainty = 0.5e-3
n_omega = int(np.size(omega_values))

E = np.block ([[Amat,Bmat],[Bmat,Amat]])                            # constructing E and S from submatrices
S = np.block([[np.identity(7),np.zeros([7,7])],[np.zeros([7,7]),-np.identity(7)]])
z = np.block ([yvec,-yvec])

A = np.array([[1,9,0,-3],[1,9,3,4],[-4,4,1,-2],[1,4,5,2]])
print (np.linalg.inv(A))
b = np.array([3,0,-4,2])


"""
L, U = LU_decomposition(A)

print (A, "\n \n")
print (L , "\n \n")
print (U , "\n \n")
print ( L @ U)

y = forward_substitution(L,b)
x = backward_substitution(U,y)
print (x)
print ("\n \n", A@x, "     ", b)
"""

P, L, U = LU_decomposition_with_pivoting(A)
p,l,u = lu(A)
print (A, "\n \n")
print (P , "\n \n")
print("\n \n s ", p)
print ("our L = ", L , "\n \n")
print("\n \n s ", l)
print("\n")
print ("our U = ", U , "\n \n")
print("\n \n s ", u)
print (P @ A, "\n \n")
print (l @ u, "\n \n")
print ( L @ U)


y = forward_substitution(L,P @ b)
x = backward_substitution(U,y)
print (x)
print ("\n \n", A@x, "     ", b)


for omega in omega_values:
    P,L,U = LU_decomposition_with_pivoting(E-omega*S)
    mat = E-omega*S
    print ("\n \n", np.linalg.det(P @ mat))
    print("\n \n", np.linalg.det(L @ U))
    y = forward_substitution(L,P@z)
    x = backward_substitution(U,y)
    print(f"with omega in pivot = {omega} , alpha = ", np.dot(z,x) )

for omega in omega_values:
    L,U = LU_decomposition(E-omega*S)
    print ("\n \n", np.linalg.det(mat))
    print("\n \n", np.linalg.det(L @ U))
    y = forward_substitution(L,z)
    x = backward_substitution(U,y)
    print(f"with omega no pivot = {omega} , alpha = ", np.dot(z,x) )
