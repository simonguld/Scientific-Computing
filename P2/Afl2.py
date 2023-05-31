#data
import numpy as np
import operator

Kmat = np.load("../Chladni-Kmat.npy")
from chladni_show import *


# A1-A3 should work with any implementation
A1   = np.array([[1,3],[3,1]])
eigvals1 = [4,-2]

A2   = np.array([[3,1],[1,3]])
eigvals2 = [4,2]

A3   = np.array([[1,2,3],[4,3.141592653589793,6],[7,8,2.718281828459045]])
eigvals3 = [12.298958390970709, -4.4805737703355,  -0.9585101385863923]

# A4-A5 require the method to be robust for singular matrices
A4   = np.array([[1,2,3],[4,5,6],[7,8,9]])
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



#chladni_basis = np.load("chladni_basis.npy")
test = np.array([[1,2,3],[4,5,6],[7,8,9]])
#a1
def gershgorin(A):
    a = np.shape(A)
    centers = np.zeros((a[0],1))
    radii = np.zeros((a[0],1))
    for i in range(a[0]):
        centers[i,0] = A[i,i]
        radii[i,0] = np.abs(np.sum(A[:,i])-A[i,i])
    return centers,radii
#a2
C,R = gershgorin(Kmat)
#print(C)
#print(R)

#b1

def rayleigh_qt(A,x):
    x_transp = np.transpose(x)
    lam =((np.matmul((np.matmul(x_transp,A)),x))/(np.matmul(x_transp,x)))
    res = np.linalg.norm(((np.matmul(A,x))-(lam*x)),'fro')
    #np.matmul(A,x0)-(max(abs(y))*x0)
    return lam,res
#test_func = rayleigh_qt(test,x_test)
#print(test_func)

#b2

def power_iterate(A):# I have also implemented that you don't need to feed an initial vector, but that it is created in the function.
    k = 0
    test_conv = ([[1],[1]])
    a = np.shape(A)
    x0 = np.ones((a[0],1))
    while np.linalg.norm(test_conv,'fro')>(10**-6):#the convergence criterion is that we test the approximate eigenvalue and eigenvector in the eigenvalue equation, and see how close it is to zero
        y =np.matmul(A,x0)
        x0 = y/max(abs(y))
        k = k+1
        test_conv = np.matmul(A,x0)-(max(abs(y))*x0)
    return x0, k
vec,it = power_iterate(Kmat)
print(vec)
print(it)

#b3 find largest eigenvalue of the example matrices, the Rayleigh residual and the number of iterations
# first step: use the power iteration to get the starting vector second step: use the starting vector for the Rayleigh quotient
#question: sensitvity for singular matrices: what should I be aware of?
#question: what is considered a good Rayleigh residual?
#A1
A1_vec,A1_it = power_iterate(A1)
A1_lam,A1_res = rayleigh_qt(A1,A1_vec)
print("A1")
print(A1_vec)
print(A1_it)
print(A1_lam)
print(A1_res)
#A2
A2_vec,A2_it = power_iterate(A2)
A2_lam,A2_res = rayleigh_qt(A2,A2_vec)
print("A2")
print(A2_vec)
print(A2_it)
print(A2_lam)
print(A2_res)
#A3
A3_vec,A3_it = power_iterate(A3)
A3_lam,A3_res = rayleigh_qt(A3,A3_vec)
print("A3")
print(A3_vec)
print(A3_it)
print(A3_lam)
print(A3_res)
#A4
A4_vec,A4_it = power_iterate(A4)
A4_lam,A4_res = rayleigh_qt(A4,A4_vec)
print("A4")
print(A4_vec)
print(A4_it)
print(A4_lam)
print(A4_res)
#A5
A5_vec,A5_it = power_iterate(A5)
A5_lam,A5_res = rayleigh_qt(A5,A5_vec)
print("A5")
print(A5_vec)
print(A5_it)
print(A5_lam)
print(A5_res)
#A6
A6_vec,A6_it = power_iterate(A6)
A6_lam,A6_res = rayleigh_qt(A6,A6_vec)
print("A6")
print(A6_vec)
print(A6_it)
print(A6_lam)
print(A6_res)

#b4 What is the largest eigenvalue of K? visualize the eigenfunction!
Kmat_vec,Kmat_it = power_iterate(Kmat)
Kmat_lam,Kmat_res = rayleigh_qt(Kmat,Kmat_vec)
print("Kmat")
print(Kmat_vec)
print("SHAPE = ", np.size(Kmat_vec))
print(Kmat_it)
print(Kmat_lam)
print(Kmat_res)
testbillede = show_waves(Kmat_vec.flatten())
######!!!!! Still need to visualize the eigenfunction when I get that part to work !!!!#######
#testvec = vector_to_function(Kmat_vec,basis_set)
#c1
# first implement a shifted inverse power algorithm:
#x = np.random.rand(3,1)
#print(x)
def inv_shift_power_iterate(A,shift):# so far it works for non-singular matrices and shifts of 0
    k = 0
    test_conv = ([[1],[1]])
    a = np.shape(A)
    x = np.random.rand(1,a[0])
    Ashifted = (A-(shift*np.identity(a[0])))#
    while  np.linalg.norm(test_conv,'fro')>(10**-6): #find real convergence criterion
        y = np.linalg.solve(Ashifted,x)
        x = y/max(abs(y))
        k=k+1
        max_index, max_value = max(enumerate(abs(y)), key=operator.itemgetter(1))
        y2 = np.matmul(A,y)
        test_conv = np.matmul(A,x)-(((np.sign(y[max_index])*np.sign(y2[max_index])/max_value)+shift)*x)
    return x, ((np.sign(y[max_index])*np.sign(y2[max_index])/max_value)+shift)# returns the eigenvector and the eigenvalue
    #return x,((1/max(abs(y)))+shift)


#x_A3,eig_A3 = inv_shift_power_iterate(A3, 4.7)
# in principle, one should be able to use the gershgorin disks for the shift input to the inv_shift_power_iterate function, but for matrix A3,
# the gershgorin disk are bad, because the matrix is not very dominated by the diagonal.
#print(x_A3)
#print(eig_A3)

def rayleigh_iterate(A,shift):# the initial shift could come from the centers of the gershgorin disks
    a = np.shape(A)
    #x = np.random.rand(a[0],1)
    # the one tracking the number of iterations in shifted inverse power
    k = 0 # the one tracking the number of iterations in the actual rayleigh iteration part
    x,vec = inv_shift_power_iterate(A,shift)
    res=1
    while res>10**-9:
        Ashifted = (A-(shift*np.identity(a[0])))
        y = np.linalg.solve(Ashifted,x)
        x = y/max(abs(y))
        shift,res = rayleigh_qt(A,x)
        k=k+1
    return x, k, shift, res

#c2: testing the Rayleigh iterate function on the test matrices
#A1:
print("A1")
x_rayit1A1,k_rayit1A1,lambda1A1, res_rayit1A1 = rayleigh_iterate(A1,1.1)
#print(x_rayit1A1)
#print(k_rayit1A1)
#print(lambda1A1)
#print(res_rayit1A1)
x_rayit2A1,k_rayit2A1,lambda2A1, res_rayit2A1 = rayleigh_iterate(A1,0)
#print(x_rayit2A1)
#print(k_rayit2A1)
#print(lambda2A1)
#print(res_rayit2A1)
#A2:
#print("A2")
x_rayit1A2,k_rayit1A2,lambda1A2, res_rayit1A2 = rayleigh_iterate(A2,3.2)
#print(x_rayit1A2)
#print(k_rayit1A2)
#print(lambda1A2)
#print(res_rayit1A2)
x_rayit2A2,k_rayit2A2,lambda2A2, res_rayit2A2 = rayleigh_iterate(A2,0)
#print(x_rayit2A2)
#print(k_rayit2A2)
#print(lambda2A2)
#print(res_rayit2A2)
#A3:
#print("A3")
x_rayit1A3,k_rayit1A3,lambda1A3, res_rayit1A3 = rayleigh_iterate(A3,10)
#print(x_rayit1A3)
#print(k_rayit1A3)
#print(lambda1A3)
#print(res_rayit1A3)
x_rayit2A3,k_rayit2A3,lambda2A3, res_rayit2A3 = rayleigh_iterate(A3,-4)
#print(x_rayit2A3)
#print(k_rayit2A3)
#print(lambda2A3)
#print(res_rayit2A3)
x_rayit3A3,k_rayit3A3,lambda3A3, res_rayit3A3 = rayleigh_iterate(A3,0)
#print(x_rayit3A3)
#print(k_rayit3A3)
#print(lambda3A3)
#print(res_rayit3A3)

#d1
#why can't you get all eigenvalues with pure power iteration?
#d2 Use Rayleigh iteration plus gershgorin disks to get all/as many eigenvalues as possible
print(C)

x_Kmat1,k_Kmat1,lambda_Kmat1,res_Kmat1 = rayleigh_iterate(Kmat,C[0])
print(lambda_Kmat1)
#x_Kmat2,k_Kmat2,lambda_Kmat2,res_Kmat2 = rayleigh_iterate(Kmat,C[1]-10000)#9041.825)# this eigenvalue is extremely sensitive to the shift!! it has to be precise to third decimal after comma to be found
#you only get the eigenvalue if you undershoot the shift(if the shift is less than the actual eigenvalue)
#print(lambda_Kmat2)
#x_Kmat3,k_Kmat3,lambda_Kmat3,res_Kmat3 = rayleigh_iterate(Kmat,C[2]-13000)#12201.2899)#this is also extrmely sensitive to the shift!
#print(lambda_Kmat3)
#This one also requires exact shift, or undershoot of shift
x_Kmat4,k_Kmat4,lambda_Kmat4,res_Kmat4 = rayleigh_iterate(Kmat,C[3])
print(lambda_Kmat4)
#x_Kmat5,k_Kmat5,lambda_Kmat5,res_Kmat5 = rayleigh_iterate(Kmat,C[4]-130) # this also requires undershooting of the shift
#print(lambda_Kmat5)
#x_Kmat6,k_Kmat6,lambda_Kmat6,res_Kmat6 = rayleigh_iterate(Kmat,C[5]-5220) # this also requires undershooting
#print(lambda_Kmat6)
#x_Kmat7,k_Kmat7,lambda_Kmat7,res_Kmat7 = rayleigh_iterate(Kmat,C[6]-1600) # requires undershooting
#print(lambda_Kmat7)
#x_Kmat8,k_Kmat8,lambda_Kmat8,res_Kmat8 = rayleigh_iterate(Kmat,C[7]+1000) # (gershorin is too low)
#print(lambda_Kmat8)
#x_Kmat9,k_Kmat9,lambda_Kmat9,res_Kmat9 = rayleigh_iterate(Kmat,C[8]-2400) # reguires overshooting
#print(lambda_Kmat9)
#x_Kmat10,k_Kmat10,lambda_Kmat10,res_Kmat10 = rayleigh_iterate(Kmat,C[9]-40) # undershooting
#print(lambda_Kmat10)
#x_Kmat11,k_Kmat11,lambda_Kmat11,res_Kmat11 = rayleigh_iterate(Kmat,C[10]+6.26767103e+02) # I cant find this guys's eigenvalue
#print(lambda_Kmat11)
x_Kmat12,k_Kmat12,lambda_Kmat12,res_Kmat12 = rayleigh_iterate(Kmat,C[11])
print(lambda_Kmat12)
#x_Kmat13,k_Kmat13,lambda_Kmat13,res_Kmat13 = rayleigh_iterate(Kmat,C[12]-2)
#print(lambda_Kmat13)
#x_Kmat14,k_Kmat14,lambda_Kmat14,res_Kmat14 = rayleigh_iterate(Kmat,C[13]-1)
#print(lambda_Kmat14)
x_Kmat15,k_Kmat15,lambda_Kmat15,res_Kmat15 = rayleigh_iterate(Kmat,0)
print(lambda_Kmat15)



#print(lambda_Kmat2)
#print(lambda_Kmat3)
#print(lambda_Kmat4)
#print(lambda_Kmat5)
#print(lambda_Kmat6)
#print(lambda_Kmat7)
#print(lambda_Kmat8)
#print(lambda_Kmat9)
#print(lambda_Kmat10)
#print(lambda_Kmat11)
#print(lambda_Kmat12)
#print(lambda_Kmat13)
#print(lambda_Kmat14)
#print(lambda_Kmat15)
eige,l = np.linalg.eig(Kmat)

eige.shape = (15,1)
#print(eige)
#d3
#d4