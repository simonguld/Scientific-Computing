import numpy as np


test_matrix = np.array([[2,1,1],[4,1,4],[-6,-5,3]])
test_vector = np.array([4,11,4])


def do_stuff_on_matrix (A):
    C = A
    for k in range(3):
        C += C
    return C

new_matrix = test_matrix
new_matrix = new_matrix/2
#new_matrix = do_stuff_on_matrix(test_matrix)
print(test_matrix)