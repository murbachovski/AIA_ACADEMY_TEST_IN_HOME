import numpy as np

x1 = np.array([[1, 2], [3, 4], [5, 6]])
x2 = np.array([[[1, 2, 3], [4, 5, 6]]])
x3 = np.array([[[1, 2]]])
x4 = np.array([[1], [2], [3], [4], [5]])
x5 = np.array([[[1]], [[2]], [[3]], [[4]]])
x6 = np.array([
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    ])
x7 = np.array([
    [[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]], [[9, 10]]
    ]) 
x8 = np.array([[[1, 2]], [[3, 4]], [[5, 6]]]) 
x9 = np.array([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]], [[9, 10]], [[11, 12]]]) 
x10 = np.array([[[1, 2]], [[3, 4]]])

print(x1.shape) #(3, 2)
print(x2.shape) #(2, 3) (1, 2, 3)
print(x3.shape) #(1, 2) (1, 1, 2)
print(x4.shape) #(1, 5) (5, 1)
print(x5.shape) #(1, 4) (4, 1, 1)
print(x6.shape) #(2, 5) (1, 5, 2)
print(x7.shape) #(2, 5) (5, 1, 2)
print(x8.shape) #(2, 3) (3, 1, 2)
print(x9.shape) #(2, 6) (6, 1, 2)
print(x10.shape) #(2, 2) (2, 1, 2)