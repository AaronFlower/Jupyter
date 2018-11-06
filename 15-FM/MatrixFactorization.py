# -*- coding: utf-8 -*-

import numpy as np

def matrix_factorization(R, K, epochs, alpha, beta):
    M, N = R.shape
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K)

    idx = R != 0
    for epoch in range(epochs):
        for i in range(M):
            for j in range(N):
                if R[i, j] > 0:
                    eij = R[i, j] - np.dot(P[i, :], Q.T[:, j])
                    for k in range(K):
                        P[i, k] = P[i, k] + alpha * (2 * eij * Q.T[k, j] - beta * P[i, k])
                        Q.T[k, j] = Q.T[k, j] + alpha * (2 * eij * P[i, k] - beta * Q.T[k, j])

        estimateR = np.dot(P, Q.T)
        mse = np.sum([x * x for x in estimateR[idx] - R[idx]])
        if mse < 0.001:
            break
    return P, Q

if __name__ == '__main__':
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]
    R = np.array(R)
    K = 2
    epochs = 5000
    alpha = 0.0002
    beta = 0.02
    P, Q = matrix_factorization(R, K, epochs, alpha, beta)
    print("The estimate matrix is : ")
    print(np.dot(P, Q.T))

