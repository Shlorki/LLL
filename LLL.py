import numpy as np

def LLL(A):
    n = A[0].size
    k = 1
    k_max = 0
    B = np.zeros((n,n))
    B[0] = A[0]
    b = np.zeros(n)
    b[0] = np.dot(A[0],A[0])
    H = np.identity(n)
    U = np.identity(n)
    
    def RED(k,l):
        if abs(U[k][l]) > .5:
            q = round(U[k][l])
            A[k] = A[k] - q*A[l]
            H[k] = H[k] - q*H[l]
            U[k][l] = U[k][l] - q
            for i in range(l):
                U[k][i] = U[k][i] - q*U[l][i]
    
    def SWAP(k):
        A[[k,k-1]] = A[[k-1,k]]
        H[[k,k-1]] = H[[k-1,k]]
        
        if k > 1:
            for j in range(k-1):
                u = U[k-1][j]
                U[k-1][j] = U[k][j]
                U[k][j] = u
                
        u = U[k][k-1]
        c = b[k] + (u ** 2)*b[k-1]
        U[k][k-1] = u*b[k-1]/c
        
        C = B[k-1]
        B[k-1] = B[k] + u*C
        B[k] = -U[k][k-1]*B[k] + b[k]*C/c
        b[k] = b[k-1]*b[k]/c
        b[k-1] = c
        
        for i in range(k+1,k_max+1):
            t = U[i][k]
            U[i][k] = U[i][k-1] - u*t
            U[i][k-1] = t + U[k][k-1]*U[i][k]
    
    def Test(k):
        RED(k,k-1)
        
        while b[k] < (.75 - U[k][k-1] ** 2)*b[k-1]:
            SWAP(k)
            k = np.max([1,k-1])
            RED(k,k-1)
        
        for l in reversed(range(k-1)):
            RED(k,l)
        return k+1
        
    def inc_GSO(k):
        B[k] = A[k]

        for j in range(k):
            U[k][j] = np.dot(A[k],B[j])/b[j]
            B[k] += -U[k][j]*B[j]

        b[k] = np.dot(B[k],B[k])
            
    while k <= n-1:
        if k <= k_max:
            k = Test(k)

        else:
            k_max += 1
            inc_GSO(k)
            
            k = Test(k)
            
    return A,H,U,b