import numpy as np

def LLLI(B):
    n = B[0].size
    k = 2
    k_max = 1
    d = np.zeros(n+1)
    d[0] = 1
    d[1] = np.dot(B[0],B[0])
    H = np.identity(n)
    U = np.identity(n)
    
    def REDI(k,l):
        if abs(2*U[k-1][l-1]) > d[l]:
            q = round(U[k-1][l-1]/d[l])
            B[k-1] = B[k-1] - q*B[l-1]
            H[k-1] = H[k-1] - q*H[l-1]
            U[k-1][l-1] = U[k-1][l-1] - q*d[l]
            for i in range(l):
                U[k-1][i-1] = U[k-1][i-1] - q*U[l-1][i-1]
    
    def SWAPI(k):
        B[[k-1,k-2]] = B[[k-2,k-1]]
        H[[k-1,k-2]] = H[[k-2,k-1]]

        if k > 2:
            for j in range(k-1):
                u = U[k-2][j-1]
                U[k-2][j-1] = U[k-1][j-1]
                U[k-1][j-1] = u

        u = U[k-1][k-2]
        c = (d[k-2]*d[k] + u ** 2)/d[k-1]

        for i in range(k+1,k_max+1):
            t = U[i-1][k-1]
            U[i-1][k-1] = (d[k]*U[i-1][k-2]-u*t)/d[k-1]
            U[i-1][k-2] = (c*t + u*U[i-1][k-1])/d[k]
        d[k-1] = c
    
    def Test(k):
        REDI(k,k-1)
        
        while d[k]*d[k-2] < 3*(d[k-1] ** 2)/4 - U[k-1][k-2] ** 2:
            SWAPI(k)
            k = np.max([2,k-1])
            REDI(k,k-1)
        
        for l in reversed(range(1,k-1)):
            REDI(k,l)
        return k+1
        
    def inc_GSOI(k):
        for j in range(1,k+1):
            u = np.dot(B[k-1],B[j-1])
            for i in range(1,j):
                u = (d[i]*u - U[k-1][i-1]*U[j-1][i-1])/d[i-1]
            if j < k:
                U[k-1][j-1] = u
            else:
                d[k] = u
            
    while k <= n:
        if k <= k_max:
            k = Test(k)

        else:
            k_max += 1
            inc_GSOI(k)
            
            k = Test(k)
            
    return B,H,U,d