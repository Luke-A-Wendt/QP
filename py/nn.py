import sig
import numpy as np
from foo import zeros, eye, randn

class nn(object):#TODO:
    def __init__(self,x,y,m):
        """
            m is a vector of integers giving each layer dimension
            
            z_i ∈ R^{m_i}, i ∈ [0,n-1]

            x ∈ R^{ x.shape[0], q }
            y ∈ R^{ m[n-1] , q }

            A[i] ∈ R^{ m[i], m[i-1] }
            A[0] ∈ R^{ m[0], x.shape[0] }

            b[i] ∈ R^{ m[i] }

        """
        self.sigmoid = sig.poly1()
        assert x.shape[1] == y.shape[1]
        assert y.shape[0] == m[-1]
        self.x = x
        self.y = y
        self.n = len(m)
        self.z = []
        self.A = []
        self.b = []
        for i in range(self.n):
            if i == 0:
                self.A.append( randn(m[0], x.shape[0]) )
            else:
                self.A.append( randn(m[i],m[i-1]) )
            self.b.append( randn(m[i]) )
            self.z.append( randn(m[i]) )
        self.train()
    def train(self):
        x = self.x[:,0]
        y = self.y[:,0]
        self.update_z(x)
        print(self.z)
        # TODO: compute gradients
    def update_z(self,x):
        for i in range(self.n):
            if i == 0:
                self.z[i]=self.s(self.A[0]*x-self.b[0])
            else:
                self.z[i]=self.s(self.A[i]*self.z[i-1]-self.b[i])
    def s(self,x):
        n = len(x)
        y = zeros(n,1)
        for i in range(n):
            y[i] = self.sigmoid.s(x[i])
        return np.matrix(y)
    def Ds(self,x):
        n = len(x)
        y = np.zeros(n,1)
        for i in range(n):
            y[i] = self.sigmoid.Ds(x[i])
        return np.matrix(np.diag(y))
    def dz_j_dz_j_minus_1(self,j):
        return self.Ds(self.A[j]*self.z[j-1]-self.b[j])*self.A[j]
    def dz_j_dvec_A_j(self,j):
        return self.Ds(self.A[j]*self.z[j-1]-self.b[j])\
            * np.kron( eye(self.m[j]), self.z[j-1].T )
    def dz_j_db_j(self,j):
        return -self.Ds(self.A[j]*self.z[j-1]-self.b[j])

def test():
    x = randn(2,100)
    y = randn(3,100)
    m = (2,2,y.shape[0])
    nn(x,y,m)