import numpy as np
from foo import sign, zeros
import norm

class non_convex_reg(object):
    """
        min    J = || A x - b ||_(1/q) + ||x||_q,    0 < q < 1
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """
    def __init__(self,A,b,q=0.1,P=None):
        assert 0 < q and q < 1
        assert A.shape[0] is b.shape[0] and b.shape[1] is 1
        self.A = A
        self.b = b
        self.q = q 
    
    def cost(self,x):
        n = len(x)
        assert self.A.shape[1] is n
        J = 0
        for i in range(n):
            temp = self.A[i,:]*x-self.b[i]
            temp = temp[0,0]
            J += pow(abs(temp),1/float(self.q))
        J = pow(J,self.q)
        J_reg = 0
        for i in range(n):
            J_reg += pow(abs(x[i,0]),self.q)
        J_reg = pow(J_reg,1/float(self.q))
        J += J_reg
        return J
    
    def grad(self,x):
        n = len(x)
        assert self.A.shape[1] is n
        temp = np.zeros((n,1))
        for i in range(n):
            temp_i = self.A[i,:]*x-self.b[i]
            temp[i] = temp_i[0,0]
        J = 0
        for i in range(n):
            J += pow(abs(temp[i]),1/float(self.q))
        J = pow(J,self.q-1)
        J_reg = 0
        for i in range(n):
            J_reg += pow(abs(x[i,0]),self.q)
        J_reg = pow(J_reg,1/float(self.q)-1)
        DJ = np.asmatrix(np.zeros((1,n)))# row
        for j in range(n):
            t1 = 0
            for i in range(n):
                t1 += pow(abs(temp[i]),1/float(self.q)-1)*sign(temp[i])*self.A[i,j]            
            t2 = pow(abs(x[j,0]),self.q-1)*sign(x[j,0])
            DJ[0,j] = J*t1+J_reg*t2
        return DJ

    @staticmethod
    def test():
        print(non_convex_reg.__doc__)
        m = 3
        n = 3
        A = np.asmatrix(np.random.rand(m,n))
        b = np.asmatrix(np.random.rand(m,1))
        x = np.asmatrix(np.random.rand(n,1))
        t = non_convex_reg(A,b)
        print('`'*80)
        print(t.cost(x))
        print(t.grad(x))
        print('`'*80)

# class QP_loop(object): # TODO:

#     def __init__(self,n):
#         self.n = n 

#     def init_x0(self,P):

#         # compute x0 in P
#         x0 = P.get_feasible_x()

#         if x0 is None:
#             print("P is infeasible")
#         else:
#             self.x0 = x0

#     def loop(self,x):

#         while True:
            
#             # get A0, b0

#             problem = None

def test():
    non_convex_reg.test()