import numpy as np
from foo import eye, vec, mat, zeros, ones, rand, randn, sprand, sprandn
from poly import poly
from QP import QP
import norm
from scipy.sparse import csr_matrix as sparse
import matplotlib.pyplot as plt
import random
import math

class spinv(object):
    """
        for A ∈ R^{m,n},
            if m > n and rank(A) = n
                spinv_A * A ≈ I_n
            if n > m and rank(A) = m
                spinv_A * A ≈ I_m
    """
    def __init__(self,A, λ = 0.1):
            m = A.shape[0]
            n = A.shape[1]
            x = norm.linear_l_p_reg_q( np.kron(eye(n),A.T), vec(eye(n)), λ, 2, 1).solve()
            mu = np.mean(x)
            for i in range(len(x)):
                if abs(x[i]) < mu:
                    x[i]=0
            self.pinv_A = mat(x,n,m)
            self.spinv_A = sparse(self.pinv_A)
    def mat(self):
        return self.pinv_A
    def sp(self):
        return self.spinv_A
    @staticmethod
    def test():
        A = np.asmatrix(np.random.rand(30,3))
        spinv_A = spinv( A, 0.1 ).sp()
        print( '\nSPINV.test()\n'+'_'*80 )
        print( 'A = \n'+ str(A) + '\n')
        print( 'spinv_A = \n'+ str(spinv_A) + '\n')
        print( 'spinv_A * A = \n'+ str(spinv_A*A) + '\n')

class interset(object):
    """
        min     ||x_1-x_2||_q
        x_1,x_2

        s.t.    x_1 ∈ POLYHEDRON_1
                x_2 ∈ POLYHEDRON_2
    """
    def __init__(self, P1, P2, q):
        n = P1.n
        assert P2.n is n
        A = np.bmat([[eye(n),-eye(n)]])
        b = zeros(n,1)
        P = poly.diag(P1,P2)
        if q is 1:
            self.problem = norm.linear_l_1(A,b,P)
        elif q is 2:
            self.problem = norm.linear_l_2(A,b,P)
        elif q is 'inf':
            self.problem = norm.linear_l_inf(A,b,P)
        else:
            assert False, "q must be 1, 2, or \'inf\'"
        self.P1 = P1
        self.P2 = P2 
        self.P = P
        self.n = n
        self.q = q
        self.sol = None
    def solve(self):
        x = self.problem.solve()
        x1 = x[0:self.n]
        x2 = x[self.n:2*2]
        x1_min = zeros(self.n,1)
        x2_min = zeros(self.n,1)
        for i in range(self.n):
            x1_min[i] = x1[i]
            x2_min[i] = x2[i]
        d_min = norm.l_p(x1_min-x2_min,self.q)
        self.sol = (d_min,x1,x2)
        return self.sol
    def __repr__(self):
        s = '\n'+'_'*80 + '\n'+self.__class__.__name__+'\n' + self.__doc__ 
        if self.sol is not None:
            s += '\n d\t= ' + str(self.sol[0])
            s += '\n x1\t= ' + str(self.sol[1])
            s += '\n x2\t= ' + str(self.sol[2])
        return s
    @staticmethod
    def test():
        P1 = poly.rand(2,4,0,1).affine_transform(eye(2),2*ones(2,1))
        P2 = poly.rand(2,4,0,1).affine_transform(eye(2),-2*ones(2,1))
        problem = interset(P1,P2,2)
        problem.solve()
        print(problem)

class k_means(object):
    def __init__(self,s,k):
        # s = vectors
        # x = means
        m = s.shape[1]
        n = s.shape[0]# s[i] = column vector as matrix in R^n
        x = randn(n,k)
        for loop in range(10):
            d = zeros(k,m)
            for j in range(m):
                for i in range(k):
                        d[i,j] = norm.l_p(s[:,j] - x[:,i], 2)
            min_d = np.full((m), np.inf)
            for j in range(m):
                for i in range(k):
                    if d[i,j] < min_d[j]:
                        min_d[j] = d[i,j]
            b = zeros(k,m)
            for j in range(m):
                for i in range(k):
                    if d[i,j] == min_d[j]:
                        b[i,j] = 1
                    else:
                        b[i,j] = 0
            for i in range(k):
                num = zeros(n,1)
                den = 0
                for j in range(m):
                    num += b[i,j]*s[:,j]
                    den += b[i,j]
                if den != 0:
                    x[:,i] = num / den
                else:
                    x[:,i] += random.gauss(0,1) # random walk
        self.s = s
        self.x = x
    def get_centers(self):
        n = self.x.shape[0]
        m = self.x.shape[1]
        x= []
        for i in range(m):
            x.append(self.x[:,i])
        return x
    def plot(self):
        if self.s.shape[0] == 2:
            plt.clf()
            s = np.asarray(self.s)
            plt.plot(s[0],s[1],'.',color="blue")
            x = np.asarray(self.x)
            plt.plot(x[0],x[1],'o',color="orange",markersize = 10)
            plt.savefig('./fig/k_means.png')
        else:
            plt.clf()
            plt.title('n>2')
            plt.savefig('./fig/k_means.png')
    @staticmethod
    def test():
        print('\n'+'_'*80 + '\nk_means [see ./fig/k_means.png]')
        import random
        n = 2
        m = 20
        k = 4
        s = randn(n,m)+np.kron(ones(1,m),10*randn(n,1))
        for i in range(k-1):
            s = np.hstack((s,randn(n,m)+np.kron(ones(1,m),10*randn(n,1))))
        k_means(s,k).plot()

class poly_cells(object):
    """
        compute POLYHEDRON_j polyhedrons for j in 0 ... k-1
    """
    def __init__(self,x):
        n = x.shape[0]
        m = x.shape[1]
        self.x = x

        P_list = []
        for j in range(m):
            A = zeros(m-1,n)
            b = zeros(m-1,1)
            i_ = 0
            for i in range(m):
                if i != j:
                    A[i_,:] = 2*(x[:,i].T-x[:,j].T)
                    b[i_] = x[:,i].T*x[:,i] - x[:,j].T*x[:,j]
                    i_ += 1
            P_list.append(poly(A,b))
        self.P_list = P_list
    def plot_poly_cells(self):
        plt.figure(1)
        plt.clf()
        from scipy.spatial import Voronoi, voronoi_plot_2d
        x = np.asarray(self.x.T)
        vor = Voronoi(x)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',\
            line_width=2, line_alpha=0.6, point_size=20 )
        plt.title('Voronoi')
        plt.savefig('./fig/Voronoi.png')
    def plot_poly_center(self):
        m = self.x.shape[1]
        n = self.x.shape[0]
        x_ = zeros(n,m)
        for i in range(m):
            x_[:,i] = poly_cells.poly_center(self.P_list[i])
        x = np.asarray(self.x)
        x_ = np.asarray(x_)
        plt.figure(1)
        plt.clf()
        plt.plot(x[0],x[1],'.')
        plt.plot(x_[0],x_[1],'o',markersize = 20,fillstyle = 'none')
        plt.title('poly_center')
        plt.savefig('./fig/Voronoi_poly_center.png')
    @staticmethod
    def poly_center(P):# poly(A,b) -> x_center
        m = P.A_ub.shape[0]
        A = P.A_ub
        assert np.linalg.matrix_rank(A) == A.shape[1], "non-unique solution"
        b = P.b_ub
        c = zeros(m)
        for i in range(A.shape[0]):
            c[i,0] = A[i,:]*A[i,:].T
        c /= 4.0
        return np.linalg.pinv(A)*(b-c)
    def test():
        print('\n'+'_'*80 + '\npoly_cells [see ./fig/Voronoi.png]')
        n = 2
        m = 100
        x = randn(n,m)
        obj = poly_cells(x)
        obj.plot_poly_cells()
        obj.plot_poly_center()

class linear_fractional(object):
    """
        min    J = (a.T*x-b)/(c.T*x-d)
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """
    def __init__(self,a,b,c,d,P):
        n = P.n 
        m = P.A_ub.shape[0]
        r = P.A_eq.shape[0]
        A_ub = np.bmat([[P.A_ub,-P.b_ub],[zeros(1,n),-ones(1)]])
        b_ub = zeros(m+1)
        A_eq = np.bmat([[P.A_eq,-P.b_eq],[c.T,-d]])
        b_eq = np.bmat([[zeros(r)],[ones(1)]])
        P = poly(A_ub,b_ub,A_eq,b_eq)
        Q = None
        c = np.bmat([[a],[b]])
        self.QP = QP(None,c,P)
        self.n = n
    def __repr__(self):
        s = '\n'+'_'*80 + '\n'+self.__class__.__name__+'\n' + self.__doc__ 
        if self.sol is not None:
            s += '\n x\t= ' + str(self.sol)
        return s
    def solve(self):
        x = self.QP.solve()
        print('*'*80)
        print(x)
        print('*'*80)
        if x is not None:
            if x[-1,0] != 0:
                self.sol = x[0:self.n,0]/x[-1,0]
            else:
                self.sol = None
        else:
            self.sol = None
        return self.sol
    @staticmethod
    def test():
        n = 2
        a = randn(n)
        b = randn(1)
        c = randn(n)
        d = randn(1)
        P = poly.rand(n,1,1)
        problem = linear_fractional(a,b,c,d,P)
        problem.solve()
        print(problem)

class E_l_2(norm.linear_l_p):
    """
        min    J = E|| (A+A_rand)(x+x_rand) - (b+b_rand) ||_{2,W}^2
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """
    def __init__(self,A,b,
        EA = None, Ex = None, Eb = None,\
        EAx = None, EAoA = None, EAox = None, EAoAx = None, EAob = None,\
        W = None, P = None):
        """
            EA      = E( A_rand )
            Ex      = E( x_rand )
            Eb      = E( b_rand )
            EAx     = E( A_rand * x_rand )
            EAoA    = E( kron(A_rand, A_rand) )
            EAox    = E( kron(A_ranc, x_rand) )
            EAoAx   = E( kron(A_rand, A_randx_rand) )
            EAob    = E( kron(A_rand, b_rand) )
        """
        m = A.shape[0]
        n = A.shape[1]
        # ----------------------------------------------------------------------
        def init_E(EA, Ex, Eb, EAx, EAoA, EAox, EAoAx, EAob, W, m ,n):
            if W is None:
                M = eye(m)
            else:
                M = W.T * W            
            if EA is None:
                EA = zeros(m,n)
            if Ex is None:
                Ex = zeros(n,1)
            if Eb is None:
                Eb = zeros(m,1)
            if EAx is None:
                EAx = zeros(m,1)
            if EAoA is None:
                EAoA = zeros(m*m,n*n)
            if EAox is None:
                EAox = zeros(m*n,n*1)
            if EAoAx is None:
                EAoAx = zeros(m*m,n*1)
            if EAob is None:
                EAob = zeros(m*m,n*1)
            return (EA, Ex, Eb, EAx, EAoA, EAox, EAoAx, EAob, M)
        EA, Ex, Eb, EAx, EAoA, EAox, EAoAx, EAob, M = init_E(\
            EA, Ex, Eb, EAx, EAoA, EAox, EAoAx, EAob, W, m ,n)
        # ----------------------------------------------------------------------
        Q = A.T*M*A + A.T*M*EA + EA.T*M*A
        # Q += E( A_rand.T * M * A_rand )
        temp = vec(M).T * EAoA 
        Q += mat(temp.T,n,n)
        Q *= 2
        # ----------------------------------------------------------------------
        c = A.T*M*A*Ex + A.T*M*EAx - A.T*M*b - EA.T*M*b - A.T*M*Eb        
        # c += E( A_rand.T * M * A * x_rand )
        temp = vec(M*A).T * EAox 
        c += mat(temp.T,n,1)
        # c += E( A_rand.T * M * A_rand * x_rand )
        temp = vec(M).T * EAoAx
        c += mat(temp.T,n,1)
        # c -= E( A_rand.T * M * b_rand )
        temp = vec(M).T * EAob
        c -= mat(temp.T,n,1)
        c *=2
        # ----------------------------------------------------------------------
        A_ub = None
        b_ub = None
        self.build(n,Q,c,A_ub,b_ub,P)
    @staticmethod
    def test():
        N = 1_000
        m = 3
        n = 3
        A = np.asmatrix(np.random.rand(m,n))
        b = np.asmatrix(np.random.rand(m,1))
        W = np.matrix(np.random.randn(m,m))
        # ----------------------------------------------------------------------
        A_=[]
        b_=[]
        x_=[]
        for i in range(N):
            A_rand = np.matrix(np.random.randn(m,n))
            b_rand = np.matrix(np.random.randn(m,1))
            x_rand = np.matrix(np.random.randn(n,1))
            A_.append(A_rand)
            b_.append(b_rand)
            x_.append(x_rand)
        # ----------------------------------------------------------------------
        EA      = zeros(m,n)
        Ex      = zeros(n,1)
        Eb      = zeros(m,1)
        EAx     = zeros(m,1)
        EAoA    = zeros(m*m,n*n)
        EAox    = zeros(m*n,n*1)
        EAoAx   = zeros(m*m,n*1)
        EAob    = zeros(m*m,n*1)
        for i in range(N):
            EA      += A_[i]
            Ex      += x_[i]
            Eb      += b_[i]
            EAx     += A_[i]*x_[i]
            EAoA    += np.kron(A_[i],A_[i])
            EAox    += np.kron(A_[i],x_[i])
            EAoAx   += np.kron(A_[i],A_[i]*x_[i])
            EAob    += np.kron(A_[i],b_[i])
        EA      /= N
        Ex      /= N
        Eb      /= N 
        EAx     /= N
        EAoA    /= N
        EAox    /= N
        EAoAx   /= N
        EAob    /= N
        problem = E_l_2(A,b, EA, Ex, Eb, EAx, EAoA, EAox, EAoAx, EAob, W)
        print(problem)
        print(problem.solve())

class sampled_convex_set(object):
    def __init__(self,s):
        """
            x ∈ R^n
        """
        s = sampled_convex_set.remove_interior_points(s)
        n = s.shape[0]
        m = s.shape[1]
        A_ub = np.bmat([[zeros(m,n),-eye(m)],[zeros(m,n),eye(m)]])
        b_ub = np.bmat([[zeros(m)],[ones(m)]])
        A_eq = np.bmat([[-eye(n),s],[zeros(1,n),ones(1,m)]])
        b_eq = np.bmat([[zeros(n)],[ones(1)]])
        self.P = poly(A_ub,b_ub,A_eq,b_eq)

    @staticmethod
    def remove_interior_points(s):
        return s
    @staticmethod
    def test():
        n = 2
        s = randn(n,100)
        sampled_convex_set(s)

class min_complexity_model(object):
    """
        y_i = A_sparse * x_i: R^n 🠊 R^m

        vec(A_sparse) ∈ POLYHEDRON
    """
    def __init__(self,x,y,λ=0.1,W=None):
        # W = list of matricies
        n = x.shape[0]
        m = y.shape[0]
        q = x.shape[1]
        assert y.shape[1] == q
        Q = zeros(m*n,m*n)
        c = zeros(m*n)
        for i in range(q):
            if W is None:
                Q += np.kron(eye(m), x[:,i] * x[:,i].T)
                c += np.kron(eye(m), x[:,i]) *y[:,i]
            else:
                Q += np.kron(eye(m), x[:,i] * W[i].T * W[i]* x[:,i].T)
                c += np.kron(eye(m), x[:,i]) * W[i].T * W[i] * y[:,i]
        Q *=  2
        c *= -2
        vec_A = norm.linear_QP_reg_l_1( Q, c ).solve()
        A = mat( vec_A, m, n )
        t = np.mean(np.abs(vec_A))
        for i in range(m):
            for j in range(n):
                if abs(A[i,j]) < t:
                    A[i,j] = 0
        self.A = sparse(A)
    def __repr__(self):
        s = '\n'+'_'*80 + '\n'+self.__class__.__name__+'\n' + self.__doc__ 
        s +=  '\nA = \n' + str(self.A)
        return s
        
    @staticmethod 
    def test():
        n = 5
        m = 10
        q = 100
        x = randn(n,q)
        y = zeros(m,q)
        while True:
            A,r = sprandn(m,n,0.1)
            if r < m*n:
                break
        for i in range(q):
            y[:,i] = A * x[:,i]
        print(min_complexity_model(x,y,1))
        print('\nA_data = \n' + str(sparse(A))+'\n')

def test():
    spinv.test()
    E_l_2.test()
    interset.test()
    k_means.test()
    poly_cells.test()
    linear_fractional.test()
    sampled_convex_set.test()
    min_complexity_model.test()