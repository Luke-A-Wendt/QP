from QP import QP
from poly import poly
import numpy as np
from foo import eye, ones, zeros, vec, mat, sign
from svdr import svdr
from scipy import linalg
from svdr import svdr
import math

def l_p(x,p):
    assert isinstance(x,np.matrix)
    m = x.shape[0]
    n = x.shape[1]
    y = 0
    if p is 1:
        for i in range(m):
            for j in range(n):
                y += abs(x[i,j])
    if p is 2:
        for i in range(m):
            for j in range(n):
                y += pow(x[i,j],2)# returns squared
        y = math.sqrt(y)
    if p is 'inf':
        for i in range(m):
            for j in range(n):
                if x[i,j] > y:
                    for j in range(n):
                        y = x[i,j]
    return y

class linear_l_p(object):
    """
        min    J = || A x - b ||_p
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """
    def __init__( self ):
        self.QP = QP()
        self.n  = None
    def build(self,n,Q,c,A_ub,b_ub,P):
        if A_ub is not None:
            if P is not None:
                P_ = poly(A_ub,b_ub)
                P  = P.add(P_)
            else:
                P = poly(A_ub,b_ub)
        self.QP = QP(Q,c,P)
        self.n = n
    def solve(self):
        x = self.QP.solve()
        if x is None:
            return None
        else:
            return x[0:self.n,0]
    def __repr__(self):
        return "\n"+'_'*80+"\n"+self.__doc__
    @staticmethod
    def weight(A,b,W):
        WA = W*A
        Wb * W*b
        return (WA,Wb)    

class linear_l_2(linear_l_p):
    """
        min    J = || A x - b ||_2^2
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """    
    def __init__(self, A, b, P = None):
        n = A.shape[1]
        Q =  2 * A.T * A
        c = -2 * A.T * b
        A_ub = None
        b_ub = None
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_l_1(linear_l_p):
    """
        min    J = || A x - b ||_1
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """       
    def __init__(self, A, b, P = None):
        n = A.shape[1]
        m = A.shape[0]
        Q = None
        c = np.bmat([[zeros(n)],[ones(m)]])
        A_ub = np.bmat([[A,-eye(m)],[-A,-eye(m)]])
        b_ub = np.bmat([[b],[-b]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_l_inf(linear_l_p):
    """
        min    J = || A x - b ||_inf
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """   
    def __init__(self, A, b, P = None):
        n = A.shape[1]
        m = A.shape[0]
        Q = None
        c = np.bmat([[zeros(n)],[ones(1)]])
        A_ub = np.bmat([[A,-ones(m)],[-A,-ones(m)]])
        b_ub = np.bmat([[b],[-b]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_l_1_1(linear_l_p):
    """
        min    J = || A x - b ||_1 + || A_aux x - b_aux ||_1
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """       
    def __init__(self, A, b, A_aux, b_aux, P = None):
        n = A.shape[1]
        m = A.shape[0]
        r = A_aux.shape[0]
        Q = None
        c = np.bmat([[zeros(n)],[ones(m)],[ones(r)]])
        A_ub = np.bmat([[A,-eye(m),zeros(m,r)],[A,-eye(m),zeros(m,r)],\
            [-A_aux,zeros(r,m),-eye(r)],[A_aux,zeros(r,m),-eye(r)]])
        b_ub = np.bmat([[-b],[b],[-b_aux],[b_aux]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_l_2_2(linear_l_p):
    """
        min    J = || A x - b ||_2^2 + || A_aux x - b_aux ||_2^2
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """       
    def __init__(self, A, b, A_aux, b_aux, P = None):
        n = A.shape[1]
        m = A.shape[0]
        r = A_aux.shape[0]
        Q =  2*(A.T*A+A_aux.T*A_aux)
        c = -2*(A.T*b+A_aux.T*b_aux)
        A_ub = None
        b_ub = None
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_l_inf_inf(linear_l_p):
    """
        min    J = || A x - b ||_inf + || A_aux x - b_aux ||_inf
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """       
    def __init__(self, A, b, A_aux, b_aux, P = None):
        n = A.shape[1]
        m = A.shape[0]
        r = A_aux.shape[0]
        Q = None
        c = np.bmat([[zeros(n)],[zeros(2)]])
        A_ub = np.bmat([[A,-ones(m),zeros(m)],[A,-ones(m),zeros(m)],\
            [-A_aux,zeros(r),-ones(r)],[A_aux,zeros(r),-ones(r)]])
        b_ub = np.bmat([[-b],[b],[-b_aux],[b_aux]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_l_1_2(linear_l_p):
    """
        min    J = || A_1 x - b_1 ||_1 + || A_2 x - b_2 ||_2^2
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """       
    def __init__(self, A_1, b_1, A_2, b_2, P = None):
        n = A_1.shape[1]
        m = A_1.shape[0]
        Q = np.bmat([[2*A_2.T*A_2,zeros(n,m)],[zeros(m,n),zeros(m,m)]])
        c = np.bmat([[-2*A_2.T*b_2],[ones(m)]])
        A_ub = np.bmat([[-A_1,-eye(m)],[A_1,-eye(m)]])
        b_ub = np.bmat([[-b_1],[b_1]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_QP_QP(linear_l_p): 
    """
        min    J = (1/2) x.T Q_1 x + c_1.T x + (1/2) x.T Q_2 x + c_2.T x
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """       
    def __init__(self, Q_1, c_1, Q_2, c_2, P = None):
        n = Q_1.shape[0]
        assert Q_2.shape[0] == n
        Q = Q_1 + Q_2
        c = c_1 + c_2
        A_ub = None
        b_ub = None 
        self.build( n, Q, c, A_ub, b_ub, P )

class linear_QP_l_1(linear_l_p): 
    """
        min    J = (1/2) x.T Q x + c.T x + || A_1 x - b_1 ||_1
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """       
    def __init__(self, Q, c, A_1, b_1, P = None):
        n = A_1.shape[1]
        m = Q.shape[0]
        Q = np.bmat([[Q,zeros(n,m)],[zeros(m,n),zeros(m,m)]])
        c = np.bmat([[c],[ones(m)]])
        A_ub = np.bmat([[-A_1,-eye(m)],[A_1,-eye(m)]])
        b_ub = np.bmat([[-b_1],[b_1]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_QP_l_inf(linear_l_p): 
    """
        min    J = (1/2) x.T Q x + c.T x + || A_inf x - b_inf ||_inf
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """    
    def __init__(self, Q, c, A_inf, b_inf, P = None):   
        n = A_inf.shape[1]
        m = A_inf.shape[0]
        assert n is Q.shape[1]
        Q = np.bmat([[Q,zeros(n)],[zeros(1,n),zeros(1)]])
        c = np.bmat([[c],[ones(1)]])
        A_ub = np.bmat([[-A_inf,-ones(m)],[A_inf,-ones(m)]])
        b_ub = np.bmat([[-b_inf],[b_inf]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_QP_reg_l_inf(linear_l_p): 
    """
        min    J = (1/2) x.T Q x + c.T x + λ || x ||_inf
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """    
    def __init__(self, Q, c, λ = 0.1, P = None):   
        n = Q.shape[0]
        Q = np.bmat([[Q,zeros(n)],[zeros(1,n),zeros(1)]])
        c = np.bmat([[c],[ λ * ones(1)]])
        A_ub = np.bmat([[-eye(n),-ones(n)],[eye(n),-ones(n)]])
        b_ub = np.bmat([[-zeros(n)],[zeros(n)]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_QP_reg_l_1(linear_l_p): 
    """
        min    J = (1/2) x.T Q x + c.T x + λ || x ||_1
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """       
    def __init__(self, Q, c, λ = 0.1, P = None):
        n = Q.shape[0]
        Q = np.bmat([[Q,zeros(n,n)],[zeros(n,n),zeros(n,n)]])
        c = np.bmat([[c],[ λ * ones(n)]])
        A_ub = np.bmat([[-eye(n),-eye(n)],[eye(n),-eye(n)]])
        b_ub = np.bmat([[-zeros(n)],[zeros(n)]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_l_1_inf(linear_l_p):
    """
        min    J = || A_1 x - b_1 ||_1 + || A_inf x - b_inf ||_inf
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """    
    def __init__(self,A_1, b_1, A_inf, b_inf, P = None):
        n = A_1.shape[1]
        m_1 = A_1.shape[0]
        m_inf = A_inf.shape[0]
        assert n is A_inf.shape[1]
        Q = None
        c = np.bmat([[zeros(n)],[ones(m_1)],[ones(1)]])
        A_ub = np.bmat([[-A_1,-eye(m_1),zeros(m_1)],\
                        [A_1,-eye(m_1),zeros(m_1)],\
                        [-A_inf,zeros(m_inf,m_1),-ones(m_inf)],\
                        [A_inf,zeros(m_inf,m_1),-ones(m_inf)]])
        b_ub = np.bmat([[-b_1],[b_1],[-b_inf],[b_inf]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_l_2_inf(linear_l_p):
    """
        min    J = || A_2 x - b_2 ||_2^2 + || A_inf x - b_inf ||_inf
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """    
    def __init__(self, A_2, b_2, A_inf, b_inf, P = None):   
        n = A_inf.shape[1]
        m = A_inf.shape[0]
        assert n is A_2.shape[1]
        Q = np.bmat([[2*A_2.T*A_2,zeros(n)],[zeros(1,n),zeros(1)]])
        c = np.bmat([[-2*A_2.T*b_2],[ones(1)]])
        A_ub = np.bmat([[-A_inf,-ones(m)],[A_inf,-ones(m)]])
        b_ub = np.bmat([[-b_inf],[b_inf]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_l_1_2_inf(linear_l_p):
    """
        min    J = || A_1 x - b_1 ||_1 + || A_2 x - b_2 ||_2^2 
                                        + || A_inf x - b_inf ||_inf
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """    
    def __init__(self, A_1, b_1, A_2, b_2, A_inf, b_inf, P = None):
        n = A_2.shape[1]
        m_1 = A_1.shape[0]
        m_inf = A_inf.shape[0]
        Q = np.bmat([[2*A_2.T*A_2,zeros(n,m_1),zeros(n)],\
            [zeros(m_1,n),zeros(m_1,m_1),zeros(m_1)],\
                [zeros(1,n),zeros(1,m_1),zeros(1)]])
        c = np.bmat([[-2*A_2.T*b_2],[ones(m_1)],[ones(1)]])
        A_ub = np.bmat([[A_1,-eye(m_1),zeros(m_1)],\
            [A_1,-eye(m_1),zeros(m_1)],\
                [-A_inf,zeros(m_inf,m_1),-ones(m_inf)],\
                    [A_inf,zeros(m_inf,m_1),-ones(m_inf)]])
        b_ub = np.bmat([[-b_1],[b_1],[-b_inf],[b_inf]])
        self.build(n,Q,c,A_ub,b_ub,P)

class linear_l_p_q(linear_l_p):# general interface
    """
        min    J = || A_p x - b_p ||_p + || A_q x - b_q ||_q
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """       
    def __init__(self, A_p, b_p, A_q, b_q, p, q, P = None):
        assert A_p.shape[1] is A_q.shape[1]
        assert A_p.shape[0] is b_p.shape[0]
        assert A_q.shape[0] is b_q.shape[0]
        self.pq = (p,q)
        if p is 1:
            if q is 1:
                self.problem = linear_l_1_1(A_p, b_p, A_q, b_q, P)
            if q is 2:
                self.problem = linear_l_1_2(A_p, b_p, A_q, b_q, P)
            if q is 'inf':
                self.problem = linear_l_1_inf(A_p, b_p, A_q, b_q, P)
        if p is 2:
            if q is 1:
                self.problem = linear_l_1_2(A_q, b_q, A_p, b_p, P)
            if q is 2:
                self.problem = linear_l_2_2(A_p, b_p, A_q, b_q, P)
            if q is 'inf':
                self.problem = linear_l_2_inf(A_p, b_p, A_q, b_q, P)
        if p is 'inf':
            if q is 1:
                self.problem = linear_l_1_2(A_q, b_q, A_p, b_p, P)
            if q is 2:
                self.problem = linear_l_1_2(A_q, b_q, A_p, b_p, P)
            if q is 'inf':
                self.problem = linear_l_inf_inf(A_p, b_p, A_q, b_q, P)
    def solve(self):
        return self.problem.solve()
    def __repr__(self):
        return "\n"+'_'*80+"\n"+self.__doc__ + "\n (p,q) = "+str(self.pq)

class linear_l_p_reg_q(linear_l_p):
    """
        min    J = || A x - b ||_p + λ ||x||_q
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """       
    def __init__(self, A, b, λ, p, q, P = None):        
        assert λ > 0
        n = A.shape[1]
        A_reg = λ*eye(n)
        b_reg = zeros(n)        
        self.pq = (p,q)
        self.problem = linear_l_p_q( A, b, A_reg, b_reg, p, q, P )
    def solve(self):
        return self.problem.solve()
    def __repr__(self):
        return "\n"+'_'*80+"\n"+self.__doc__ + "\n (p,q) = "+str(self.pq)

class DeadZone_vec(linear_l_p):
    """
        min    J = ∑_{i=0}^{m-1} f_i( a_i^T x - b_i ),  f_i = DeadZone(t_i), t ∈ R^m
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """
    def __init__(self, A, b, t, P = None):
        # TODO: t as vec
        n = A.shape[1]
        m = A.shape[0]
        assert t.shape[0] is m and t.shape[1] is 1
        Q = None
        c = np.bmat([[zeros(n)],[ones(m)]])
        A_ub = np.bmat([[A,-eye(m)],[-A,eye(m)],[zeros(m,n),-eye(m)]])
        b_ub = np.bmat([[t+b],[t-b],[zeros(m)]])
        self.build(n,Q,c,A_ub,b_ub,P)

class DeadZone(linear_l_p):
    """
        min    J = ∑_{i=0}^{m-1} f( a_i^T x - b_i ),  f = DeadZone(t), t > 0
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """
    def __init__(self, A, b, t = 1, P = None):
        assert t > 0
        self.t = t
        m = A.shape[0]
        self.problem = DeadZone_vec(A,b,t*ones(m),P)
    def solve(self):
        return self.problem.solve()
    def f(self,u):
        if u <= self.t:
            return pow(u,2)
        else:
            return t*(2*abs(u)-self.t)
    def Df(self,u):
        if u <= self.t:
            return 2*u
        else:
            return 2*t*sign(i)

class Huber_vec(linear_l_p):
    """
        min    J = ∑_{i=0}^{m-1} f_i( a_i^T x - b_i ),  f_i = Huber(t_i)
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """
    def __init__(self, A, b, t, P = None):
        m = A.shape[0]
        n = A.shape[1]
        assert t.shape[0] is m and t.shape[1] is 1
        Q = np.bmat([[zeros(n,n),zeros(n,m),zeros(n,m)],\
            [zeros(m,n),2*eye(m),zeros(m,m)],\
                [zeros(m,n),zeros(m,m),zeros(m,m)]])
        c = np.bmat([[zeros(n)],[zeros(m)],[2*t]])
        A_ub = np.bmat([[A,-eye(m),-eye(m)],\
            [-A,-eye(m),-eye(m)],[zeros(m,n),eye(m),zeros(m,m)],\
                [zeros(m,n),-eye(m),zeros(m,m)],\
                    [zeros(m,n),zeros(m,m),-eye(m)]])
        b_ub = np.bmat([[b],[-b],[t],[zeros(m)],[zeros(m)]])
        self.build(n,Q,c,A_ub,b_ub,P)

class Huber(linear_l_p):
    """
        min    J = ∑_{i=0}^{m-1} f( a_i^T x - b_i ),  f = Huber(t)
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """
    def __init__(self, A, b, t = 1, P = None):
        assert t > 0
        self.t = t
        m = A.shape[0]
        self.problem = Huber_vec(A,b,t*ones(m),P)
    def solve(self):
        return self.problem.solve()
    def f(self,u):
        if u <= self.t:
            return 0
        else:
            return abs(u)-self.t
    def Df(self,u):
        if u <= self.t:
            return 0
        else:
            return sign(u)

class ReLu(linear_l_p):
    """
        min    J = ∑_{i=0}^{m-1} f( a_i^T x - b_i ),  f = ReLu()
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """    
    def __init__(self, A, b, P = None ):
        m = A.shape[0]
        n = A.shape[1]
        Q = None
        c = np.bmat([[zeros(n)],[ones(m)]])
        A_ub = np.bmat([[A,-eye(m)],[zeros(m,n),-eye(m)]])
        b_ub = np.bmat( [[b],[zeros(m)]] )
        self.build(n,Q,c,A_ub,b_ub,P)

class LeakyReLu(linear_l_p):
    """
        min    J = ∑_{i=0}^{m-1} f( a_i^T x - b_i ),  f = LeakyReLu(t)
        x ∈ R^n
                s.t.  x ∈ POLYHEDRON
    """    
    def __init__(self, A, b, t, P = None ):
        assert 0 <= t and t <= 1
        m = A.shape[0]
        n = A.shape[1]
        Q = None
        c = np.bmat([[zeros(n)],[ones(m)]])
        A_ub = np.bmat([[A,-eye(m)],[A,-eye(m)*(1/t)]])
        b_ub = np.bmat( [[b],[b]] )
        self.build(n,Q,c,A_ub,b_ub,P)

class QP_to_linear_l_2(object):
    def __init__(self,Q,c):
        """
            ||A x - b||_2^2 = x.T * A.T * A *x - 2 * b.T * A *x + b.T * b
                            = (1/2) x.T * Q * x + c.T * x + r

            Q   = 2*A.T*A
                = U * S * U.T   => A    = S^0.5 * U.T / sqrt(2)
                                => A^-1 = U  * S-^0.5 * sqrt(2)

            c = -2*A.T*b        => b = -inv(A.T)*c/2.0

        """
        Q = svdr(Q)
        assert Q.r == Q.m and Q.r == Q.n
        A =  np.diag(np.sqrt(Q.s)) * Q.U_range().T / math.sqrt(2)
        b = -svdr( A.T ).pinv() * c / 2.0
        self.A = A
        self.b = b
    def solve(self):
        return (self.A,self.b)
    @staticmethod
    def test( n = 2 ):
        def rand_pos_def_matrix(n):
            Q = np.asmatrix(np.random.rand(n,n))
            return np.asmatrix(Q.T+Q+eye(n))
        Q = rand_pos_def_matrix(n)
        c = np.asmatrix(np.random.rand(n,1))        
        A,b = QP_to_linear_l_2(Q,c).solve()
        print( "_"*80+"\nQP_to_linear_l_2 error = " +\
            str(np.linalg.norm(Q/2.0  - A.T * A )\
                + np.linalg.norm( c + 2 * A.T * b ))+'\n'+"_"*80)

def test():

    n       = 3

    m       = n-1
    m_ub    = n-1
    m_eq    = n-1
    m_1     = n
    m_2     = n
    m_inf   = n

    A       = np.asmatrix(np.random.rand(m,n))
    b       = np.asmatrix(np.random.rand(m,1))
    A_1     = np.asmatrix(np.random.rand(m_1,n))
    b_1     = np.asmatrix(np.random.rand(m_1,1))
    A_2     = np.asmatrix(np.random.rand(m_2,n))
    b_2     = np.asmatrix(np.random.rand(m_2,1))
    A_inf   = np.asmatrix(np.random.rand(m_inf,n))
    b_inf   = np.asmatrix(np.random.rand(m_inf,1))

    P       = poly.rand(n, m_ub, m_eq)
    print(P.full_rank_inverse(A,b))
    print(P.full_rank_right_pseudo_inverse(A,b))
    print(P.full_rank_left_pseudo_inverse(A,b))
    print( 'x_feasible = ' + str(P.get_feasible()) )

    problems = []
    problems.append(linear_l_2(A,b,P))
    problems.append(linear_l_1(A,b,P))
    problems.append(linear_l_inf(A,b,P))
    problems.append(linear_l_1_2(A_1,b_1,A_2,b_2,P))
    problems.append(linear_l_1_inf(A_1,b_1,A_inf,b_inf,P))
    problems.append(linear_l_2_inf(A_2,b_2,A_inf,b_inf,P))
    problems.append(linear_l_1_2_inf(A_1, b_1, A_2, b_2, A_inf, b_inf,P))

    λ = 0.1
    problems.append(linear_l_p_reg_q(A,b,λ,1,1,P))
    problems.append(linear_l_p_reg_q(A,b,λ,1,2,P))
    problems.append(linear_l_p_reg_q(A,b,λ,1,'inf',P))
    problems.append(linear_l_p_reg_q(A,b,λ,2,1,P))
    problems.append(linear_l_p_reg_q(A,b,λ,2,2,P))
    problems.append(linear_l_p_reg_q(A,b,λ,2,'inf',P))
    problems.append(linear_l_p_reg_q(A,b,λ,'inf',1,P))
    problems.append(linear_l_p_reg_q(A,b,λ,'inf',2,P))
    problems.append(linear_l_p_reg_q(A,b,λ,'inf','inf',P))

    Q = np.asmatrix(np.random.rand(n,n))
    Q = Q + Q.T + Q + eye(n)
    c = np.asmatrix(np.random.rand(n,1))
    problems.append(linear_QP_QP(Q, c, Q, c, P))
    problems.append(linear_QP_l_1(Q, c, A_1, b_1, P))
    problems.append(linear_QP_l_inf(Q, c, A_inf, b_inf, P))
    problems.append(linear_QP_reg_l_1(Q,c,λ,P))
    problems.append(linear_QP_reg_l_1(Q,c,λ,P))

    t = 0.5
    problems.append(DeadZone(A, b, t, P))
    problems.append(Huber(A, b, t, P))
    problems.append(ReLu(A, b, P))
    problems.append(LeakyReLu(A, b, t, P))

    print( '\nnorm.test()\n'+'_'*80 )
    for problem in problems:
        print(problem)
        print("\nx = " + str(problem.solve())+"\n")

    QP_to_linear_l_2.test()

    # --------------------------------------------------------------------------
    x = np.asmatrix(np.random.rand(3,1))
    print('\nx= ' + str(x) + '\n'\
        +'\nl_1(x)\t\t= ' + str(l_p(x,1))\
        +'\nl_2(x))\t\t= ' + str(l_p(x,2))\
        +'\nl_inf(x))\t= ' + str(l_p(x,'inf')))
