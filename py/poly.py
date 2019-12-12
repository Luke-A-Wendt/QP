import numpy as np
from foo import zeros, eye, ones
from QP import QP

from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import matrix_rank
from scipy.linalg import block_diag
def is_full_rank(A):
    if matrix_rank(A) == min( A.shape[0], A.shape[1] ):
        return True
    else:
        return False

class poly(object):# POLYHEDRONhedron Class

    def __init__(self, A_ub=None, b_ub=None,\
        A_eq=None, b_eq=None,\
        x_lb=None, x_ub=None):

        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.x_lb = x_lb
        self.x_ub = x_ub

        self.n = self.dim()

        # convert x_lb and x_ub to A_ub and b_ub
        if x_lb is not None:
            if x_ub is not None:
                assert all(x_lb <= x_ub)
            self.A_ub = np.bmat([[self.A_ub],[-eye(self.n)]])
            self.b_ub = np.bmat([[self.b_ub],[-self.x_lb]])
            self.x_lb = None
        if x_ub is not None:
            if x_lb is not None:
                assert all(x_lb <= x_ub)
            self.A_ub = np.bmat([[self.A_ub],[eye(self.n)]])
            self.b_ub = np.bmat([[self.b_ub],[self.x_ub]])
            self.x_ub = None

    def __repr__(self):
        def dim(M):
            try:
                return " ∈ R^" + str(M.shape)
            except:
                return ""
        return "\n\n"\
                + "\33[1;33m\n" + ("-"*80)\
                + "\n\n P.A_ub =\n\n" + str(self.A_ub) + dim(self.A_ub)\
                + "\n\n P.b_ub =\n\n" + str(self.b_ub) + dim(self.b_ub)\
                + "\n\n P.A_eq =\n\n" + str(self.A_eq) + dim(self.A_eq)\
                + "\n\n P.b_eq =\n\n" + str(self.b_eq) + dim(self.b_eq)\
                + "\n\n P.n = " + str(self.n)\
                + "\n\n"\
                + ("-"*80) + "\33[m"

    def dim(self):
        # dim check
        n = None
        if self.A_ub is None:
            assert self.b_ub is None, "None b_ub"
        else:
            assert self.A_ub.shape[0] == self.b_ub.shape[0]
            if n is None:
                n = self.A_ub.shape[1]
            else:
                assert n is self.A_ub.shape[1]          
        if self.A_eq is None:
            assert self.b_eq is None, "None b_eq"
        else:
            assert self.A_eq.shape[0] == self.b_eq.shape[0]
            if n is None:
                n = self.A_ub.shape[1]
            else:
                assert n is self.A_ub.shape[1]
        if self.x_lb is not None:
            if n is None:
                n = self.x_lb.shape[0]
            else:
                assert n is self.x_lb.shape[0]
        if self.x_ub is not None:
            if n is None:
                n = self.x_ub.shape[0]
            else:
                assert n is self.x_ub.shape[0]
        assert n is not None
        return n

    def affine_transform(self,A,b):
        """
        Ay-b ∈ POLYHEDRON_self  🠊  y ∈ POLYHEDRON_out
        """
        assert A.shape[0] is self.n
        assert self.x_lb is None and self.x_ub is None
        A_ub = self.A_ub * A
        b_ub = self.b_ub+self.A_ub*b
        A_eq = self.A_eq * A
        b_eq = self.b_eq+self.A_eq*b
        return poly(A_ub,b_ub,A_eq,b_eq)

    def full_rank_inverse(self,A,b):
        """
            min    J = f( A x - b )    🠊   min    J = f( y )
            x ∈ R^n                         y ∈ R^n
                    s.t. x ∈ POLYHEDRON_self                 s.t. y ∈ POLYHEDRON_out

            A^{-1} ( y - b ) ∈ POLYHEDRON_self       🠊   y ∈ POLYHEDRON_out
        """
        m = A.shape[0]
        n = A.shape[1]
        if is_full_rank(A) and n == m:
            inv_A = inv(A)
            return self.affine_transform( inv_A, inv_A*b )
        else:
            print('warning: full_rank_inverse')
            return None

    def  full_rank_right_pseudo_inverse(self,A,b):
        """
            min    J = f( A x - b )    🠊   min    J = f( A y )
            x ∈ R^n                         y ∈ R^n
                    s.t. x ∈ POLYHEDRON_self                 s.t. y ∈ POLYHEDRON_out

            y + A^+ b ∈ POLYHEDRON_self            🠊   y ∈ POLYHEDRON_out
        """
        m = A.shape[0]
        n = A.shape[1]
        if is_full_rank(A) and min(m,n) is m:
            return self.affine_transform(eye(n),pinv(A)*b)
        else:
            print("warning: full_rank_right_pseudo_inverse")
            return None

    def  full_rank_left_pseudo_inverse(self,A,b):
        """
            min    J = f( A x - b )    🠊   min    J = f( y )
            x ∈ R^n                         y ∈ R^m
                    s.t. x ∈ POLYHEDRON_self                s.t. y ∈ POLYHEDRON_out

            A^+ ( y + b) ∈ POLYHEDRON_self        🠊   y ∈ POLYHEDRON_out
        """
        m = A.shape[0]
        n = A.shape[1]
        if is_full_rank(A) and min(m,n) is n:
            pinv_A = pinv(A)
            return self.affine_transform(pinv_A,pinv_A*b)
        else:
            print("warning: full_rank_left_pseudo_inverse")
            return None

    def add(self,P):
        """
        x ∈ POLYHEDRON_self + [x;y] ∈ POLYHEDRON_in  🠊  [x;y] ∈ POLYHEDRON_out
        """
        n = self.n
        m = P.n - n
        assert m >= 0
        A_ub = None
        b_ub = None 
        A_eq = None 
        b_eq = None 
        if self.A_ub is not None and P.A_ub is not None:
            A_ub =  np.bmat([\
                    [\
                        np.bmat([\
                            [ self.A_ub, zeros(self.A_ub.shape[0],m) ]\
                        ]) 
                    ],\
                    [P.A_ub]\
                        ])
            b_ub = np.bmat([[self.b_ub],[P.b_ub]])
        elif self.A_ub is None and P.A_ub is not None:
            A_ub = P.A_ub 
            b_ub = P.b_ub

        if self.A_eq is not None and P.A_eq is not None:
            A_eq =  np.bmat([\
                    [\
                        np.bmat([\
                            [ self.A_eq, zeros(self.A_eq.shape[0],m) ]\
                        ]) 
                    ],\
                    [P.A_eq]\
                        ])
            b_eq = np.bmat([[self.b_eq],[P.b_eq]])
        elif self.A_eq is None and P.A_eq is not None:
            A_eq = P.A_eq 
            b_eq = P.b_eq

        return poly(A_ub,b_ub,A_eq,b_eq)

    def get_feasible(self):
        Q = eye(self.n)
        c = None
        return QP(Q,c,self).solve()

    def check_inequality(self,x):
        if all(self.A_ub*x <= self.b_ub):
            return True 
        else:
            return False

    @staticmethod
    def rand(n, m_ub = None, m_eq = None,x_max = 1_000):
        if m_ub is None and m_eq is None:
            return None
        else:
            A_ub    = np.asmatrix(np.random.rand(m_ub,n))
            b_ub    = np.asmatrix(np.random.rand(m_ub,1))
            A_eq    = np.asmatrix(np.random.rand(m_eq,n))
            b_eq    = np.asmatrix(np.random.rand(m_eq,1))
            x_ub    = np.asmatrix(np.random.rand(n,1))+x_max
            x_lb    = np.asmatrix(np.random.rand(n,1))-x_max
            return poly(A_ub,b_ub,A_eq,b_eq,x_lb,x_ub)
    
    @staticmethod
    def diag(P1,P2):
        """
        x ∈ POLYHEDRON1 + y ∈ POLYHEDRON2  🠊  [x;y] ∈ POLYHEDRON_out
        """
        A_ub = block_diag(P1.A_ub,P2.A_ub)
        b_ub = np.bmat([[P1.b_ub],[P2.b_ub]])
        A_eq = block_diag(P1.A_eq,P2.A_eq)
        b_eq = np.bmat([[P1.b_eq],[P2.b_eq]])
        return poly(A_ub,b_ub,A_eq,b_eq)

#     def get_sparse(self):

#         from scipy import sparse
        
#         A_ub = sparse.csr_matrix(self.A_ub)
#         b_ub = sparse.csr_matrix(self.b_ub)
#         A_eq = sparse.csr_matrix(self.A_eq)
#         b_eq = sparse.csr_matrix(self.b_eq)
#         x_lb = sparse.csr_matrix(self.x_lb)
#         x_ub = sparse.csr_matrix(self.x_ub)

#         return sparse_poly(self.n, A_ub, b_ub, A_eq, b_eq, x_lb, x_ub)

# class sparse_poly(object):#

#     def __init__(self, n, A_ub=None, b_ub=None,\
#             A_eq=None, b_eq=None,\
#                 x_lb=None, x_ub=None):

#         self.A_ub = A_ub
#         self.b_ub = b_ub
#         self.A_eq = A_eq
#         self.b_eq = b_eq
#         self.x_ub = x_ub
#         self.x_lb = x_lb
#         self.n = n

#     def __repr__(self):
#         return "\n\nself.A_ub =\n\n" + str(self.A_ub)\
#                 + "\n\nself.b_ub =\n\n" + str(self.b_ub)\
#                 + "\n\nself.A_eq =\n\n" + str(self.A_eq)\
#                 + "\n\nself.b_eq =\n\n" + str(self.b_eq)\
#                 + "\n\nself.x_lb =\n\n" + str(self.x_lb)\
#                 + "\n\nself.x_ub =\n\n" + str(self.x_ub)\
#                 + "\n\nself.n = " + str(self.n)