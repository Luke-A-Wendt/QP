import numpy as np
import cvxopt

class QP(object):#

    def __init__(self, Q = None, c = None, P = None):

        self.Q = Q
        self.c = c
        self.P = P
        self.dim_check()

        self.x_needs_recovery = False

    def cvxopt_base_matrix_2_np_matrix(self,x):# TODO: validate
        m = x.shape[0]
        n = x.shape[1]
        y = np.asmatrix(np.zeros((m,n)))
        for i in range(m):
            for j in range(n):
                y[i,j] = x[i,j]
        return y

    def solve(self):

        cvxopt.solvers.options['show_progress'] = False
        if self.P is not None:
            Q,c,A_ub,b_ub,A_eq,b_eq = self.np_matrix_2_cvxopt_matrix(\
                self.Q,self.c,\
                self.P.A_ub,self.P.b_ub,self.P.A_eq,self.P.b_eq)
        else:
            Q,c,A_ub,b_ub,A_eq,b_eq = self.np_matrix_2_cvxopt_matrix(\
                self.Q,self.c)
        Q, A_ub, A_eq = self.cvxopt_matrix_2_cvxopt_sparse( Q, A_ub, A_eq )
        if Q is None:
            # http://cvxopt.org/examples/tutorial/lp.html
            sol = cvxopt.solvers.lp(c, A_ub, b_ub, A_eq, b_eq)            
        else:
            # http://cvxopt.org/examples/tutorial/qp.html
            sol = cvxopt.solvers.qp(Q, c, A_ub, b_ub, A_eq, b_eq)
        try:
            x = sol['x']
            if x[0,0] is None:# x = [[None]]
                self.x = None
            else:
                self.x = np.asmatrix(np.array(x))
        except:
            self.x = None
        if self.x_needs_recovery and self.x is not None:
            self.x = self.A_recover * x - self.b_recover
        return self.x

    def dim_check(self):

        # dim check
        if self.P is not None:
            n = self.P.n
        else:
            n = None
        if self.Q is not None:
            if n is None:
                n = self.Q.shape[0]
            else:
                assert n is self.Q.shape[0]
        if self.c is not None:
            if n is None:
                n = self.c.shape[0]
            else:
                assert n is self.c.shape[0]
        else:
            self.c = np.asmatrix(np.zeros((n,1)))
        return n

    def __repr__(self):
        return  "\n\nQ =\n\n" + str(self.Q)\
                + "\n\nc =\n\n" + str(self.c)\
                + str(self.P)

    @staticmethod
    # http://cvxopt.org/examples/tutorial/creating-matrices.html
    def cvxopt_matrix_2_cvxopt_sparse( Q = None, A_ub = None, A_eq = None ):

        if Q is not None:
            Q = cvxopt.sparse(Q)
        if A_ub is not None:
            A_ub = cvxopt.sparse(A_ub)
        if A_eq is not None:
            A_eq = cvxopt.sparse(A_eq)
        return (Q,A_ub,A_eq)
    @staticmethod
    def np_matrix_2_cvxopt_matrix(Q = None, c = None,\
            A_ub = None,b_ub = None,A_eq = None,b_eq = None):

        if Q is not None:
            Q = cvxopt.matrix(Q)
        if c is not None:
            c = cvxopt.matrix(c)
        if A_ub is not None:
            A_ub = cvxopt.matrix(A_ub)
            b_ub = cvxopt.matrix(b_ub)
        if A_eq is not None:
            A_eq = cvxopt.matrix(A_eq)
            b_eq = cvxopt.matrix(b_eq)
        return (Q,c,A_ub,b_ub,A_eq,b_eq)

class LP(object):
    def __init__(self, c = None, P = None):
        self.QP = QP(None,c,P)
    def solve(self):
        x = self.QP.solve()
        return x
    def __repr__(self):
        return  "\n\nc =\n\n" + str(self.QP.c)\
                + "\n\nP =\n\n" + str(self.QP.P)