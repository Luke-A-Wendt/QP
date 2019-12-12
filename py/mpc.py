
import numpy as np
from scipy.linalg import block_diag
class lti(object):
    def __init__(self,A,B,x0):
        """
        lti.x[t+1]  = lti.A * lti.x[t] + lti.B * lti.u[t]
        lti.x[0]    = lti.x0
        """
        self.n = A.shape[0] # dimension of x
        self.m = B.shape[1] # dimension of u
        assert self.n == B.shape[0], "B rows equal A rows"
        self.A  = A
        self.B  = B
        self.x0 = x0
    
    def __repr__(self):
        return "\nlti:"\
                + '\n'+'`'*80+'\n\n'\
                + "x[t+1]\t= A * x[t] + B * u[t]\n"\
                + "x[0]\t= x0\n"\
                "\n\nA =\n\n" + str(self.A)\
                + "\n\nB =\n\n" + str(self.B)\
                + "\n\nx[0] =\n\n" + str(self.x0)

class lti2qp(object):
    def __init__(self, lti, T):

        self.__lti = lti
        self.__T = T

        """
        
        lti.x[t+1] = lti.A * lti.x[t] + lti.B * lti.u[t], t = 0, ..., T-1
        
        pypi_qp.b_eq = pypi_qp.A_eq * pypi_qp.x,

        pypi_qp.x =  [
                    lti.x[1]
                    ...
                    lti.x[T]
                    lti.u[0]
                    ...
                    lti.u[T-1]
                ]

        pypi_qp.A_eq =   [
                        [zeros(n,n(T-1)), zeros(n); 
                            kron(eye(T-1), lti.A), zeros( n(T-1),n)] - eye(nT),
                        kron( eye( T ), lti.A )
                    ]
        
        pypi_qp.b_eq = [ -lti.A * lti.x0; zeros((T-1)*n,1) ]

        """

        m11 = np.zeros( ( lti.n, lti.n*(T-1) ) )
        m12 = np.zeros( ( lti.n, lti.n ) )
        m21 = np.kron( np.identity( T-1 ), lti.A )
        m22 = np.zeros( ( lti.n*(T-1), lti.n ) )

        m1_ = np.concatenate( ( m11, m12 ), axis = 1 )
        m2_ = np.concatenate( ( m21, m22 ), axis = 1 )
        m = np.concatenate( ( m1_, m2_ ), axis = 0 ) - np.identity( lti.n*T )

        self.__A_eq = np.concatenate( (m,\
            np.kron( np.identity( T ), lti.B )), axis = 1 )

        self.__b_eq = np.concatenate( (-lti.A*lti.x0,\
            np.zeros( ( lti.n*(T-1), 1 ) )), axis = 0 )
        
        self.__A_ub = None
        self.__b_ub = None
        self.__Q = None
        self.__c = None
    
    def __repr__(self):
        return "\nQP:"\
                + '\n'+'`'*80+'\n\n'\
                + "min\t1/2 * x^T * Q * x + c^T * x\n\n"\
                + "s.t.\tA_ub * x <= b_ub\n"\
                + "\tA_eq * x  = b_eq\n\n"\
                + "x = \t[\n\t\tlti.u[0]\n\t\t.\n\t\t.\n\t\t.\n\t\tlti.u[T-1]"\
                + "\n\t\tlti.x[0]\n\t\t.\n\t\t.\n\t\t.\n\t\tlti.x[T]"\
                + "\n\t]"\
                + "\n\nQ =\n\n" + str(self.__Q)\
                + "\n\nc =\n\n" + str(self.__c)\
                + "\n\nA_ub =\n\n" + str(self.__A_ub)\
                + "\n\nb_ub =\n\n" + str(self.__b_ub)\
                + "\n\nA_eq =\n\n" + str(self.__A_eq)\
                + "\n\nb_eq =\n\n" + str(self.__b_eq)

    def lqr(self,P,R):
        """
            J = ∑_{t=0}^{T-1} x[t+1]^T * P * x[i+1] + u[i]^T * R * u[i]
        """
        assert P.shape[0] == self.__lti.B.shape[0], "P must be n"
        assert R.shape[0] == self.__lti.B.shape[1], "R must be m"

        self.__Q = block_diag   (\
                                    np.kron( np.identity( self.__T ), R ),\
                                    P,\
                                    np.kron( np.identity( self.__T ), P ) \
                                )

    def bounds(self,u_min,u_max,x_min,x_max):

        # TODO:
        n = self.__lti.n
        m = self.__lti.m
        T = self.__T

        assert len(u_min) == m and len(u_max) == m, "dim u"
        assert len(x_min) == n and len(x_max) == n, "dim x"

        A_ub = np.array([], dtype=np.int64).reshape(0,T*m+n+T*n)
        b_ub = np.array([], dtype=np.int64).reshape(0,1)
        print('here:')
        for i in range(m):
            if u_min[i] is not None:
                temp = np.zeros((1,n+m))
                for j in range(n+m):
                    if j == i:
                        temp[i] = 1
                print(temp)
        #         A_ub = np.concatenate( ( A_ub, temp ), axis = 0)
        #         b_ub = np.concatenate( ( b_ub, u_min[i] ), axis = 0 )

    @staticmethod
    def test():
        print( '\nmpc.test()\n'+'_'*80 )
        sys = lti( np.matrix('1,2;3,4'), np.matrix('5;6'), np.matrix('7;8') )
        print(sys)
        qp = lti2qp(sys,3)
        qp.lqr( np.matrix("1,0;0,1"), np.matrix("2") )
        print(qp)

def test():
    lti2qp.test()