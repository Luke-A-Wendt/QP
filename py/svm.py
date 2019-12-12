

import numpy as np
from foo import eye, vec, mat, zeros, ones, inf, randn, veclist2mat
from QP import QP
from poly import poly
import math
import matplotlib.pyplot as plt
import norm

class svm(object):
    @staticmethod
    def xy_2_xposxneg(x,y):# TODO: validate
        """
            x ∈ R^{n,m}
            y ∈ {-1,1}^m
            
            to
            
            x_pos ∈ R^{n,m_pos}
            x_neg ∈ R^{n,m_neg}
        """
        n = x.shape[0]
        m = x.shape[1]
        assert y.shape[0] == m and y.shape[1] == 1
        x_pos = []
        x_neg = []
        for i in range(m):
            if y[i,0] == -1:
                x_neg.append(x[:,i])
            if y[i,0] ==  1:
                x_pos.append(x[:,i])
        m_pos = len(x_pos)
        m_neg = len(x_neg)
        x_pos = veclist2mat(x_pos)
        x_neg = veclist2mat(x_neg)
        return (x_pos, x_neg, m_pos, m_neg)# = self.xy_2_xposxneg(x,y)
    @staticmethod
    def xposxneg_2_xy(x_pos,x_neg):# TODO: validate
        """
            x_pos ∈ R^{n,m_pos}
            x_neg ∈ R^{n,m_neg}

            to

            x ∈ R^{n,m}
            y ∈ {-1,1}^m
        """
        n     = x_pos.shape[0]
        m_pos = x_pos.shape[1]
        m_neg = x_neg.shape[1]
        assert x_neg.shape[0] == n
        x = np.bmat([[x_pos,x_neg]])
        y = np.bmat([[ones(m_pos)],[-ones(m_neg)]])
        return (x,y)# = self.xposxneg_2_xy(x_pos,x_neg)
    @staticmethod
    def generate_data( N = 100 ):    
        while True:
            a = np.matrix(np.random.randn(2,1))
            a = a / math.sqrt(a.T*a)
            b = np.matrix(np.random.randn(1,1))
            x_pos=None
            x_neg=None
            for i in range(N):
                z = np.random.randn(2,1)
                noise = (2*np.matrix(np.random.randn(1,1))-1)*0.3
                if a.T*np.matrix(z)-b > 0 + noise:
                    if x_pos is None:
                        x_pos = z
                    else:
                        x_pos = np.concatenate((x_pos,z),axis=1)
                else:
                    if x_neg is None:
                        x_neg = z
                    else:
                        x_neg = np.concatenate((x_neg,z),axis=1)
            x_pos = np.matrix(x_pos)
            x_neg = np.matrix(x_neg)
            if x_pos.shape[1] > 0.25*N and x_neg.shape[1] > 0.25*N :
                break
        return (x_pos,x_neg,a,b) # = svm.generate_data( N = 100 )
    @staticmethod
    def plot_svm(x_pos, x_neg, a, b, title = ""):
        title = str(title)
        assert x_pos.shape[0] is 2 and x_neg.shape[0] is 2
        x_pos_0 = []
        x_pos_1 = []
        for i in range(x_pos.shape[1]):
            x_pos_0.append(x_pos[0,i])
            x_pos_1.append(x_pos[1,i])
        x_neg_0 = []
        x_neg_1 = []
        for i in range(x_neg.shape[1]):
            x_neg_0.append(x_neg[0,i])
            x_neg_1.append(x_neg[1,i])     
        plt.clf()        
        plt.scatter(x_pos_0,x_pos_1,c = "orange")
        plt.scatter(x_neg_0,x_neg_1,c = "blue")
        axes = plt.gca()
        x_pos_0_min, x_pos_0_max = axes.get_xlim()
        x_pos_1_min, x_pos_1_max = axes.get_ylim()
        #  a[0] * x0 + a[1] * x1 - b = 0
        x_0_ = []
        x_1_ = []
        if abs(a[1])>abs(a[0]):
            x_0 = np.linspace(x_pos_0_min, x_pos_0_max, 100)
            for i in range(len(x_0)):
                x_1 = ( b - a[0] * x_0[i] ) / a[1]
                x_1 = x_1[0,0]
                if x_1 < x_pos_1_max and x_1 > x_pos_1_min:
                    x_0_.append(x_0[i])
                    x_1_.append(x_1)         
        else:# to avoid div by zero
            x_1 = np.linspace(x_pos_1_min, x_pos_1_max, 100)
            for i in range(len(x_1)):
                x_0 = ( b - a[1] * x_1[i] ) / a[0]
                x_0 = x_0[0,0]
                if x_0 < x_pos_0_max and x_0 > x_pos_0_min:
                    x_1_.append(x_1[i])
                    x_0_.append(x_0)
        plt.plot(x_0_,x_1_,'-k', lw = 3)
        plt.title(title)
        plt.savefig('./fig/svm:'+title+'.png')

class svm_primal(svm):
    def __init__(self,x_pos,x_neg,p='inf',λ=1):
        """
            x_pos ∈ R^{n,m_pos}
            x_neg ∈ R^{n,m_neg}
        """
        n = x_pos.shape[0]
        m_pos = x_pos.shape[1]
        m_neg = x_neg.shape[1]
        assert x_neg.shape[0] is n
        if p is 1:
            Q = None
            c = np.bmat([[zeros(n+1)],[λ*ones(m_pos+m_neg)],[ones(n)]])
            A_ub =  np.bmat([\
                    [ -x_pos.T, ones(m_pos), -eye(m_pos), zeros(m_pos,m_neg), zeros(m_pos,n) ],\
                    [ x_neg.T, -ones(m_neg), zeros(m_neg,m_pos), -eye(m_neg), zeros(m_neg,n) ],\
                    [ -eye(n), zeros(n), zeros(n,m_pos), zeros(n,m_neg), -eye(n) ],\
                    [ eye(n), zeros(n), zeros(n,m_pos), zeros(n,m_neg), -eye(n) ],\
                    [ zeros(m_pos,n), zeros(m_pos), -eye(m_pos), zeros(m_pos,m_neg), zeros(m_pos,n) ],\
                    [ zeros(m_neg,n), zeros(m_neg), zeros(m_neg,m_pos), -eye(m_neg), zeros(m_neg,n) ]])
            b_ub =  np.bmat([\
                    [ -ones(m_pos) ],\
                    [ -ones(m_neg) ],\
                    [ zeros(n) ],\
                    [ zeros(n) ],\
                    [ zeros(m_pos) ],\
                    [ zeros(m_neg) ]])
            P = poly(A_ub,b_ub)
            sol = QP(Q,c,P).solve()
            self.a = sol[0:n,0]
            self.b = sol[n,0]
        if p is 2:
            Q = zeros(n+1+m_pos+m_neg,n+1+m_pos+m_neg)
            Q[0:n,0:n] = 2*eye(n)
            c = zeros(n+1+m_pos+m_neg,1)
            c[(n+1+1-1):(n+1+m_pos+m_neg)] = λ * ones(m_pos+m_neg,1)

            A_ub =  np.bmat([\
                    [ -x_pos.T, ones(m_pos), -eye(m_pos), zeros(m_pos,m_neg) ],\
                    [ x_neg.T, -ones(m_neg), zeros(m_neg,m_pos), -eye(m_neg) ],\
                    [ zeros(m_pos,n), zeros(m_pos), -eye(m_pos), zeros(m_pos,m_neg) ],\
                    [ zeros(m_neg,n), zeros(m_neg), zeros(m_neg,m_pos), -eye(m_neg) ]])
            b_ub = np.bmat([\
                    [ -ones(m_pos) ],\
                    [ -ones(m_neg) ],\
                    [ zeros(m_pos) ],\
                    [ zeros(m_neg) ]\
                    ])
            P = poly(A_ub,b_ub)
            sol = QP(Q,c,P).solve()
            self.a = sol[0:n,0]
            self.b = sol[n,0]
        if p is 'inf':
            Q = None
            c = np.bmat([[zeros(n+1)],[λ*ones(m_pos+m_neg)],[ones(1)]])
            A_ub =  np.bmat([\
                    [ -x_pos.T, ones(m_pos), -eye(m_pos), zeros(m_pos,m_neg), zeros(m_pos) ],\
                    [ x_neg.T, -ones(m_neg), zeros(m_neg,m_pos), -eye(m_neg), zeros(m_neg) ],\
                    [ -eye(n), zeros(n), zeros(n,m_pos), zeros(n,m_neg), -ones(n) ],\
                    [ eye(n), zeros(n), zeros(n,m_pos), zeros(n,m_neg), -ones(n) ],\
                    [ zeros(m_pos,n), zeros(m_pos), -eye(m_pos), zeros(m_pos,m_neg), zeros(m_pos) ],\
                    [ zeros(m_neg,n), zeros(m_neg), zeros(m_neg,m_pos), -eye(m_neg), zeros(m_neg) ]])
            b_ub = np.bmat([\
                    [ -ones(m_pos) ],\
                    [ -ones(m_neg) ],\
                    [ zeros(n) ],\
                    [ zeros(n) ],\
                    [ zeros(m_pos) ],\
                    [ zeros(m_neg) ]])
            P = poly(A_ub,b_ub)
            sol = QP(Q,c,P).solve()
            self.a = sol[0:n,0]
            self.b = sol[n,0]            
    @staticmethod
    def test():
        x_pos,x_neg,a,b = svm_primal.generate_data()
        λ=1
        for p in (1,2,'inf'):
            print( '\nsvm_primal.test() [see ./fig for output]\n'+'_'*80 )
            s = svm_primal(x_pos,x_neg,p,λ)
            title = ' (p,λ) = ' + str((p,λ))
            svm_primal.plot_svm(x_pos,x_neg,s.a,s.b,title)

class svm_dual(svm):# TODO: validate (currently not working)
    def __init__(self,x,y,λ=0.5):
        """
            x ∈ R^{n,m}, y ∈ {-1,1}^m
        """
        assert y.shape[0] == x.shape[1] and y.shape[1] == 1
        self.x = x
        self.y = y
        self.z = self.dual(x,y)
        x_pos, x_neg, m_pos, m_neg = svm.xy_2_xposxneg(x,y)
        self.b = self.bias(x_pos,x_neg)
        svm.plot_svm(x_pos, x_neg, self.get_a(), self.get_b(),'fit')
    def get_a(self):
        m = self.x.shape[1]
        a = 0
        for i in range(m):
            a += self.x[:,i]*self.z[i,0]*self.y[i,0]
        return a
    def get_b(self):
        return self.b
    def classify(self,x):
        t = 0
        for i in range(len(self.z)):
            t += self.z[i,0] * self.y[i,0] * self.K(self.x[:,i],x)
        t -= self.b
        if t > 0:
            return  1
        else:
            return -1        
    def K(self,x_i,x_j):
        t = x_i.T * x_j
        return t[0,0]
    def a(self,x):
        w = 0
        for i in range(len(self.z)):
            w += self.z[i,0] * self.y[i,0] * self.K(self.x[:,i],x)
        return w
    def dual(self,x,y):
        m = y.shape[0]
        c_dual = -ones(m)
        Q_dual =  eye(m)
        for i in range(m):
            for j in range(m):
                Q_dual[i,j] = y[i,0]*y[j,0] * self.K(x[:,i],x[:,j])        
        A_ub = -eye(m)
        b_ub =  zeros(m)
        A_eq = y.T
        b_eq = zeros(1)
        P_dual = poly(A_ub,b_ub,A_eq,b_eq)
        z = QP(Q_dual,c_dual,P_dual).solve()
        return z
    def soft_dual(self,x,y,λ=0.1):
        assert λ >= 0, "λ >= 0"
        m = y.shape[0]
        c_dual = None
        Q_dual = eye(m)
        for i in range(m):
            for j in range(m):
                Q_dual[i,j] = y[i,0]*y[j,0] * self.K(x[:,i],x[:,j])        
        A_ub =  np.bmat([\
                [   -eye(m)     ],\
                [    eye(m)     ]\
                ])
        b_ub =  np.bmat([\
                [   zeros(m)    ],\
                [   λ*ones(1)   ]\
                ])    
        A_eq = y.T
        b_eq = zeros(1)
        P_dual = poly(A_ub,b_ub,A_eq,b_eq)
        z = QP(Q_dual,c_dual,P_dual).solve()
        return z
    def soft_dual_alternative(self,x,y,λ=0.1):
        assert λ >= 0 and λ <=1, "λ ∈ R[0,1]"
        m = y.shape[0]
        c_dual = None
        Q_dual = eye(m)
        for i in range(m):
            for j in range(m):
                Q_dual[i,j] = y[i,0]*y[j,0] * self.K(x[:,i],x[:,j])        
        A_ub =  np.bmat([\
                [   -eye(m)     ],\
                [    eye(m)     ],\
                [    ones(1,m)  ]\
                ])
        b_ub =  np.bmat([\
                [   zeros(m)    ],\
                [   ones(m)/float(m) ],\
                [   λ*ones(1)   ]\
                ])
        A_eq = y.T
        b_eq = zeros(1)
        P_dual = poly(A_ub,b_ub,A_eq,b_eq)
        z = QP(Q_dual,c_dual,P_dual).solve()
        return z
    def bias(self,x_pos,x_neg):
        m_pos = x_pos.shape[1]
        m_neg = x_neg.shape[1]
        Q = None
        c =     np.bmat([\
                [   zeros(1)    ],\
                [   ones(m_pos) ],\
                [   ones(m_neg) ]\
                ])
        A_ub =  np.bmat([\
                [  ones(m_pos),  -eye(m_pos),             zeros(m_pos,m_neg) ],\
                [ -ones(m_neg),   zeros(m_neg,m_pos),    -eye(m_neg)         ],\
                [  zeros(m_pos), -eye(m_pos),             zeros(m_pos,m_neg) ],\
                [  zeros(m_neg),  zeros(m_neg,m_pos),    -eye(m_neg)         ]\
                ])
        b_ub = np.bmat([\
                [ -ones(m_pos)  ],\
                [ -ones(m_neg)  ],\
                [  zeros(m_pos) ],\
                [  zeros(m_neg) ]\
                ])
        for i in range(m_pos):
            b_ub[i,0] += self.a(x_pos[:,i])
        for i in range(m_neg):
            b_ub[i+ m_pos,0] -= self.a(x_neg[:,i])
        P = poly(A_ub,b_ub)
        sol = QP(Q,c,P).solve()
        b = sol[0,0]
        return b
    @staticmethod
    def test():
        (x_pos,x_neg,a,b) = svm.generate_data()
        svm.plot_svm(x_pos, x_neg, a, b,'init')
        (x,y) = svm.xposxneg_2_xy(x_pos,x_neg)
        svm_dual(x,y)

def test():
    svm_primal.test()
    # svm_dual.test()