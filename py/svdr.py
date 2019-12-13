#!/usr/bin/env python3

import numpy as np
class svdr(object):

    def __init__(self, A, eps = 1E-6 ):

        assert isinstance(A,np.matrix)

        # A.shape = (m,n)
        self.m = A.shape[0] # rows
        self.n = A.shape[1] # cols

        # SVD: A = U*S*V.H
        self.U, self.s, V_H = np.linalg.svd( A, full_matrices = True )
        self.V = V_H.getH()

        # compute rank
        self.r = min(self.m,self.n)
        if self.s[0] == 0:
            self.r = 0
        else:
            for i in range(self.r):
                if self.s[i]/self.s[0] < eps:
                    self.r = i
                    break

        self.mat_ = A
        self.S_ = None
        self.approx_ = None
        self.pinv_ = None

    def s_pos(self):
        return self.s[0:self.r]
    def shape(self):
        return (self.m,self.n)
    def rank(self):
        return self.r
    def mat(self):
        return self.mat_
    def S(self):
        if self.S_ is None:
            S = np.diag(self.s) # sorted largest to smallest
            if self.m < self.n:
                temp = np.zeros((self.m,self.n-self.m))
                self.S_ = np.concatenate(\
                        ( S, temp ), axis = 1 )
            if self.m > self.n:
                temp = np.zeros((m-n,n))
                self.S_ = np.concatenate(\
                        ( S, temp ), axis = 0 )
        return self.S_    
    def S_pos(self):
        if self.S_ is None:
            self.S()
        return self.S_[0:self.r,0:self.r]
    def U_range(self):
        return self.U[:,0:self.r]
    def U_null(self):
        return self.U[:,self.r:self.m]
    def V_range(self):
        return self.V[:,0:self.r]
    def V_null(self):
        return self.V[:,self.r:self.n]
    def pinv(self):
        if self.pinv_ is None:
            if self.r != 0:
                S_pos_inv = np.diag(np.reciprocal(self.s_pos()))
                self.pinv_ = self.V_range() * S_pos_inv * self.U_range().H
            else:
                self.pinv_ = np.asmatrix(np.zeros((n,m)))
        return self.pinv_
    def approx(self):
        if self.approx_ is None:
            self.approx_ = self.U_range() * self.S_pos() * self.V_range().H
        return self.approx_

    def __repr__(self):
        def line(name,value):
            return name + " = \n" + str(value) + "\n\n"
        return "\n" + '-'*80+\
            "\n(m,n,r) = " + str((self.m,self.n,self.r)) +"\n"\
            "\n"+\
            line(".U",self.U)+\
            line(".V",self.V)+\
            "\n"+\
            line(".s",self.s)+\
            line(".s_pos()",self.s_pos())+\
            line(".S()",self.S())+\
            line(".S_pos()",self.S_pos())+\
            "\n"+\
            line(".V_range()",self.V_range())+\
            line(".V_null()",self.V_null())+\
            line(".U_range()",self.U_range())+\
            line(".U_null()",self.U_null())+\
            "\n"+\
            line(".mat()",self.mat())+\
            line(".approx()",self.approx())+\
            line(".pinv()",self.pinv())+\
            '-'*80 + "\n\n"

    @staticmethod
    def test():

        A = np.matrix('1,2,3;4,5,6')*1.0
        
        A_svdr = svdr(A)
        print(A_svdr)

        # test inverse
        print(np.linalg.norm(A*A_svdr.pinv()-np.identity(A.shape[0])))

        # test reapproxion
        print(np.linalg.norm(A-A_svdr.approx()))
