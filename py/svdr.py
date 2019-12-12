#!/usr/bin/env python3

import numpy as np
class svdr(object):

    def __init__(self, A, eps = 1E-6 ):

        # A.shape = (m,n)
        m = A.shape[0]
        n = A.shape[1]

        u, s, vh = np.linalg.svd( A, full_matrices = True )
        
        ls = u
        s_diag = np.diag(s) # sorted largest to smallest
        rs = vh.getH()
        
        if m < n:
            temp = np.zeros((m,n-m))
            s_diag = np.concatenate(\
                    ( s_diag, temp ), axis=1 )
        if m > n:
            temp = np.zeros((m-n,n))
            s_diag = np.concatenate(\
                    ( s_diag, temp ), axis = 0 )        

        r = min(m,n)
        if s[0] == 0:
            r = 0
        else: # compute rank
            for i in range(r):
                if s[i]/s[0] < eps:
                    r = i
                    break

        psv         = s_diag[0:r,0:r]

        lrs         = ls[:,0:r]
        lns         = ls[:,r:m]
        rrs         = rs[:,0:r]
        rns         = rs[:,r:n]

        self.m      = m     # rows
        self.n      = n     # cols
        self.r      = r     # rank
        self.s      = s     # singluar value vector
        self.psv    = psv   # positive singular value matrix
        self.ls     = ls    # left orthonormal space matrix
        self.lrs    = lrs   # left orthonormal range space matrix
        self.lns    = lns   # left orthonormal null space matrix
        self.rs     = rs    # right orthonormal space matrix
        self.rrs    = rrs   # right orthonormal range space matrix
        self.rns    = rns   # right orthonormal null space matrix
    
    def pim(self):
        s_inv = np.reciprocal(self.s)
        s_inv_diag = np.diag(s_inv)
        psv_inv  = s_inv_diag[0:self.r,0:self.r]
        return self.rrs * psv_inv * self.lrs.H
    
    def A(self):
        return self.lrs * self.psv * self.rrs.H

    def __repr__(self):
        return \
            "m      =   "   + str(self.m)   + "\n\n"\
            "n      =   "   + str(self.n)   + "\n\n"\
            "r      =   "   + str(self.r)   + "\n\n"\
            "s      = \n"   + str(self.s)   + "\n\n"\
            "psv    = \n"   + str(self.psv) + "\n\n"\
            "ls     = \n"   + str(self.ls)  + "\n\n"\
            "lrs    = \n"   + str(self.lrs) + "\n\n"\
            "lns    = \n"   + str(self.lns) + "\n\n"\
            "rs     = \n"   + str(self.rs)  + "\n\n"\
            "rrs    = \n"   + str(self.rrs) + "\n\n"\
            "rns    = \n"   + str(self.rns) + "\n\n"\

    @staticmethod
    def test():

        A = np.matrix('1,2,3;4,5,6')
        B = svdr(A)

        print(B)

        # test inverse
        print( np.linalg.norm( A*B.pim() - np.identity(A.shape[0]) ) )

        # test reconstruction
        print( np.linalg.norm( A - B.A() ) )

