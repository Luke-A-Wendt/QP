import numpy as np
import random
def veclist2mat(x):# TODO: validate
    """
        { x[0], ..., x[n-1] },  x[i] ∈ R^m 🠊 y ∈ R^{m,n}
    """
    m = x[0].shape[0]
    n = len(x)
    y = np.asmatrix(np.zeros((m,n)))
    for i in range(n):
        assert isinstance(x[i],np.matrix)
        assert x[i].shape[1] == 1 
        y[:,i] = x[i]
    return y
def mat2veclist(x):# TODO: validate
    """
        x ∈ R^{m,n} 🠊 { y[0], ..., y[n-1] },  y[i] ∈ R^m
    """
    assert isinstance(x,np.matrix)
    n = x.shape[1]
    y = []
    for i in range(n):
        y.append(x[:,i])
    return y
def randn(m,n=1):
    return np.asmatrix(np.random.randn(m,n))
def rand(m,n=1):
    return np.asmatrix(np.random.rand(m,n))
def sprandn(m,n=1,p=0.5):
    y = randn(m,n)
    r = 0
    for i in range(m):
        for j in range(n):
            p_ = random.random()
            if p_ > p:
                y[i,j] = 0
                r += 1
    return (y,r)
def sprand(m,n=1,p=0.5):
    y = randn(m,n)
    r = 0
    for i in range(m):
        for j in range(n):
            p_ = random.random()
            if p_ > p:
                y[i,j] = 0
                r += 1
    return (y,r)
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:# x is 0
        return 0
def vec(A):# assumes np.matrix?
    A_vec = A.flatten().T
    assert A_vec.shape[1] is 1
    return A_vec
def list2vec(x):
    assert isinstance(x,list)
    n = len(x)
    y = zeros(n,1)
    for i in range(n):
        assert isinstance(x[i],int) or isinstance(x[i],float)
        y[i,0] = x[i]
    return y
def mat(A_vec, m = -1, n = -1):# -1 auto
    """
    A_vec ∈ R^mn 🠊 A ∈ R^{m,n}
    """
    # assert bool(m is -1)!=bool(n is -1), "can only auto 1"
    A_vec = np.matrix(A_vec)
    assert A_vec.shape[0] is 1 or A_vec.shape[1] is 1
    if A_vec.shape[0] is 1:
        A_vec = A_vec.T
    assert A_vec.shape[0] is m*n    
    return A_vec.reshape(m,n)
def zeros(m,n=1):
    return np.asmatrix(np.zeros((m,n)))
def ones(m,n=1):
    return np.asmatrix(np.ones((m,n)))
def inf(m,n=1):
    return np.asmatrix(np.full((m,n),np.inf))
def eye(n):
    return np.asmatrix(np.identity(n))
def hinge(x):
    if x<0:
        x=0
    return x
def dz(x,a=1):
    return hinge(x-a)-hinge(-x-a)
def sat(x,a=1):
    if x>a:
        x=a
    if x<-a:
        x=-a
    return x

# plot functions---------------
import matplotlib.pyplot as plt
def plot_dz(a=1):
    x=[]
    N=100
    for i in range(N):
        x.append(i/float(N))
    for i in range(N):
        x[i]=2*a*(2*x[i]-1)
    y=[]
    for i in range(N):
        y.append(dz(x[i],a))    
    plt.plot(x,y)
    plt.show()
def plot_sat(a=1):
    x=[]
    N=100
    for i in range(N):
        x.append(i/float(N))
    for i in range(N):
        x[i]=2*a*(2*x[i]-1)
    y=[]
    for i in range(N):
        y.append(sat(x[i],a))    
    plt.plot(x,y)
    plt.show()


