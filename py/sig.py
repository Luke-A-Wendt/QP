#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import mpmath

class sig(object):
    def __init__(self,name='sig',x_min=-1,x_max=1,y_min=-1,y_max=1):
        self.name = name
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    def s(self,x):
        return None
    def Ds(self,x):
        return None
    def DDs(self,x):
        return None
    def Is(self,x):
        return None
    def plot(self):
        d = (self.x_max-self.x_min)*0.25
        x = np.linspace(self.x_min-d,self.x_max+d, num=1000)

        y=[]
        Dy=[]
        DDy=[]
        Iy=[]
        for i in range(len(x)):
            y.append(self.s(x[i]))
            Dy.append(self.Ds(x[i]))
            DDy.append(self.DDs(x[i]))
            Iy.append(self.Is(x[i]))
        tol=0.1
        for i in range(len(y)-1):
            if y[i] is not None and y[i+1] is not None:
                if abs(y[i+1]-y[i])>tol:
                    y[i]=None
            if Dy[i] is not None and Dy[i+1] is not None:
                if abs(Dy[i+1]-Dy[i])>tol:
                    Dy[i]=None
            if DDy[i] is not None and DDy[i+1] is not None:
                if abs(DDy[i+1]-DDy[i])>tol:
                    DDy[i]=None

        fig = plt.figure(num=None, figsize=(10, 1.5), dpi=300, facecolor=None, edgecolor=None)
        fig.subplots_adjust(wspace=0.6)
        plt.rcParams.update({'font.size': 12})

        subfig = plt.subplot(1,4,1)
        subfig.plot( x, Iy, 'k')
        plt.title('$\int$s(u)du')
        subfig.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        subfig.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        subfig.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        subfig = plt.subplot(1,4,2 )
        subfig.plot( x, y, 'k')
        plt.title('s(u)')
        subfig.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        subfig.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        subfig.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        subfig = plt.subplot(1,4,3 )
        subfig.plot( x, Dy, 'k')
        plt.title('s\'(u)')
        subfig.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        subfig.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        subfig.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        subfig = plt.subplot(1,4,4 )
        subfig.plot( x, DDy, 'k')
        plt.title('s\'\'(u)')
        subfig.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        subfig.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        fig.savefig('./fig/'+self.name+'.png',bbox_inches='tight')
    @staticmethod
    def sat(x,a=-1,b=1):
        if x<a:
            return a
        if x>b:
            return b
        else:
            return x
    def sign(x):
        if x<0:
            return -1
        elif x>0:
            return 1
        else:
            return 0
class logistic(sig):
    def __init__(self):
        super(logistic,self).__init__('logistic',-2*math.pi,2*math.pi)
    def s(self,x):
        return 1/(1+math.exp(-x))
    def Ds(self,x):
        return self.s(x)*(1-self.s(x))
    def DDs(self,x):
        return self.s(x)*pow(1-self.s(x),2)-pow(self.s(x),2)*(1-self.s(x))
    def Is(self,x):
        return math.log(1+math.exp(x))
class relu(sig):
    def __init__(self):
        super(relu,self).__init__('relu',-1,1)
    def s(self,x):
        if x<0:
            return 0
        else:
            return 1
    def Ds(self,x):
        return 0
    def DDs(self,x):
        return 0
    def Is(self,x):
        if x<0:
            return 0
        else:
            return x
class soft_relu(sig):
    def __init__(self):
        super(soft_relu,self).__init__('soft_relu',-1,1)
    def s(self,x):
        return sig.sat(x,0,1)
    def Ds(self,x):
        if x< 0 or x>1:
            return 0
        else:
            return 1
    def DDs(self,x):
        return 0
    def Is(self,x):
        if x<0:
            return 0
        elif x>1:
            return x-0.5
        else:
            return pow(x,2)/2.0
class trap(sig):
    def __init__(self):
        super(trap,self).__init__('trap',-1.5,1.5,0,1)
    def s(self,x):
        if x<0:
            if x<-1:
                return -1
            else:
                return pow(x,2)+2*x
        else:
            if x>1:
                return 1
            else:
                return 2*x-pow(x,2)
    def Ds(self,x):
        if x<0:
            if x<-1:
                return 0
            else:
                return 2*x+2
        else:
            if x>1:
                return 0
            else:
                return 2-2*x
    def DDs(self,x):
        if x<0:
            if x<-1:
                return 0
            else:
                return 2
        else:
            if x>1:
                return 0
            else:
                return -2
    def Is(self,x):
        if x<0:
            if x<-1:
                return -x-1/3.0
            else:
                return pow(x,3)/3.0+pow(x,2)
        else:
            if x>1:
                return x-1/3.0
            else:
                return pow(x,2)-pow(x,3)/3.0
class sign(sig):
    def __init__(self):
        super(sign,self).__init__('sign',-1,1,0,1)
    def s(self,x):
        if x<0:
            return -1
        elif x>0:
            return 1
        else:
            return 0
    def Ds(self,x):
        return 0
    def DDs(self,x):
        return 0
    def Is(self,x):
        if x<0:
            return -x
        else:
            return x
class Gauss(sig):
    def __init__(self):
        super(Gauss,self).__init__('Gauss',-5,5,0,1)
    def s(self,x):
        return math.erf(x)
    def Ds(self,x):
        return 2*math.exp(-pow(x,2))/math.sqrt(math.pi)
    def DDs(self,x):
        return -4*math.exp(-pow(x,2))*x/math.sqrt(math.pi)
    def Is(self,x):
        return x*math.erf(x)+(math.exp(-pow(x,2))-1)/math.sqrt(math.pi)
class exp(sig):
    def __init__(self):
        super(exp,self).__init__('exp',-math.pi,math.pi)
    def s(self,x):
        if x<0:
            return math.exp(x)-1
        else:
            return -math.exp(-x)+1
    def Ds(self,x):
        if x<0:
            return math.exp(x)
        else:
            return math.exp(-x)
    def DDs(self,x):
        if x<0:
            return math.exp(x)
        else:
            return -math.exp(-x)
    def Is(self,x):
        if x<0:
            return math.exp(x)-x-1
        else:
            return math.exp(-x)+x-1
class inv_log(sig):
    def __init__(self):
        super(inv_log,self).__init__('inv_log',-10,10)
    def s(self,x):
        if x<0:
            return math.log(2)/math.log(2-x)-1
        else:
            return -math.log(2)/math.log(2+x)+1
    def Ds(self,x):
        if x<0:
            return -math.log(2)/(x-2)/pow(math.log(2-x),2)
        else:
            return math.log(2)/(x+2)/pow(math.log(2+x),2)
    def DDs(self,x):
        if x<0:
            return math.log(2)*(2+math.log(2-x))/pow(x-2,2)/pow(math.log(2-x),3)
        else:
            return -math.log(2)*(2+math.log(2+x))/pow(x+2,2)/pow(math.log(2+x),3)
    def Is(self,x):
        if x<0:
            return -x-math.log(2)*mpmath.li(2-x)+math.log(2)*mpmath.li(2)
        else:
            return x-math.log(2)*mpmath.li(2+x)+math.log(2)*mpmath.li(2)
class inv_lin(sig):
    def __init__(self):
        super(inv_lin,self).__init__('inv_lin',-10,10)
    def s(self,x):
        if x<0:
            return 1/(1/x-1)
        else:
            return 1/(1/x+1)
    def Ds(self,x):
        if x<0:
            return 1/pow(x-1,2)
        else:
            return 1/pow(x+1,2)
    def DDs(self,x):
        if x<0:
            return -2/pow(x-1,3)
        else:
            return -2/pow(x+1,3)
    def Is(self,x):
        if x<0:
            return -(x+math.log(1-x))
        else:
            return x-math.log(1+x)
class poly0(sig):
    def __init__(self):
        super(poly0,self).__init__('poly0',-2,2)
    def s(self,x):
        if abs(x)<=1:
            return x
        else:
            return sig.sign(x)
    def Ds(self,x):
        if abs(x)<=1:
            return 1
        else:
            return 0
    def DDs(self,x):
        return 0
    def Is(self,x):
        if abs(x)<=1:
            return pow(x,2)/2.0
        else:
            return -0.5+abs(x)
class poly1(sig):
    def __init__(self):
        super(poly1,self).__init__('poly1',-2,2)
    def s(self,x):
        if abs(x)<=1:
            return (3*x-pow(x,3))/2.0
        else:
            return sig.sign(x)
    def Ds(self,x):
        if abs(x)<=1:
            return (3-3*pow(x,2))/2.0
        else:
            return 0
    def DDs(self,x):
        if abs(x)<=1:
            return x
        else:
            return 0
    def Is(self,x):
        if abs(x)<=1:
            return (6*pow(x,2)-pow(x,4))/8.0
        else:
            return 5/8.0-1+abs(x)
class poly2(sig):
    def __init__(self):
        super(poly2,self).__init__('poly2',-2,2)
    def s(self,x):
        if abs(x)<=1:
            return (15*x-10*pow(x,3)+3*pow(x,5))/8.0
        else:
            return sig.sign(x)
    def Ds(self,x):
        if abs(x)<=1:
            return (15-30*pow(x,2)+15*pow(x,4))/8.0
        else:
            return 0
    def DDs(self,x):
        if abs(x)<=1:
            return 60*(-x+pow(x,3))/8.0
        else:
            return 0
    def Is(self,x):
        if abs(x)<=1:
            return (15*pow(x,2)-5*pow(x,4)+pow(x,6))/16.0
        else:
            return 11/16-1+abs(x)
class poly3(sig):
    def __init__(self):
        super(poly3,self).__init__('poly3',-2,2)
    def s(self,x):
        if abs(x)<=1:
            return (35*x-35*pow(x,3)+21*pow(x,5)-5*pow(x,7))/16.0
        else:
            return sig.sign(x)
    def Ds(self,x):
        if abs(x)<=1:
            return (35-105*pow(x,2)+105*pow(x,4)-35*pow(x,6))/16.0
        else:
            return 0
    def DDs(self,x):
        if abs(x)<=1:
            return (-210*x+420*pow(x,3)-210*pow(x,5))/16.0
        else:
            return 0
    def Is(self,x):
        if abs(x)<=1:
            return (140*pow(x,2)-70*pow(x,4)+28*pow(x,6)-5*pow(x,8))/128.0
        else:
            return 93/128-1+abs(x)
class atan(sig):
    def __init__(self):
        super(atan,self).__init__('atan',-5,5)
        self.C = 2/math.pi
    def s(self,x):
        return self.C*math.atan(x)
    def Ds(self,x):
        return self.C/(1+pow(x,2))
    def DDs(self,x):
        return -self.C*2*x/pow(1+pow(x,2),2)
    def Is(self,x):
        return self.C*(x*math.atan(x)-math.log(1+pow(x,2))/2)
class tanh(sig):
    def __init__(self):
        super(tanh,self).__init__('tanh',-5,5)
    def s(self,x):
        return mpmath.tanh(x)
    def Ds(self,x):
        return pow(mpmath.sech(x),2)
    def DDs(self,x):
        return -2*pow(mpmath.sech(x),2)*mpmath.tanh(x)
    def Is(self,x):
        return math.log(mpmath.cosh(x))
class inv_poly0(sig):
    def __init__(self):
        super(inv_poly0,self).__init__('inv_poly0',-5,5)
    def s(self,x):
        if abs(x)<=1:
            return x/2.0
        else:
            return sig.sign(x)+(0.5-1)/x
    def Ds(self,x):
        if abs(x)<=1:
            return 1/2.0
        else:
            return -(0.5-1)/pow(x,2)
    def DDs(self,x):
        if abs(x)<=1:
            return 0
        else:
            return 2*(0.5-1)/pow(x,3)
    def Is(self,x):
        if abs(x)<=1:
            return pow(x,2)/4.0
        else:
            return abs(x)+(0.5-1)*math.log(abs(x))+1/4.0-1
class inv_poly1(sig):
    def __init__(self):
        super(inv_poly1,self).__init__('inv_poly1',-5,5)
    def Is(self,x):
        if abs(x)<=1:
            return (6*pow(x,2)/2.0-pow(x,4)/4.0)/8.0
        else:
            return abs(x)+(5/8.0-1)*math.log(abs(x))+(6/2.0-1/4.0)/8-1
    def s(self,x):
        if abs(x)<=1:
            return (6*x-pow(x,3))/8.0 
        else:
            return sig.sign(x)+(5/8.0-1)/x
    def Ds(self,x):
        if abs(x)<=1:
            return (6-3*pow(x,2))/8.0
        else:
            return -(5/8.0-1)/pow(x,2)
    def DDs(self,x):
        if abs(x)<=1:
            return -6*x/8.0
        else:
            return 2*(5/8.0-1)/pow(x,3)
class inv_poly2(sig):
    def __init__(self):
        super(inv_poly2,self).__init__('inv_poly2',-5,5)
    def s(self,x):
        if abs(x)<=1:
            return (15*x-5*pow(x,3)+pow(x,5))/16.0
        else:
            return sig.sign(x)+(11/16.0-1)/x
    def Ds(self,x):
        if abs(x)<=1:
            return (15-15*pow(x,2)+5*pow(x,4))/16.0
        else:
            return -(11/16.0-1)/pow(x,2)
    def DDs(self,x):
        if abs(x)<=1:
            return (-30*x+20*pow(x,3))/16
        else:
            return 2*(11/16.0-1)/pow(x,3)
    def Is(self,x):
        if abs(x)<=1:
            return (15*pow(x,2)/2.0-5*pow(x,4)/4.0+pow(x,6)/6.0)/16.0
        else:
            return abs(x)+(11/16.0-1)*math.log(abs(x))-1+(15/2.0-5/4.0+1/6.0)/16.0
class inv_poly3(sig):
    def __init__(self):
        super(inv_poly3,self).__init__('inv_poly3',-5,5)
    def s(self,x):
        if abs(x)<=1:
            return (140*x-70*pow(x,3)+28*pow(x,5)-5*pow(x,7))/128.0
        else:
            return sig.sign(x) -35/128.0/x
    def Ds(self,x):
        if abs(x)<=1:
            return -35/128.0*(-4 + 6*pow(x,2)-4*pow(x,4)+pow(x,6))
        else:
            return 35/128.0/pow(x,2)
    def DDs(self,x):
        if abs(x)<=1:
            return  -35/64.0*x*(6 - 8*pow(x,2) + 3*pow(x,4))
        else:
            return -35/64.0/pow(x,3)
    def Is(self,x):
        if abs(x)<=1:
            return (1680*pow(x,2)-420*pow(x,4)+112*pow(x,6)-15*pow(x,8))/3072.0
        else:
            return abs(x)-35/128.0*math.log(abs(x))-1715/3072.0
class gd(sig):
    def __init__(self):
        super(gd,self).__init__('gd',-5,5)
    def s(self,x):
        return 2 * math.atan(math.tanh(x/2.0))
    def Ds(self,x):
        return self.sech(x)
    def DDs(self,x):
        return -math.tanh(x)*self.sech(x)
    #=====================================================
    def sech(self,x):
        return 1/(pow(math.sinh(x),2)+pow(math.cosh(x),2))
class parabola(sig):
    def __init__(self):
        super(parabola,self).__init__('parabola',-5,5)
    def s(self,x):
        return x/math.sqrt(1+pow(x,2))
    def Ds(self,x):
        return 1/pow(1+pow(x,2),3/2.0)
    def DDs(self,x):
        return -3*x/pow(1+pow(x,2),5/2.0)
    def Is(self,x):
        return math.sqrt(1+pow(x,2))-1 
class deadzone(sig):
    def __init__(self):
        super(deadzone,self).__init__('deadzone',-5,5)
    def s(self,x):
        if abs(x)<=1:
            return 0
        else:
            return sig.sign(x) - 1/x
    def Ds(self,x):
        if abs(x)<=1:
            return 0
        else:
            return 1/pow(x,2)
    def DDs(self,x):
        if abs(x)<=1:
            return  0
        else:
            return -2/pow(x,3)
    def Is(self,x):
        if abs(x)<=1:
            return 0
        else:
            return sig.sign(x)*x - math.log(abs(x))-1
class Huber(sig):
    def __init__(self):
        super(Huber,self).__init__('Huber',-5,5)
    def s(self,x):
        if abs(x)<=1:
            return x/2.0
        else:
            return sig.sign(x) - 1/x/2.0
    def Ds(self,x):
        if abs(x)<=1:
            return 1/2.0
        else:
            return 1/pow(x,2)/2.0
    def DDs(self,x):
        if abs(x)<=1:
            return  0
        else:
            return -1/pow(x,3)
    def Is(self,x):
        if abs(x)<=1:
            return pow(x,2)/4.0
        else:
            return sig.sign(x)*x - math.log(abs(x))/2.0-1.0+1/4.0
def test():
    sigmoids = []
    sigmoids.append(logistic())
    sigmoids.append(relu())
    sigmoids.append(soft_relu())
    sigmoids.append(soft_relu())
    sigmoids.append(trap())
    sigmoids.append(sign())
    sigmoids.append(Gauss())
    sigmoids.append(exp())
    sigmoids.append(inv_log())
    sigmoids.append(inv_lin())
    sigmoids.append(poly0())
    sigmoids.append(poly1())
    sigmoids.append(poly2())
    sigmoids.append(poly3())
    sigmoids.append(atan())
    sigmoids.append(tanh())
    sigmoids.append(inv_poly0())
    sigmoids.append(inv_poly1())
    sigmoids.append(inv_poly2())
    sigmoids.append(inv_poly3())
    sigmoids.append(gd())
    sigmoids.append(parabola())
    sigmoids.append(deadzone())
    sigmoids.append(Huber())
    for i in range(len(sigmoids)):
        sigmoids[i].plot()