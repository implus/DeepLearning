from numpy import *
import matplotlib.pyplot as plt
import time
from logRegression import *

EXP_LIMIT = 20
eps = 1e-300
inf = 1e300

def sigmoid(a):
    return  1.0 / (1.0 + exp(-a))

def gki(w, xi, yi):
    return (sigmoid(w.transpose() * xi) - yi) * xi
def gk(w, x, y):
    return ((sigmoid(x * w) - y).transpose() * x).transpose()

# min J(a) = -( y * log(ha(x)) + (1 - y) * log(1 - ha(x)) )
def Jai(w, xi, yi):
    t = sigmoid(w.transpose() * xi)
    return -(yi * log(t) + (1 - y) * log(1 - t))

def dcmp(x):
    a = 0; b = 0
    if x > eps:
        a = 1
    if x < -eps:
        b = 1
    return a - b

def Ja(w, x, y):
    t = sigmoid(x * w)
    sx, sy = shape(y)
    I = ones((sx, sy))
# ***************** very important !! double limit exceeded, deal with inf *******************
    for i in range(sx):
        if dcmp(t[i, 0] - 1) == 0 or dcmp(t[i, 0]) == 0:
            return inf
    return -(y.transpose() * log(t) + (I - y).transpose() * log(I - t))


def optStep(w, d, x, y):
    lamda = 1
    while Ja(w + d * lamda, x, y) > Ja(w, x, y) + 0.1 * lamda * gk(w, x, y).transpose() * d:
        lamda = 0.9 * lamda
    return lamda

"""    
def optStep(w, d, x, y):
    return 0.09
"""

# min J(a) = -( y * log(ha(x)) + (1 - y) * log(1 - ha(x)) )
def trainLBFGS(train_x, train_y, opts):
    startTime = time.time()      

    numSamples, numFeatures = shape(train_x)
    maxIter = opts['maxIter']; m = opts['windowLen']
    w = ones((numFeatures, 1))
    g = gk(w, train_x, train_y);
    k = 0
    ql = 0
    qr = 0
    cnt = 0
    qy = [zeros((numFeatures, 1))] * m
    qs = [zeros((numFeatures, 1))] * m
    Dg = g
    a = [0] * m
    b = 0

    for k in range(maxIter):
        d = -Dg
        lamda = optStep(w, d, train_x, train_y)
        if( lamda < 0.00001 ):
            break
        s = lamda * d
        w = w + s
        ng = gk(w, train_x, train_y)
        if ng.transpose() * ng < eps:
            break
        y = ng - g
        g = ng
        

        q = g
        l = ql; r = qr
        while cnt > 0:
            i = (r - 1 + m) % m
            p = float(1.0 / (qy[i].transpose() * qs[i]))
            a[i] = float(p * qs[i].transpose() * q)
            q = q - a[i] * qy[i]
            r = (r - 1 + m) % m
            if (r + m + m) % m == l:
                break

        l = ql; r = qr
        Dg = q
        while cnt > 0:
            p = float(1.0 / (qy[l].transpose() * qs[l]))
            b = float(p * qy[l].transpose() * Dg)
            Dg = Dg + (a[l] - b) * qs[l]
            l = (l + 1) % m
            if l % m == r:
                break;


        qs[qr] = s
        qy[qr] = y
        if cnt == 1 and qr == ql:
            ql = (ql + 1) % m
        qr = (qr + 1) % m
        cnt = 1


        #D = (I - s * y.transpose() / (y.transpose() * s)) * D * (I - y * s.transpose() / (y.transpose() * s)) + s * s.transpose() / (y.transpose() * s);
        #p = (1 / (s.transpose() * y))
        #D = D + s * s.transpose() * float(p + p * p * (y.transpose() * D * y)) - (D * y * s.transpose() + s * y.transpose() * D) * float(p) # O(n^2)
        k = k + 1
        accuracy = testLogRegres(w, train_x, train_y)
        print 'lamda = %f' % lamda
        print '%d times, The classify accuracy is: %.3f%%' % (k, accuracy * 100)
    
    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return w




