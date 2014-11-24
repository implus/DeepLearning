from numpy import *
import matplotlib.pyplot as plt
import time
from logRegression import *

EXP_LIMIT = 20
eps = 1e-300
inf = 1e300
tereps = 1e-6

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
    lamda = ((w.transpose() * w)[0, 0] / ((w - d).transpose() * (w - d))[0, 0]);
    #lamda = sqrt(1.0 / float((w - d).transpose() * (w - d)));
    #lamda = 1;0.1;
    #lamda = (w.transpose() * w)[0, 0] / (d.transpose() * d)[0, 0]
    c1 = 1e-4; c2 = 0.01
    while Ja(w + lamda * d, x, y) > Ja(w, x, y) + c1 * lamda * gk(w, x, y).transpose() * d or gk(w + lamda * d, x, y).transpose() * d < c2 * gk(w, x, y).transpose() * d:
        while Ja(w + lamda * d, x, y) > Ja(w, x, y) + c1 * lamda * gk(w, x, y).transpose() * d:
            lamda = 0.5 * lamda
        if gk(w + lamda * d, x, y).transpose() * d < c2 * gk(w, x, y).transpose() * d:
            lamda = 1.5 * lamda

    return lamda

"""    
#back ward
def optStep(w, d, x, y):
    lamda = 0.5
    while Ja(w + d * lamda, x, y) > Ja(w, x, y) + 0.01 * lamda * gk(w, x, y).transpose() * d:
        lamda = 0.9 * lamda
    return lamda

def optStep(w, d, x, y):
    return 0.09
"""

# min J(a) = -( y * log(ha(x)) + (1 - y) * log(1 - ha(x)) )
def trainLxDFP(train_x, train_y, opts):
    startTime = time.time()      

    numSamples, numFeatures = shape(train_x)
    maxIter = opts['maxIter']
    w = ones((numFeatures, 1))
    D = eye(numFeatures)
    g = gk(w, train_x, train_y);
    k = 0
    I = eye(numFeatures)
    Dg = g
    v = Dg
    p = 1.0 / float(v.transpose() * g)
    v = sqrt(p) * v
    print 'g = '
    print g
    print 'v * vt * g = '
    print v * v.transpose() * g

    for k in range(maxIter):
        d = - Dg
        lamda = optStep(w, d, train_x, train_y)
        s = lamda * d
        w = w + s
        ng = gk(w, train_x, train_y)

        accuracy = testLogRegres(w, train_x, train_y)
        print '%d times, The classify accuracy is: %.3f%%\tlamda = %f\tgradecent = %f\tchangeofw = %f\tf(x) = %f' % (k, accuracy * 100, lamda,(ng.transpose() * ng), (s.transpose() * s), Ja(w, train_x, train_y) )
        if ng.transpose() * ng < tereps or s.transpose() * s < tereps:
            break
        y = ng - g
        g = ng
        u = s - (float(v.transpose() * y)) * v
        p = 1.0 / float(u.transpose() * y)
        Dg = float(v.transpose() * g) * v + p * float(u.transpose() * g) * u
        v = Dg
        p = 1.0 / float(v.transpose() * g)
        v = sqrt(p) * v
        '''
        u = s - D * y;
        p = 1.0 / float(u.transpose() * y)
        D = D + p * u * u.transpose()
        k = k + 1
        '''
    
    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return w


