from numpy import *
import matplotlib.pyplot as plt
import time

tereps = 1e-3 
eps = 1e-20
inf = 1e300

def sigmoid(a):
    return  1.0 / (1.0 + exp(-a))

def gk(w, x, y):
    return ((sigmoid(x * w) - y).transpose() * x).transpose()

# min J(a) = -( y * log(ha(x)) + (1 - y) * log(1 - ha(x)) )
'''
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
    for i in range(sx):
        if dcmp(t[i, 0] - 1) == 0 or dcmp(t[i, 0]) == 0:
            return inf
    return -(y.transpose() * log(t) + (I - y).transpose() * log(I - t))
'''
def Ja(w, x, y):
    t = x * w
    tt = log(1 + exp(t))
    sx, sy = shape(y)
    I = ones((sx, sy))
    return -(y.transpose() * (t - tt)) + (I - y).transpose() * log(I - t + tt)))


def optStep(w, d, x, y):
    lamda =((w.transpose() * w)[0, 0] / ((w - d).transpose() * (w - d))[0, 0]);
    c1 = 1e-4; c2 = 0.01
    print c1, c2
    while Ja(w + lamda * d, x, y) > Ja(w, x, y) + c1 * lamda * gk(w, x, y).transpose() * d or gk(w + lamda * d, x, y).transpose() * d < c2 * gk(w, x, y).transpose() * d:
        while Ja(w + lamda * d, x, y) > Ja(w, x, y) + c1 * lamda * gk(w, x, y).transpose() * d:
            lamda = 0.5 * lamda
        if gk(w + lamda * d, x, y).transpose() * d < c2 * gk(w, x, y).transpose() * d:
            lamda = 1.5 * lamda
    return lamda


def testLogRegres(weights, test_x, test_y):
	numSamples, numFeatures = shape(test_x)
	matchCount = 0
	for i in range(0, numSamples):
		predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
		if predict == bool(test_y[i, 0]):
			matchCount += 1
	accuracy = float(matchCount) / numSamples
	return accuracy

# min J(a) = -( y * log(ha(x)) + (1 - y) * log(1 - ha(x)) )
def trainLBFGS(train_x, train_y, opts):
    startTime = time.time()      

    numSamples, numFeatures = shape(train_x)
    maxIter = opts['maxIter']; m = opts['windowLen']
    print 'window length = %d' % m
    w = ones((numFeatures, 1))
    print w
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
        s = lamda * d
        w = w + s
        ng = gk(w, train_x, train_y)
        #print s.transpose() * s

        accuracy = testLogRegres(w, train_x, train_y)
        print '%d times, The classify accuracy is: %.3f%%\tlamda = %f\tgradecent = %f\tchangeofw = %f\t' % (k, accuracy * 100, lamda,(ng.transpose() * ng), (s.transpose() * s) )
        if ng.transpose() * ng < tereps or s.transpose() * s < tereps:
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
    
    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return w





