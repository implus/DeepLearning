from LBFGS import *
from numpy import *
import time

featureNumber = 4 
kind = 3

def loadData(fileTrain, dictionary):
	train_x = []
	train_y = []
	print 'ready to open file train.txt'
        fileIn = open(fileTrain)
	num = 0
	for line in fileIn.readlines():
		lineArr = line.strip().split()
		first = 1
                x = [0.0] * (featureNumber + 1)
                ind = 0
                for a in lineArr:
                        if first == 1:
                            first = 0; train_y.append(dictionary[a])
                        else:
                            if ind == 0:
                                x[ind] = 1.0
                                ind += 1
                                continue
                            x[ind] = float(a)
                            ind += 1
                train_x.append(x)
		num += 1
	print '---- total file num is %d -----' % (num)
        return mat(train_x), mat(train_y).transpose()

dictionary = {'1':1.0, '2':2.0, '3':3.0}

def testAC(w, s, t, test_x, test_y):
    nS, nF = shape(test_x)
    mC = 0
    for i in range(nS):
        p = s; mx = 0.0
        for j in range(s, t):
            predict = sigmoid(test_x[i, :] * w[j])[0, 0]
            if predict > mx:
                mx = predict
                p = j
        if p == int(test_y[i, 0]):
            mC += 1
        print ' %dth true is %d, predict is %d ' % (i, int(test_y[i, 0]), p)
    accuracy = float(mC) / nS
    return accuracy


def gaoLBFGS(fileTrain, maxIter):
    st = time.time()
    w = [ones((featureNumber + 1, 1))] * (kind + 1) 

    dictionary = {'1':1.0, '2':2.0, '3':3.0}
    allx, ally = loadData(fileTrain, dictionary)
    test_x  = allx[0:10] 
    test_x = vstack((test_x, allx[50:60]))
    test_x = vstack((test_x, allx[100:110]))
    print test_x
    test_y  = ally[0:10] 
    test_y = vstack((test_y, ally[50:60]))
    test_y = vstack((test_y, ally[100:110]))
    print test_y

    print shape(test_x)
    print shape(test_y)


    for i in range(1, kind + 1):
        print '------------- number %d calculating ----------------' % (i)
        dictionary = {}
        for j in range(1, kind + 1):
            if j == i:
                dictionary[str(j)] = 1.0
            else:
                dictionary[str(j)] = 0.0
        allx, ally = loadData(fileTrain, dictionary)
        train_x = allx[10:50];  train_x = vstack((train_x, allx[60:100]));  train_x = vstack((train_x, allx[110:150])); 
        train_y = ally[10:50];  train_y = vstack((train_y, ally[60:100]));  train_y = vstack((train_y, ally[110:150])); 
        opts = {'maxIter': maxIter, 'windowLen':int(1)}
        print shape(train_x)
        print shape(train_y)
        w[i] = trainLBFGS(train_x, train_y, opts);

    ac = testAC(w, 1, kind + 1, test_x, test_y) 
    print 'total time is %f\n' % (time.time() - st)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)

gaoLBFGS('iris.txt', 100)
