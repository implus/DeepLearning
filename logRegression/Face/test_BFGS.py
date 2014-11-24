from LBFGS import *
from numpy import *
import time

featureNumber = 4 
Kind = 3

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
                x[0] = 1
                ind = 1
                for a in lineArr:
                        if first == 1:
                            first = 0; train_y.append(dictionary[a])
                        else:
                            x[ind] = float(a)
                            ind += 1
                train_x.append(x)
		num += 1
	print '---- total train num is %d -----' % (num)
        return mat(train_x), mat(train_y).transpose()

def loadTest(fileTest):
	test_x  = []
	test_y  = []

	print 'ready to open file test.txt'
        fileIn = open(fileTest)
	num = 0
	for line in fileIn.readlines():
		lineArr = line.strip().split()
		first = 1
                x = [0.0] * (featureNumber + 1)
                x[0] = 1
                ind = 1
                for a in lineArr:
                        if first == 1:
                            first = 0
                            test_y.append(int(a))
                        else:
                            x[ind] = float(a)
                            ind += 1
                test_x.append(x)
                num += 1
	print '---- total test num is %d -----' % (num)
	return mat(test_x), mat(test_y).transpose()


dictionary = {'0':0.0, '1':1.0}


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



def gaoLBFGS(fileTrain, fileTest, maxIter):
    st = time.time()
    w = [ones((featureNumber + 1, 1))] * (kind + 1) 
    for i in range(1, kind + 1):
        print '------------- number %d calculating ----------------' % (i)
        dictionary = {}
        for j in range(1, kind + 1):
            if j == i:
                dictionary[str(j)] = 1.0
            else:
                dictionary[str(j)] = 0.0
        train_x, train_y = loadData(fileTrain, dictionary)
        opts = {'maxIter': maxIter, 'windowLen':int(1)}
        w[i] = trainLBFGS(train_x, train_y, opts);

    test_x, test_y = loadTest(fileTest)
    ac = testAC(w, 1, kind + 1, test_x, test_y) 
    print 'total time is %f\n' % (time.time() - st)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)



gaoLBFGS('./testsub2.txt', './train.txt', 100)
