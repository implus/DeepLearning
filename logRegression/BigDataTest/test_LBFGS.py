from LBFGS import *
from LxBFGS import *
from LxDFP import *
from LxDFP2 import *
from BFGS import *
from numpy import *
import time

def findMax(fileInput):
    fileIn = open(fileInput)
    ans = int(0)
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        first = 1
        for a in lineArr:
            if first == 1:
                first = 0
            else:
                b = a.split(':')
                if int(b[0]) > ans:
                    ans = int(b[0])
    return ans

def loadData(fileTrain, fileTest, dictionary):
	train_x = []
	train_y = []
	test_x  = []
	test_y  = []
	mx = findMax(fileTrain)
	print 'ready to open file train.txt'
        fileIn = open(fileTrain)
	num = 0
	for line in fileIn.readlines():
		lineArr = line.strip().split()
		first = 1
        
                x = [0.0] * (mx + 1)
                x[0] = 1.0
                for a in lineArr:
                        if first == 1:
                            first = 0; train_y.append(dictionary[a])
                        else:
                            b = a.split(':')
                            ind = int(b[0])
                            v = float(b[1])
                            x[ind] = v

		train_x.append(x)
		#train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
		#train_y.append(float(lineArr[2]))
		num += 1
	print '---- total train num is %d -----' % (num)

	print 'ready to open file test.txt'
        fileIn = open(fileTest)
	num = 0
	for line in fileIn.readlines():
		lineArr = line.strip().split()
		first = 1
        
                x = [0.0] * (mx + 1)
                x[0] = 1.0
		for a in lineArr:
                        if first == 1:
                            first = 0
                            test_y.append(dictionary[a])
                        else:
                            b = a.split(':')
                            ind = int(b[0])
                            v = float(b[1])
                            x[ind] = v

		test_x.append(x)
		num += 1
	print '---- total test num is %d -----' % (num)
	return mat(train_x), mat(train_y).transpose(), mat(test_x), mat(test_y).transpose()


dictionary = {'1':0.0, '2':1.0}

def gaoLBFGS(fileTrain, fileTest, maxIter):
    train_x, train_y, test_x, test_y = loadData(fileTrain, fileTest, dictionary)
    print "---------------LBFGS method---------------------"
    mx = findMax(fileTrain)
    opts = {'maxIter': maxIter, 'windowLen':int(1)}
    w = trainLBFGS(train_x, train_y, opts)
    ac = testLogRegres(w, test_x, test_y)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
    print w

def gaoLxBFGS(fileTrain, fileTest, maxIter):
    train_x, train_y, test_x, test_y = loadData(fileTrain, fileTest, dictionary)
    print "---------------LxBFGS method---------------------"
    mx = findMax(fileTrain)
    opts = {'maxIter': maxIter }
    w = trainLxBFGS(train_x, train_y, opts)
    ac = testLogRegres(w, test_x, test_y)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
    print w

def gaoBFGS(fileTrain, fileTest, maxIter):
    train_x, train_y, test_x, test_y = loadData(fileTrain, fileTest, dictionary)
    print "---------------BFGS method---------------------"
    mx = findMax(fileTrain)
    opts = {'maxIter': maxIter }
    w = trainBFGS(train_x, train_y, opts)
    ac = testLogRegres(w, test_x, test_y)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
    print w

def gaoGraDecent(fileTrain, fileTest, maxIter):
    train_x, train_y, test_x, test_y = loadData(fileTrain, fileTest, dictionary)
    print "---------------Gradecent method---------------------"
    mx = findMax(fileTrain)
    opts = {'alpha':0.01, 'maxIter': maxIter, 'optimizeType': 'gradDescent'}
    w = trainLogRegres(train_x, train_y, opts)
    ac = testLogRegres(w, test_x, test_y)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
    print w


def gaoLxDFP(fileTrain, fileTest, maxIter):
    train_x, train_y, test_x, test_y = loadData(fileTrain, fileTest, dictionary)
    print "---------------LxDFP method---------------------"
    mx = findMax(fileTrain)
    opts = {'maxIter': maxIter }
    w = trainLxDFP2(train_x, train_y, opts)
    ac = testLogRegres(w, test_x, test_y)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
    print w

# nosqrt+0.01: { 32(82.25%) }   sqrt+0.01: { 8(80.25%) }   sqrt+0.5: { 9(83.75%) }  2~7s
gaoLBFGS('./train.txt', './test.txt', 100)

# nosqrt+0.01: { 8(80.5%) }     sqrt+0.01: { 32(82.25%) }   2~7s
#gaoLxBFGS('./train.txt', './test.txt', 200)

# nosqrt+0.01: { 21(79.25%) }   sqrt+0.01: { 17(77.75%) }  125~160s
#gaoBFGS('./train.txt', './test.txt', 100)

# 109(81.25%)
#gaoGraDecent('./train.txt', './test.txt', 200)


#gaoLxDFP('./train.txt', './test.txt', 200)
