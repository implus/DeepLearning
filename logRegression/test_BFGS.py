from BFGS import *
from LBFGS import *
from numpy import *
from LxBFGS import *
from LxDFP import *
import matplotlib.pyplot as plt
import time

iterateTime = 300

def loadData(fileStr):
	train_x = []
	train_y = []
	print 'ready to open file testSet.txt'
        #fileIn = open('./testSet.txt')
        #fileIn = open('./in.txt')
        #fileIn = open('./in1000.txt')
        fileIn = open(fileStr)
	num = 0
	for line in fileIn.readlines():
		lineArr = line.strip().split()
		train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
		train_y.append(float(lineArr[2]))
		num += 1
	print '---- total sample num is %d -----' % (num)
	print '---- total iterator time is %d ----' % iterateTime
	return mat(train_x), mat(train_y).transpose()

def gaoBFGS(fileStr, maxIter):
    train_x, train_y = loadData(fileStr)
    test_x = train_x; test_y = train_y
    print "---------------BFGS method---------------------"
    opts = {'maxIter': maxIter}
    w = trainBFGS(train_x, train_y, opts)
    ac = testLogRegres(w, train_x, train_y)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
    print w
    showLogRegres(w, train_x, train_y)

def gaoLxBFGS(fileStr, maxIter):
    train_x, train_y = loadData(fileStr)
    test_x = train_x; test_y = train_y
    print "---------------LxBFGS method---------------------"
    opts = {'maxIter': maxIter}
    w = trainLxBFGS(train_x, train_y, opts)
    ac = testLogRegres(w, train_x, train_y)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
    print w
    showLogRegres(w, train_x, train_y)

def gaoLBFGS(fileStr, maxIter, windowLen):
    train_x, train_y = loadData(fileStr)
    test_x = train_x; test_y = train_y
    print "---------------LBFGS method---------------------"
    opts = {'maxIter': maxIter, 'windowLen': windowLen}
    w = trainLBFGS(train_x, train_y, opts)
    ac = testLogRegres(w, train_x, train_y)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
    print w
    showLogRegres(w, train_x, train_y)


def gaoLxDFP(fileStr, maxIter):
    train_x, train_y = loadData(fileStr)
    test_x = train_x; test_y = train_y
    print "---------------LxDFP method---------------------"
    opts = {'maxIter': maxIter}
    w = trainLxDFP(train_x, train_y, opts)
    ac = testLogRegres(w, train_x, train_y)
    print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
    print w
    showLogRegres(w, train_x, train_y)


#nosqrt+0.01 { 17, 1, 6 }  
def testLBFGS():
    gaoLBFGS('./testSet.txt', 100, 1);
    gaoLBFGS('./in1000.txt', 100, 1);
    gaoLBFGS('./in.txt', 100, 1);


#nosqrt+0.01 { 10, 16, 4 }
def testBFGS():
    gaoBFGS('./testSet.txt', 20);
    gaoBFGS('./in1000.txt', 100);
    gaoBFGS('./in.txt', 20);

#nosqrt+0.01 { 10, 17, 5 }
def testLxBFGS():
    gaoLxBFGS('./testSet.txt', 20);
    gaoLxBFGS('./in1000.txt', 100);
    gaoLxBFGS('./in.txt', 40);


def testLxDFP():
    gaoLxDFP('./testSet.txt', 20);
    gaoLxDFP('./in1000.txt', 100);
    gaoLxDFP('./in.txt', 40);


#testLBFGS()
#testBFGS()
testLxBFGS()
#testLxDFP()
