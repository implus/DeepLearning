from BFGS import *
from LBFGS import *
from numpy import *
import matplotlib.pyplot as plt
import time

iterateTime = 300

def loadData():
	train_x = []
	train_y = []
	print 'ready to open file testSet.txt'
        #fileIn = open('./testSet.txt')
        fileIn = open('./in.txt')
	num = 0
	for line in fileIn.readlines():
		lineArr = line.strip().split()
		train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
		train_y.append(float(lineArr[2]))
		num += 1
	print '---- total sample num is %d -----' % (num)
	print '---- total iterator time is %d ----' % iterateTime
	return mat(train_x), mat(train_y).transpose()


'''
## step 1: load data
print "step 1: load data..."
train_x, train_y = loadData()
test_x = train_x; test_y = train_y


print "---------------BFGS method---------------------"
opts = {'maxIter': 10}
print "step 2: training..."
w = trainBFGS(train_x, train_y, opts)
print "step 3: testing..."
ac = testLogRegres(w, train_x, train_y)
print "step 4: show the result..."	
print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
print w
showLogRegres(w, train_x, train_y)
'''

## step 1: load data
print "step 1: load data..."
train_x, train_y = loadData()
test_x = train_x; test_y = train_y
print "---------------LBFGS method---------------------"
opts = {'maxIter': 100, 'windowLen': 2}
print "step 2: training..."
w = trainLBFGS(train_x, train_y, opts)
print "step 3: testing..."
ac = testLogRegres(w, train_x, train_y)
print "step 4: show the result..."	
print 'The classify accuracy is: %.3f%%\n' % (ac * 100)
print w
showLogRegres(w, train_x, train_y)

