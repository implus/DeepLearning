from numpy import *
import matplotlib.pyplot as plt
import time
from LBFGS import *

tereps = 1e-3


def trainLogRegres(train_x, train_y, opts):
	# calculate training time
	startTime = time.time()

	numSamples, numFeatures = shape(train_x)
	alpha = opts['alpha']; maxIter = opts['maxIter']
	weights = ones((numFeatures, 1));lxshow = int(0)
	if opts['optimizeType'] == 'gradDescent':
	    print '!!!!!!!!we use ***gradDescent*** method!!!!!!!!!!!!'
	elif opts['optimizeType'] == 'stocGradDescent':
	    print '!!!!!!!!we use ***stocGradDescent*** methon!!!!!!!!'
	else:
	    print '!!!!!!!!we use ***smoothStocGradDescent*** methon!!!!!!!!'

	# optimize through gradient descent algorilthm
	for k in range(maxIter):
	    lw = weights
	    if opts['optimizeType'] == 'gradDescent': # gradient descent algorilthm
			output = sigmoid(train_x * weights)
			error = train_y - output
			weights = weights + alpha * train_x.transpose() * error
            elif opts['optimizeType'] == 'stocGradDescent': # stochastic gradient descent
			i = random.randint(0, numSamples - 1)
			output = sigmoid(train_x[i, :] * weights)
			error = train_y[i, 0] - output
			weights = weights + alpha * train_x[i, :].transpose() * error
            elif opts['optimizeType'] == 'smoothStocGradDescent': # smooth stochastic gradient descent
			# randomly select samples to optimize for reducing cycle fluctuations 
			i = random.randint(0, numSamples - 1)
			alpha = 4.0 / (1.0 + k + i) + 0.01
			#randIndex = int(random.uniform(0, len(dataIndex)))
			randIndex = i
			output = sigmoid(train_x[randIndex, :] * weights)
			error = train_y[randIndex, 0] - output
			weights = weights + alpha * train_x[randIndex, :].transpose() * error
				#del(dataIndex[randIndex]) # during one interation, delete the optimized sample
            else:
                    raise NameError('Not support optimize method type!')
            w = weights - lw
            if w.transpose() * w < tereps:
                break
            lxshow += 1; print '%d times, accuracy = %f, w change = %f' % (lxshow, testLogRegres(weights, train_x, train_y), w.transpose() * w)
            #if lxshow % 10 == 1:
                #showLogRegres(weights, train_x, train_y)
	
	print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
	return weights

