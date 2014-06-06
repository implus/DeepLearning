import random

n = 1000
for i in range(0, n):
    x = random.uniform(-1000, 1000)
    y = random.uniform(-1000, 1000)
    print x, y, 
    t = 2.1456 * x + 3.3672 * y + 5.1238
    if(-100 < t and t < 100):
        print random.randint(0, 1)
    elif(t < 0):
        if(random.randint(0, 100) < 0):
            print 1
        else:
            print 0
    else:
        if(random.randint(0, 100) < 0):
            print 0
        else:
            print 1
