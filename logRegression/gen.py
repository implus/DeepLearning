import random

n = 10000
for i in range(0, n):
    x = random.uniform(-1000000, 1000000)
    y = random.uniform(-1000000, 1000000)
    print x, y, 
    t = 2 * x + 3 * y + 5
    if(-1 < t and t < 1):
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
