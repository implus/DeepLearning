
def findMax
fileIn = open('train.txt')
ans = int(0)
for line in fileIn.readlines():
    lineArr = line.strip().split()
    first = 1
    for a in lineArr:
        if first == 1:
            first = 0
        else:
            b = a.split(':')
            #print b[0], b[1]
            if int(b[0]) > ans:
                ans = int(b[0])
print ans
