from nn import *
import sys
import matplotlib.pyplot as plt
def main():
	n=neuralNet(2,5,4)
        print "Test Data"
        

        f = sys.argv[1]
        f=f.split('s')
        f=f[1].split(".")[0]
 		datafile=sys.argv[2]
        n.loadWeightsFromFile(f)

 	n=n.updateWeights()
	print "\n\n\n\n"
	fireOnTestData(n,datafile)
	filename = "classregions.png"
        makeClassificationRegions(n,filename)
        
        
def makeClassificationRegions(n,filename):
    print "Generating Image"
    for x in range(100):
        for y in range(100):
            newX = x/100.0
            newY = y/100.0
            i = n.fireOnInputValues([newX,newY,1])
            classification = classify(i)
            if classification == 1:
                plt.plot(newX,newY,"ro")
            elif classification == 2:
                plt.plot(newX,newY,"bo")
            elif classification == 3:
                plt.plot(newX,newY,"go")
            elif classification == 4:
                plt.plot(newX,newY,"yo")            
    #plt.show()
    plt.savefig(filename)
           

def fireOnTestData(n,datafile):
	#bolt nut ring scrap
	matrix = [[0,0,0,0] for x in range(4)]
	total = 0
	correct = 0.0
	sse = 0.0
	for line in open(datafile):
		#print line
		total+=1
		line = line.strip()
		line = line.split(",")
		#print "Firing on: "+str([float(line[0]),float(line[1]),1])
		outweights = n.fireOnInputValues([float(line[0]),float(line[1]),1])
		classification = classify(outweights)
		targetVector = [0,0,0,0]
		targetVector[int(line[2])-1] = 1
		if classification == int(line[2]):
			correct+=1

		else:
			sse+=sumError(outweights,targetVector)
		matrix[classification-1][int(line[2])-1]+=1	


	printConfusionMatrix(matrix)
	printProfit(matrix)
	print "We got "+str(int(correct))+" out of "+str(total)+" correct for an accuracy of: "+str(correct/total)+" percent on the training data."

	return sse
def printConfusionMatrix(l):
	print "   B  N  R  S"
	count = 0
	lst = ["B","N","R","S"]
	for x in l:
		print str(lst[count])+" "+str(x)
		count+=1


def printProfit(l):
	profit = 0.0
	for x in range(4):
		if x == 0:
			profit+=.2 * l[0][0]
			profit-= .07 * (l[0][1]+l[0][2]+l[0][3])
		if x == 1:
			profit+=.15 * l[1][1]
			profit-= .07 * (l[1][0]+l[1][2]+l[1][3])
		if x == 2:
			profit+=.05 * l[2][2]
			profit-= .07 * (l[2][1]+l[2][0]+l[2][3])
		if x == 3:
			profit-=.03 * l[3][3]
			profit-= .03 * (l[3][1]+l[3][2]+l[3][0])

	print "Profit: "+str(profit)



def classify(outweights):
	return outweights.index(max(outweights))+1
	
main()