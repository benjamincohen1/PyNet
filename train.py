from nn import *
import time,sys
import matplotlib.pyplot as plt
def main():
	n=neuralNet(2,5,4)
	intervals = [100,1000,10000]
	trainingfile = open("training.csv")
	lines = []
	for line in trainingfile:
		line = line.strip()
		line = line.split(",")
		lines.append(line)
	n.generateRandomWeights()
	sses=[]
	t = time.time()
	for x in range(101):
	 	n=train(n,lines)
 		n=n.updateWeights()
                #sse=findSSE(n)
                #sses.append((x,sse))
 		if x in intervals:
 			print "Trained over " +str(x)+" epocs in "+str(time.time() - t) + " seconds" 
			n.saveWeights(x)
			filename = str(x)+"classregions.png"
			makeClassificationRegions(n,filename)

 	n=n.updateWeights()
 	plotSSEs(sses)
 	print "Done Training"
def plotSSEs(s):
    for x in s:
        plt.plot(x[0],x[1],"ro")

    plt.savefig("errors.png")
def findSSE(n):
 	for line in open("training.csv"):
		#print line
	        sse = 0.0
		line = line.strip()
		line = line.split(",")
		outweights = n.fireOnInputValues([float(line[0]),float(line[1]),1])
		classification = classify(outweights)
		targetVector = [0,0,0,0]
		targetVector[int(line[2])-1] = 1
		#if classification != int(line[2]):
		sse+=sumError(outweights,targetVector)
		return sse
		
def train(n,lines):
	for line in lines:
		outweights = n.fireOnInputValues([float(line[0]),float(line[1]),1])
		targetVector = [0,0,0,0]
		targetVector[int(line[2])-1] = 1
		n.weights=update(n,targetVector,outweights,[float(line[0]),float(line[1]),1])	
	return n
def fireOnTrainingData(n):
	total = 0
	correct = 0.0
	for line in open("training.csv"):
		#print line
		total+=1
		line = line.strip()
		line = line.split(",")
		#print "Firing on: "+str([float(line[0]),float(line[1]),1])
		outweights = n.fireOnInputValues([float(line[0]),float(line[1]),1])
		classification = classify(outweights)
		targetVector = [0,0,0,0]
		targetVector[int(line[2])-1] = 1
		#update(n,targetVector,outweights)
		if classification == int(line[2]):
			correct+=1
		#print "Machine scored: "+str(classification)+".  Expected Value was: "+line[2]
		#print "Out Values: "+str(outweights)
	print "We got "+str(int(correct))+" out of "+str(total)+" correct for an accuracy of: "+str(correct/total)+" percent on the training data."
def update(n,targetOutVector,realOutVector,invalues,alpha = .1):
	#print targetOutVector
	deltas = {}
	for x in n.outputLayer:
		#print x.value-targetOutVector[n.outputLayer.index(x)]
		deltas[str(x)] = (x.value-targetOutVector[n.outputLayer.index(x)]) * sigmoid(x.value) * (1-sigmoid(x.value)) 
	for x in n.hiddenLayer:
		d = 0
		for y in n.outputLayer:
			d+=deltas[str(y)]*n.weights[(str(x),str(y))]
		d *= (sigmoid(x.value)*(1-sigmoid(x.value)))
		deltas[str(x)] = d
	#Deltas are calculated 	
	for weight in n.weights:
		if "H" in weight[0]: #hidden to out
			hiddenNodes = [str(x) for x in n.hiddenLayer]
			valueOut = n.hiddenLayer[hiddenNodes.index(weight[0])].value
			n.weights[weight] = n.weights[weight] - alpha*deltas[weight[1]]*valueOut
		else:
			inputNodes = [str(x) for x in n.inputLayer]
			valueOut = n.inputLayer[inputNodes.index(weight[0])].value

			n.weights[weight] = n.weights[weight] - alpha * deltas[weight[1]]*valueOut
	return n.weights


def classify(outweights):
	return outweights.index(max(outweights))+1

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
def sumError(out,correct):
	s = 0.0
	for x in range(len(out)):
		s+=(out[x]-correct[x])**2
	return s

main()