from nn import *
def main():
	n=neuralNet(2,5,4)
	

	n.generateRandomWeights()

	fireOnTrainingData(n)
	fireOnTestData(n)
	#n.loadWeightsFromFile()
	#n.printWeights()

	# #fireOnTrainingData(n)
	#f = open("data.txt","wb")
	#buff = ""
	for x in range(1000):

	 	n=train(n)
 		n=n.updateWeights()
	 	#sse=fireOnTestData(n)
	 	#buff.join(str(x)+","+str(sse)+"A")
	#f.write(buff)
	

	#n.printWeights()
	n.saveWeights()
 	n=n.updateWeights()
	print "\n\n\n\n"
		
	fireOnTrainingData(n)
	fireOnTestData(n)

	# f = open("data.txt","wb")
	# x=0
	# y=0
	# for x in [z/50.0 for z in range(0,50)]:
	# 	for y in [z/50.0 for z in range(0,50)]:
	# 		outweights = n.fireOnInputValues([x,y,1])
	# 		classification = classify(outweights)
	# 		f.write(str(x)+","+str(y)+","+str(classification)+"A")

	# f=open("data.txt","wb")
	# for line in open("training.csv"):
	# 	#print line
	# 	line = line.strip()
	# 	line = line.split(",")
	# 	outweights = n.fireOnInputValues([float(line[0]),float(line[1]),1])
	# 	classification = classify(outweights)
	# 	targetVector = [0,0,0,0]
	# 	targetVector[int(line[2])-1] = 1
	# 	#update(n,targetVector,outweights)
	# 	if classification == int(line[2]):
	# 		f.write(line[0]+","+line[1]+",cA")
	# 	else:
	# 		f.write(line[0]+","+line[1]+",xA")



def train(n):
	for line in open("training.csv"):
		line = line.strip()
		line = line.split(",")
		outweights = n.fireOnInputValues([float(line[0]),float(line[1]),1])
		targetVector = [0,0,0,0]
		#print line[2]
		targetVector[int(line[2])-1] = 1
		#print targetVector
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
def fireOnTestData(n):
	total = 0
	correct = 0.0
	sse = 0.0
	for line in open("test.csv"):
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
		sse+=sumError(outweights,targetVector)
		#print "Machine scored: "+str(classification)+".  Expected Value was: "+line[2]
		#print "Out Values: "+str(outweights)
	print "We got "+str(int(correct))+" out of "+str(total)+" correct for an accuracy of: "+str(correct/total)+" percent on the test data."
	return sse
def sumError(out,correct):
	s = 0
	for x in range(len(out)):
		s+=(out[x]-correct[x])**2
	return s
def update(n,targetOutVector,realOutVector,invalues,alpha = .1):
	#print targetOutVector
	deltas = {}
	for x in n.outputLayer:
		#print x.value-targetOutVector[n.outputLayer.index(x)]
		deltas[str(x)] = (x.value-targetOutVector[n.outputLayer.index(x)]) * x.value * (1-x.value) 
	for x in n.hiddenLayer:
		d = 0
		for y in n.outputLayer:
			d+=deltas[str(y)]*n.weights[(str(x),str(y))]
		d *= (x.value*(1-x.value))
		deltas[str(x)] = d
	#Deltas are calculated 	
	#print len(deltas)
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
	
main()