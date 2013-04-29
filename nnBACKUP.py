import math,random,pickle
class node(object):
	#__slots__=('name','activation','predecessors','value','weights')
	def __init__(self,name,activation,predecessors,weights={},value=0):
		self.name = name
		self.activation = activation
		self.predecessors = predecessors
		self.weights = weights
		self.value=value
	def fire(self,inputs=[]):
		if "B" in self.name:
			return self.activation(self.value)
		elif "I" in self.name:  #input node
			sigma = sum(inputs)
			self.value = self.activation(sigma)
			return self.activation(sigma)
		else:  #hidden or output
			sigma = sum([x.value*self.weights[(str(x),str(self))] for x in self.predecessors])
			#print "SIGMA: "+str(sigma)
			self.value = self.activation(sigma)
			return self.value	
	def __str__(self):
		return self.name

class neuralNet(object):
	def __init__(self):
		i1 = node("I1",sigmoid,[])
		i2 = node("I2",sigmoid,[])
		bi = node("BI",bias,[],{})
		self.inputLayer = [i1,i2,bi]



		h1 = node("H1",sigmoid,[i1,i2])
		h2 = node("H2",sigmoid,[i1,i2])
		h3 = node("H3",sigmoid,[i1,i2,bi])
		h4 = node("H4",sigmoid,[i1,i2,bi])
		h5 = node("H5",sigmoid,[i1,i2,bi])
		bh = node("BH",bias,[],{})
		self.hiddenLayer = [h1,h2,h3,h4,h5,bh]


		o1 = node("O1",sigmoid,[h1,h2,h3,h4,h5,bh])
		o2 = node("O2",sigmoid,[h1,h2,h3,h4,h5,bh])
		o3 = node("O3",sigmoid,[h1,h2,h3,h4,h5,bh])
		o4 = node("O4",sigmoid,[h1,h2,h3,h4,h5,bh])
		self.outputLayer = [o1,o2,o3,o4]

		
		self.weights = {}
	def generateRandomWeights(self):
		self.weights = makeRandomWeights(self.inputLayer,self.hiddenLayer,self.outputLayer)
		#weights = makeRandomWeights(self.inputLayer,self.hiddenLayer,self.outputLayer)

		pickle.dump(self.weights,open("weights.txt","wb"))
		for x in self.inputLayer:
			x.weights = self.weights

		for x in self.hiddenLayer:
			x.weights = self.weights

		for x in self.outputLayer:
			x.weights = self.weights

	def loadWeightsFromFile(self):
		self.weights = unpickleWeights()

		weights = unpickleWeights()
		for x in self.inputLayer:
			x.weights = weights

		for x in self.hiddenLayer:
			x.weights = weights

		for x in self.outputLayer:
			x.weights = weights
	def fireOnInputValues(self,inputVals):
		#print "Firing on: "+str(inputVals)
		for i in range(len(self.inputLayer)):
			self.inputLayer[i].fire([inputVals[i]])
			#print str(self.inputLayer[i]) + " "+str(self.inputLayer[i].value)
		for i in range(len(self.hiddenLayer)):
			self.hiddenLayer[i].fire()
			#print str(self.hiddenLayer[i]) + " "+str(self.hiddenLayer[i].value)
		outvals = []
		for i in range(len(self.outputLayer)):
			self.outputLayer[i].fire()
			outvals.append(self.outputLayer[i].value)
			#print str(self.outputLayer[i]) + " "+str(self.outputLayer[i].value)
		#return [x.value for x in self.outputLayer]
		return outvals
	def printWeights(self):
		print "PRINTING WEIGHTS"
		print "-----------------"
		for x in self.weights:
			print str(x[0]) +"->"+ str(x[1]) + " "+ str(self.weights[x])
		#print "\nFiring I1,I2 and H1"
		print "\n"
	def saveWeights(self):
		pickle.dump(self.weights,open("weights.txt","wb"))
	def updateWeights(self):
		for x in self.inputLayer:
			x.weights = self.weights
		for x in self.hiddenLayer:
			x.weights = self.weights
		for x in self.outputLayer:
			x.weights = self.weights
		return self

	
def sigmoid(x):
	return (1/(1+math.exp(-1*x)))
def threshold(x):
	return x >= .75  #sample threshold
def bias(x):
	return 1
def unpickleWeights():
	return pickle.load(open("weights.txt"))
def unpickleBiases():
	return pickle.load(open("biasfile.txt"))
def makeRandomWeights(inputLayer,hiddenLayer,outputLayer):
	weights = {}
	for i in inputLayer:
		for h in hiddenLayer:
			w=random.random()
			modifier = random.randint(0,1)
			if modifier == 0:
				w = -w
			weights[str(i),str(h)] = w
	for h in hiddenLayer:
		for o in outputLayer:
			w=random.random()
			modifier = random.randint(0,1)
			if modifier == 0:
				w = -w
			weights[str(h),str(o)] = w
	
	return weights
def makeRandomBiases(b):
	for x in b:
		w = random.random()
		modifier = random.randint(0,1)
		if modifier == 0:
			w = -w
		x.value = w
	return b
