## Create a simple neural network

## Aaron J. Bradley

from random import randint
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

class NeuralNet():
	"""Create and train a neural network

	"""

	def __init__(self):
		"""Initialize the neural network

		"""

	def create_network(self, neurons_input=1, neurons_hidden=0):
		"""Create the neural network

		   Currently, the only networks supported have:
		    1) One input neuron per feature
		    2) Connections from all input neurons to all hidden neurons
		    3) One output neuron unless there is only one input neuron
		"""
		
		self.rate = 0.01	#Learning rate
		self.weights_input = []
		self.weights_hidden = []
		self.weights_output = []
		self.neurons_input = neurons_input
		self.neurons_hidden = neurons_hidden

		if neurons_input > 1:
			neurons_output = 1
		else:
			neurons_output = 0
		self.neurons_output = neurons_output

		# set random starting weights
		for i in range(neurons_input):
			self.weights_input.append(randint(-1,1))
		for i in range(neurons_hidden):
			for j in range(neurons_input*neurons_hidden):
				self.weights_hidden.append(randint(-1,1))
		for i in range(neurons_output):
			for j in range(neurons_hidden):
				self.weights_output.append(randint(-1,1))


	def feed_forward(self, inputs):
		"""Return an output based on inputs

		   Expects inputs of the form [x,y] where both x and y are features
		"""

		#Input weights
		input_hidden = [] 	#values passed from input -> hidden neurons
		for i,w in enumerate(self.weights_input):
			input_hidden.append(inputs[i] * w)
		#Hidden weights
		hidden_output = []
		for i,w in enumerate(self.weights_hidden):
			_sum = 0
			for j, values_input in enumerate(self.input_hidden):
				_sum += values_input * w
			hidden_output.append(_sum)
		#Output weights
		output = []
		for i,w in enumerate(self.weights_output):
			_sum = 0
			for j, values_hidden in enumerate(self.hidden_output):
				_sum += values_hidden * w
			output.append(_sum)

		return self.activate(output)

	def activate(self, sum):
		"""Output either +1 or -1

		This is the output filter. Will replace with a sigmoid function soon.
		"""

		if sum > 0:
			return 1
		else:
			return -1


	def train(self, inputs):
		"""Train the network against known data

		"""

		self.X = inputs[:,[0,1]] 	#Features
		self.y = inputs[:,2] 		#Answers
		
		for j in range(len(self.X)):
			guess = self.feed_forward(self.X[j])
			error = self.y[j] - guess
			#print "(x, y, guess,answer,error): (%s, %s, %s, %s)" %(self.X[j], guess, self.y[j], error)

			## REPLACE WITH GRADIENT ASCENT METHOD
			#for i, w in enumerate(self.weights[:]):
			#	self.weights[i] += self.rate * error * self.X[j][i]

		#print "Nueron Weights: "
		#print self.weights

class Trainer():
	def __init__(self, length):
		"""Create a training data set with points above and below a line

		"""

		self.x_limits = [-10,10]
		self.y_limits = [-10,10]

		self.training = np.zeros((length, 3))

		for i in range(length):
			x = randint(self.x_limits[0], self.x_limits[1])
			y = randint(self.y_limits[0], self.y_limits[1])
			answer = 1
			if y < self.f(x):
				answer = -1

			self.training[i] = [x,y,answer]


	def plot(self, inputs):
		"""Plot the inputs

		"""

		training_vals = zip(*self.training)
		x_training = training_vals[0]
		y_training = training_vals[1]
		
		answer_training = training_vals[2]

		plt.scatter(x_training, y_training, c=answer_training, s=200, cmap=cm.gray)
		plt.plot(self.x_limits, 
			     [self.f(self.x_limits[0]), self.f(self.x_limits[1])], 
			     c='black', 
			     lw=5)
		plt.xlim((self.x_limits[0], self.x_limits[1]))
		plt.ylim((self.y_limits[0], self.y_limits[1]))
		plt.show()

	def f(self, x):
		return 2*x+1



trainer = Trainer(1000)

net = NeuralNet()
net.create_network(2, 0)
net.train(trainer.training)

inputs = []
outputs = []
for i in range(100):
	input_vals = [randint(-10,10), randint(-10,10)]
	inputs.append(input_vals)
	outputs.append(net.feed_forward(input_vals))

plt.scatter(zip(*inputs)[0], zip(*inputs)[1], c=outputs, s=200, cmap=cm.gray)
plt.plot([-10,10], [-2*10+1, 2*10+1], lw=5, c='black')
plt.show()


