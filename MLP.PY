import numpy
import scipy.special
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.pyplot 
import datetime

# neural network definition
class neuralNetwork:
    
    #init neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #set number of nodes in each input, hidden and output layer
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes
        
        #learning rate
        self.lr = learningrate
        
        # linkweight matrices, wih and who
        self.wih = numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        
        #activation function (sigmoid function expit)
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
    
    # train the neural network
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # calculate signals emerging from hidden layer
        final_outputs = self.activation_function(final_inputs)
        
        #output layer error is target - actual
        output_errors = targets - final_outputs
        #hidden layer error is the outputs error
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        #update the weights for the links between the hidden and output layers
        self.who += self.lr*numpy.dot(output_errors*final_outputs*(1.0-final_outputs),numpy.transpose(hidden_outputs))
        
        #update the weights for the links between the input and hidden layers
        self.wih += self.lr*numpy.dot(hidden_errors*hidden_outputs*(1.0-hidden_outputs),numpy.transpose(inputs))
        
        pass
    
    #query the neural network
    def query(self,inputs_list):
        #convert inputs_list to 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # calculate signals emerging from hidden layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
        
rates = []
performance = []
#for nn in frange(60,200,20):
n= neuralNetwork(784,200,10,0.01)
#print("NN",nn)
# start time
starttime = datetime.datetime.now()