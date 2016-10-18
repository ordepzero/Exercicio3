# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 23:16:44 2016

@author: PeDeNRiQue
"""
import random
import math
import numpy as np

class Weight:
    def __init__(self,value=0):
        self.value = value        

class Neuron:
    def __init__(self,weights):
        self.weights = weights
        self.learn_rate = 0.5
        
    def get_weights_values(self):
        return [w.value for w in self.weights]

    def linear_combination(self,entries):
        return sum([(entry*weight.value) for entry,weight in zip(entries,self.weights)])
    
    def activation(self,value):
        return 1/(1+math.exp(-value))#LOGISTICA
        #return (math.tanh(value))#TANGENTE HIPERBOLICA
    
    def calculate_output(self,entries):
        self.output = self.activation(self.linear_combination(entries))
        return self.output
    
    def calculate_error(self,entries,target):
        output = self.calculate_output(entries)
        self.error = (math.pow(target-output,2))/2
        return self.error
    
    def update_weights(self,error):        
        result = self.learn_rate*error
        
        for index in range(len(self.weights)):
            #print(self.weights[index].value)
            self.weights[index].value = self.weights[index].value+(result*self.entries[index])
        
    def update_output_weights(self,entries,target):
        partial_error_total = self.output - target
        derivative = self.output * (1 - self.output)
        self.delta = partial_error_total * derivative
        #print(partial_error,derivative,entries)
        
        partial_erro = [self.delta * entry for entry in entries]
        
        cont = 0
        for weight in self.weights:
            weight.value = weight.value - self.learn_rate * partial_erro[cont]
            cont = cont + 1
            
        print(self.get_weights_values())
        
    def execute(self,entries,target):
        self.entries = entries
        self.target = target
        
        error = self.calculate_error()
        if(error != 0):
            self.update_weights(error)
            print("(1)",self.get_weights_values())
            return 1
        else:
            print("(0)",self.get_weights_values())
            return 0
            
class Layer:
    
    def __init__(self,entries,neurons):
        self.entries = entries
        self.neurons = neurons
        
    def calculate_output(self):
        outputs = [neuron.calculate_output(self.entries) for neuron in self.neurons]
        return outputs 
 
    def calculate_error(self,targets):
        outputs = [neuron.calculate_error(self.entries,target) for neuron,target in zip(self.neurons,targets)]
        self.error_total = sum(outputs)        
        return self.error_total 
    
    def update_output_weights(self,targets):
        [neuron.update_output_weights(self.entries,target) for neuron,target in zip(self.neurons,targets)]
        
if __name__ == "__main__":
    
    inputs = [1, 0.05, 0.1]
    targets = [[0.01, 0.99]]
    
    wb1_1 = Weight(0.35)
    w1 = Weight(0.15)
    w2 = Weight(0.20)
    wb1_2 = Weight(0.35)
    w3 = Weight(0.25)
    w4 = Weight(0.30)
    wb2_1 = Weight(0.60)    
    w5 = Weight(0.40)
    w6 = Weight(0.45)
    wb2_2 = Weight(0.60) 
    w7 = Weight(0.50)
    w8 = Weight(0.55)

    weights1 = [wb1_1,w1,w2]
    weights2 = [wb1_2,w3,w4]
    weights3 = [wb2_1,w5,w6]    
    weights4 = [wb2_2,w7,w8]

    h1 = Neuron(weights1)
    h2 = Neuron(weights2)
    o1 = Neuron(weights3)
    o2 = Neuron(weights4)
    
    neurons1 = [h1,h2]
    neurons2 = [o1,o2]



    layer1 = Layer(inputs,neurons1)
    layer2 = Layer([1]+layer1.calculate_output(),neurons2)
    final_error = layer2.calculate_error(targets[0])

    if(final_error > 0.1):
        layer2.update_output_weights(targets[0])

    print(layer2.error_total)
'''
def separate_input_target(data,first_target_index):
    database = {}
    database["input"] = data[:,:-first_target_index]
    database["target"] = data[:,-first_target_index:]
    
    return database
    
 
def put_file_int_array(filename,separator):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([float(x) for x in line.split(separator)])   
    return np.array(array);
'''