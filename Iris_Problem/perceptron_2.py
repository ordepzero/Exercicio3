# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:43:52 2016

@author: PeDeNRiQue
"""
import numpy as np
import math


def read_file(filename,separator):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([x for x in line.split(separator)])   
    return np.array(array);

def str_to_number(data):
    return[[float(j) for j in i] for i in data]
    
def insert_bias(file_array):    
    
    data = [np.append([1],d) for d in file_array]
    #print(data)
    return data
    
def load_data(filename,separator):
    
    file = read_file(filename,separator)
    file = str_to_number(file)
    file_array = np.array(file)
    #return file_array
    #print(file_array)
    return insert_bias(file_array)

def separate_input_target(data):
    inputs  = [d[:3] for d in data] 
    targets = [np.array(d[-1]) for d in data] 
    return inputs, targets
    #print(targets)
    
def initialize_weights(size):
    return [0.5 for i in range(size)]

class Neuron:
    def __init__(self,weights):
        self.weights = weights
        self.learn_rate = 0.25
    
    def weighted_sum(self,inputs):
        #print(inputs,self.weights)
        total = 0 
        for i,w in zip(inputs,self.weights):
            total = total + (i*w)
        
        return total
        
    def activation(self,value):
        #return 1/(1+math.exp(-value))#LOGISTICA
        return (math.tanh(value))#TANGENTE HIPERBOLICA
        
    def calculate_output(self,inputs):
        self.output = self.weighted_sum(inputs)
        self.output = self.activation(self.output)
        return self.output
        
    def update_weights(self,inputs,targets):
        
        #print(inputs,targets)
        new_weights = []
        
        for i in range(len(self.weights)):
            new_weights.append(self.weights[i] + (self.learn_rate  * (targets-self.output)) * inputs[i])

        self.weights = new_weights;
        
        
        
if __name__ == "__main__":
    
    filename = "samples.txt"
    #AS ENTRADAS DEVEM POSSUIR O VALOR DO BIAS 1 (POSITIVO) NA PRIMEIRA COLUNA
    inputs,targets = separate_input_target(load_data(filename," "))
    weights = initialize_weights(len(inputs[0]))
    
    neuron = Neuron(weights)
    result = neuron.calculate_output(inputs[0])
    
    epoch = 0
    cont = 0
    while True:
        error_total = 0
        
        print("Epoch",epoch)
        for i in range(len(inputs)):
            while True:
                result = neuron.calculate_output(inputs[i])
                #print(i,result)
                error = math.pow(targets[i] - result,2)/2
                print("Error",targets[i],result,error)
                print(neuron.weights)
                if( error > 0.05):
                    neuron.update_weights(inputs[i],targets[i])
                else:
                    break
                cont = cont + 1
            error_total = error_total + error   
            
        epoch = epoch + 1
        mean_error = error_total / len(inputs)
        if(epoch >= 100 or mean_error < 0.001):
            print(mean_error)
            break
    
    
    for i in range(len(inputs)):
        result = neuron.calculate_output(inputs[i])
        print(i,result)
    
    
    
    
    
    