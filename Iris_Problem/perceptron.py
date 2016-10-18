# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 07:52:30 2016

@author: PeDeNRiQue
"""
import math
import random
import numpy as np

class Weight:
    def __init__(self):
        self.value = 0.0        

class Neuron:
    def __init__(self,weights):
        self.weights = weights
        self.learn_rate = 0.5
        
    def get_weights_values(self):
        return [w.value for w in self.weights]

    def linear_combination(self):
        return sum([(entry*weight.value) for entry,weight in zip(self.entries,self.weights)])
    
    def activation(self,value):
        #return (math.tanh(value))
        if(value > 0):
            return 1
        else:
            return 0
    
    def calculate_output(self):
        return self.activation(self.linear_combination())
    
    def calculate_error(self):
        out = self.calculate_output()
        print("=",self.target,out)
        return (self.target-out)
    
    def update_weights(self,error):        
        delta = self.learn_rate*error
        
        for index in range(len(self.weights)):
            #print(self.weights[index].value)
            self.weights[index].value = self.weights[index].value+(delta*self.entries[index])
        
        
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
            
if __name__ == "__main__":   
    
    
    w1 = Weight()
    w2 = Weight()    
    w3 = Weight() 
    
    bias = 1
    entries = [[bias,0,0],[bias,0,1],[bias,1,0],[bias,1,1]] 
    targets = [[0],[0],[0],[1]]
    
    weights = [w1,w2,w3]
    n = Neuron(weights)
    

    cont = True
    while cont:
        cont = False
        for index in range(len(entries)):
            print("<",index)
            if(n.execute(entries[index],targets[index][0]) == 1):
                cont = True
            print(">")






