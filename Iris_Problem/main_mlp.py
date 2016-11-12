# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 23:16:44 2016

@author: PeDeNRiQue
"""
import random
import math
import numpy as np

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
    return file_array
    
def separate_input_target(data):
    inputs  = [d[:3] for d in data] 
    targets = [np.array(d[-1]) for d in data] 
    return inputs, targets
    #print(targets)

class Weight:
    def __init__(self,value=0):
        self.value = value      
        self.momentum = 0

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
        value = self.linear_combination(entries)
        self.output = self.activation(value)
        return self.output
    
    def calculate_error(self,entries,target):
        value = self.calculate_output(entries)
        error = math.pow(value,2)/2
        return error        
    
    def update_hidden_weights(self,front_deltas,front_weights,entry):    
        pass
        
    def update_output_weights(self,inputs,target):
        
        self.new_weights = []
        
        for i in range(len(self.weights)):
            #print(self.weights[i].value,self.learn_rate,target,self.output,inputs[i])            
            
            new_weight = Weight(self.weights[i].value + (self.learn_rate  * (target-self.output)) * inputs[i])
            self.new_weights.append(new_weight)  
            #print("NOVO PESO",new_weight.value)
            #input("PRESS ENTER TO CONTINUE.")
            
    def update_weights(self):
        #print("ATUALIZOU PESOS")
        self.weights = self.new_weights
        
        
            
class Layer:
    
    def __init__(self,neurons,name):
        self.name = name
        self.neurons = neurons
        
    def calculate_output(self,entries):
        self.entries = entries
        results = []
        for neuron in self.neurons:
            result = neuron.calculate_output(entries)
            results.append(result)
        return results
 
    def calculate_error(self,targets,entries):
        pass
    
    def get_deltas(self):
        deltas = [neuron.delta for neuron in self.neurons]
        return deltas
    
    def get_weights(self):
        weights = [neuron.get_weights_values() for neuron in self.neurons]
        return weights
        
    def update_output_weights(self,targets):
        
        for n in self.neurons:
            n.update_output_weights(self.entries,targets)
        
        
    
    def update_hidden_weights(self,front_layer):
        pass
    
    def update_weights(self):
        for n in self.neurons:
            #print("ATULIZANDO PESOS")
            n.update_weights()
            
class Network:
    
        
    def __init__(self,topology):        
        topology.reverse()

        self.layers = []
        for x in range(len(topology)-1):
            #print(x,topology[x],topology[x+1])
            neurons = self.initialize_neurons(topology[x+1],topology[x])
            
            if(x == 0):#CAMADA DE SAIDA
                layer = Layer(neurons,"saida")
            elif(x == len(topology)-2):
                layer = Layer(neurons,"entrada")
            else:
                layer = Layer(neurons,"escondida")
            self.layers.append(layer)
            
        self.layers.reverse()#COLOCA AS CAMADAS CONFIGURADAS NA ORDEM NORMAL
    
    def initialize_neurons(self,n_entries,n_neurons):    
        #weights = [[Weight(0.5) for x in range(n_entries) ] for y in range(n_neurons)]
        
        neurons = []
        for x in range(n_neurons):
            weights = []
            for y in range(n_entries+1):#MAIS UM POR CAUSA DO PESO DO BIAS
                weights.append(Weight(0.5))
                
            neuton = Neuron(weights)
            neurons.append(neuton)
        return neurons  
    
    #FUNCAO RECEBE UMA ENTRADA DE CADA VEZ
    def execute(self,inputs,targets):
        for entry,target in zip(inputs,targets): 
            while True:
                entry_t = entry
                for i in range(len(self.layers)-1):
                    entry_t = self.layers[i].calculate_output(entry_t)    
                    
                outputs = self.layers[-1].calculate_output(entry_t)
                
                error = math.pow(outputs - target,2)/2            
                #print(outputs,target,error)
                
                if(error > 0.01):
                    self.bakcpropagation(target)
                    
                    for layer in self.layers:
                        layer.update_weights()
                else:
                    break
                
      
     #PASSA TODA A BASE DE TREINAMENTO
    def trainning(self,inputs,targets):
        
        for x in range(100):
            self.execute(inputs,targets)
        

    def test(self, inputs,targets):
        
        results = []
        for entry,target in zip(inputs,targets):
            entry_t = entry
            for i in range(len(self.layers)):
                entry_t = self.layers[i].calculate_output(entry_t) 
            print("saida",entry_t)
            
    def bakcpropagation(self,targets):
        self.layers[-1].update_output_weights(targets)
        
        

    
    
if __name__ == "__main__":
    filename = "samples.txt"
    #AS ENTRADAS DEVEM POSSUIR O VALOR DO BIAS 1 (POSITIVO) NA PRIMEIRA COLUNA
    data = load_data(filename," ")
    data = insert_bias(data)
    topology = [2,1]
    
    #inputs = [[1, 0.05, 0.1, 0.2]]
    #targets = [[0.01, 0.99]]    
    
    inputs,targets = separate_input_target(data)
    
    net = Network(topology)
    

    net.trainning(inputs,targets)
    net.test(inputs,targets)
    
    
    
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