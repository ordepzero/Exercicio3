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
        #print("<<")
        #[print("LC",entry,weight.value) for entry,weight in zip(entries,self.weights)]
        #print(">>")        
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
    
    def update_hidden_weights(self,front_deltas,front_weights,entry):    
        delta_temp = sum([delta*weight for delta,weight in zip(front_deltas,front_weights)])
        
        self.delta = delta_temp * self.output * (1 - self.output) 
        
        cont = 0        
        updated_weights = []
        for weight in self.weights:
            new_weight = Weight(weight.value - self.learn_rate * self.delta * entry)
            updated_weights.append(new_weight)
            cont = cont + 1
        self.updated_weights = updated_weights
        
    def update_output_weights(self,entries,target):
        partial_error_total = self.output - target
        derivative = self.output * (1 - self.output)
        self.delta = partial_error_total * derivative
        #print(partial_error,derivative,entries)
        
        partial_erro = [self.delta * entry for entry in entries]
        
        cont = 0
        updated_weights = []
        for weight in self.weights:
            new_weight = Weight(weight.value - self.learn_rate * partial_erro[cont])
            updated_weights.append(new_weight)
            cont = cont + 1
        self.updated_weights = updated_weights
    
    def update_weights(self):
        self.weights = self.updated_weights
        
            
class Layer:
    
    def __init__(self,neurons,name):
        self.name = name
        self.neurons = neurons
        
    def calculate_output(self,entries):
        self.entries = entries
        outputs = [neuron.calculate_output(self.entries) for neuron in self.neurons]
        return outputs 
 
    def calculate_error(self,targets,entries):
        #print("ENTRIES",entries,self.name)
        self.entries = entries
        outputs = [neuron.calculate_error(self.entries,target) for neuron,target in zip(self.neurons,targets)]
        self.error_total = sum(outputs)        
        return self.error_total 
    
    def get_deltas(self):
        deltas = [neuron.delta for neuron in self.neurons]
        return deltas
    
    def get_weights(self):
        weights = [neuron.get_weights_values() for neuron in self.neurons]
        return weights
        
    def update_output_weights(self,targets):
        #print(self.name)
        [neuron.update_output_weights(self.entries,target) for neuron,target in zip(self.neurons,targets)]
    
    def update_hidden_weights(self,front_layer):
        front_deltas = front_layer.get_deltas()
        front_weights = np.array(front_layer.get_weights())
        
        #print(front_deltas,front_weights,len(self.neurons),self.entries)        
        
        cont = 1
        
        for neuron in self.neurons:
            for entry in self.entries:
                neuron.update_hidden_weights(front_deltas,front_weights[:,cont],entry)
            cont = cont + 1
    def update_weights(self):
        [neuron.update_weights() for neuron in self.neurons]
    
class Network:
    
        
    def __init__(self,topology):        
        topology.reverse()

        self.layers = []
        for x in range(len(topology)-1):
            print(x,topology[x],topology[x+1])
            neurons = self.initialize_neurons(topology[x+1],topology[x])
            
            if(x == 0):#CAMADA DE SAIDA
                layer = Layer(neurons,"saida")
            elif(x == len(topology)-2):
                layer = Layer(neurons,"entrada")
            else:
                layer = Layer(neurons,"escondida")
            self.layers.append(layer)
            
        self.layers.reverse()#COLOCOU AS CAMADAS CONFIGURADAS NA ORDEM NORMAL
    
    def initialize_neurons(self,n_entries,n_neurons):    
        #weights = [[Weight(0.5) for x in range(n_entries) ] for y in range(n_neurons)]
        
        neurons = []
        for x in range(n_neurons):
            weights = []
            for y in range(n_entries+1):#MAIS UM POR CAUSA DO PESO DO BIAS
                weights.append(Weight(0.5))
                
            neuton = Neuron(weights)
            neurons.append(neuton)
            
        for x in neurons:
            weights = x.get_weights_values()
            for y in weights:
                print(y, " ",end="")
            print("")
            
        return neurons  
    
    def execute(self,inputs,targets):
        #[print(layer.name) for layer in self.layers]
                     
        entries = [1]+self.layers[0].calculate_output(inputs)#CAMADA DE ENTRADA RECEBE COMO ENTRADA OS EXEMPLOS
        for x in range(1,len(self.layers)-1):
            entries = [1]+self.layers[x].calculate_output(entries)  
        
        final_error = self.layers[-1].calculate_error(targets,entries)#RECEBE OS TARGETS PARA CALCULAR O ERRO
        
        print("ERRO FINAL",final_error)
        if(final_error > 0.1):
            self.bakcpropagation(targets)
            
    def bakcpropagation(self,targets):
        #ORDEM DA CAMADA DE SAIDA PARA CAMADA DE ENTRADA
        self.layers.reverse()
        
        self.layers[0].update_output_weights(targets)
        #print("<",self.layers[0].name)
        for x in range(1,len(self.layers)): 
            #print(self.layers[x].name)
            self.layers[x].update_hidden_weights(self.layers[x-1])
            
        #RETIRNA A ORDEM NORMAL
        #print(">")
        self.layers.reverse()
        
        #SUBSTITUI OS PESOS ANTERIORES PELOS PESOS ATUALIZADOS            
        [layer.update_weights() for layer in self.layers]
        
if __name__ == "__main__":
       
    topology = [4,3,4,2]
    
    inputs = [[1, 0.05, 0.1, 0.2]]
    targets = [[0.01, 0.99]]    
    
    net = Network(topology)
    net.execute(inputs[0],targets[0])
    
    '''
    
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



    layer1 = Layer(neurons1)
    layer2 = Layer(neurons2)
    
    final_error = layer2.calculate_error(targets[0],[1]+layer1.calculate_output(inputs))

    if(final_error > 0.1):
        layer2.update_output_weights(targets[0])
        #print("DELTAS",layer2.get_deltas())
        layer1.update_hidden_weights(layer2)
        
        
    print(layer2.error_total)
    '''
    
    
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