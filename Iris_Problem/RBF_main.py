# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:08:04 2016

@author: PeDeNRiQue
"""
import math
import random
import numpy as np
from sklearn.cluster import KMeans

def read_file(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([x for x in line.split(",")])   
    return np.array(array);
    
def normalize_data(f,has_target=True):
    
    x = np.array(f)
    x_normed = (x - x.min(axis=0))/ (x.max(axis=0) - x.min(axis=0))
    
    #SUBSTITUIO OS VALORES ALVO DA ULTIMA COLUNA NOS DADOS NORMALIZADOS
    if(has_target):
        x_normed[:,-1] = f[:,-1]

    return x_normed    
    
def change_class_name(data,dic):
    for x in range(len(data)):
        data[x][-1] = dic[data[x][-1]]
    return data

def str_to_number(data):
    return[[float(j) for j in i] for i in data]

def convert(values):
    position = 0;
       
    for i in range(len(values)):
        if(values[i] > values[position]):
            position = i
    result = [0]*3
    result[position] = 1
    return result
    
class Neuron:
    
    def __init__(self,weights,label):
        self.weights = weights
        self.label = label
        
    def execute(self,inputs,targets):
        for cont in range(20):
            results = []
            for i,target in zip(inputs,targets):
                temp = [1]
                temp.extend(i)
                #print(temp, np.transpose(self.weights))
                result = np.dot(temp, np.transpose(self.weights))
                results.append(result)
                delta = [ 0.1 * tp for tp in temp]
                
                if (target != self.label and result >= 0):
                    #self.weights = self.weights - delta
                    self.weights = [wei - dlt  for wei,dlt in zip(self.weights, delta)]
                if (target == self.label and result < 0):
                    self.weights = [wei + dlt  for wei,dlt in zip(self.weights, delta)]
                    
        return results
        #print(self.weights)
        
    def test(self,inputs):
        results = []
        for i in inputs:
            temp = [1]
            temp.extend(i)
            #print(temp, np.transpose(self.weights))
            result = np.dot(temp, np.transpose(self.weights))
            results.append(result)
        return results

class Neuro_Basis_Radial(Neuron):
    def __init__(self,weights):
        self.weights = weights
        
    def execute(self,inputs):
        pseudo_samples = []
        for i in inputs:
            new_sample = []
            for weight,ii in zip(self.weights,i):
                new_sample.append(ii - weight)
            new_sample = np.array(new_sample)
            pseudo_samples.append(math.exp(-np.dot(new_sample, new_sample.T)/2))
        return pseudo_samples
        
    def activation_function(self,u):
        u = u * u;
        variation  = self.variation
        value = u / (2*variation)
        return math.exp(-value)
    
    
class RBF_Network:
    
    def __init__(self,num_hid_nr, num_out_nr):
        self.num_hid_nr = num_hid_nr
        self.num_out_nr = num_out_nr
        pass
    
    def train(self,data):
        kmeans = KMeans(n_clusters = self.num_hid_nr).fit(data_train["input"])
        centers = kmeans.cluster_centers_        
         
        
        #variations = self.calculate_variations(data,centers,kmeans)
        
        self.hidden_layer = [Neuro_Basis_Radial(centers[n]) for n in range(self.num_hid_nr)] 
        
        entries = [neuron.execute(data["input"]) for neuron in self.hidden_layer] 
        pseudo_samples = np.transpose(entries)
        
        #+1 por causa do bias
        weights = [[random.uniform(-10, 10) for n in range(self.num_hid_nr+1) ] for n in range(self.num_out_nr)]
        self.output_layer = [Neuron(weights[n],n) for n in range(self.num_out_nr)] 
        
        outputs = [neuron.execute(pseudo_samples,data["target"]) for neuron in self.output_layer] 
        outputs = np.transpose(outputs)
        #print(len(outputs))
        #[print(convert(out),target) for out,target in  zip(outputs,data["target"])]
        correct = 0
        for output,target in zip(outputs,data["target"]):            
            value = np.dot(convert(output), [0,1,2])
            if(value ==  target):
                correct += 1
        print("Acurácia treino:",correct/len(data["input"]))
     
    def test(self,data):
        #outputs = [neuron.test(data["input"]) for neuron in self.output_layer] 
        entries = [neuron.execute(data["input"]) for neuron in self.hidden_layer] 
        pseudo_samples = np.transpose(entries)
        
        outputs = [neuron.test(pseudo_samples) for neuron in self.output_layer] 
        outputs = np.transpose(outputs)
        correct = 0
        for output,target in zip(outputs,data["target"]):            
            value = np.dot(convert(output), [0,1,2])
            if(value ==  target):
                correct += 1
        print("Acurácia test:",correct/len(data["input"]))
    
if __name__ == "__main__":
    
    NUMBER_OF_CENTERS = 4
    NUMBER_OF_CLASSES = 3
    
    dic = {'Iris-setosa\n': 0, 'Iris-versicolor\n': 1, 'Iris-virginica\n': 2}    
    
    filename = "iris2.txt"
    file = read_file(filename)
    file = change_class_name(file,dic)
    file = str_to_number(file)
    file_array = np.array(file)
    data = normalize_data(file_array)
    data = {"input": file_array[:,:-1], "target":file_array[:,-1]}
    data_test = {"input": data["input"][90:150], "target":data["target"][90:150]} #data[90:150]
    data_train = {"input": data["input"][:90], "target":data["target"][:90]}  
    
    #print(data)
    
    rbf_net = RBF_Network(NUMBER_OF_CENTERS,NUMBER_OF_CLASSES)
    
    rbf_net.train(data_train)
    
    rbf_net.test(data_test)