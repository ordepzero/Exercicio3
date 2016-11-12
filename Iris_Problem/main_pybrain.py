# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:19:13 2016

@author: PeDeNRiQue
"""

from pybrain.datasets  import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

import numpy as np

def convert(values):
    position = 0;
       
    for i in range(len(values)):
        if(values[i] > values[position]):
            position = i
    result = [0]*3
    result[position] = 1
    
    return np.array(result)

def read_file(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([x for x in line.split(",")])   
    return np.array(array);
    
def change_class_name(data,dic):
    for x in range(len(data)):
        data[x][-1] = dic[data[x][-1]]
    return data

def str_to_number(data):
    return[[float(j) for j in i] for i in data]
    
def order_data(data):
    #n_class = [1.,2.,3.] #VERSAO ORIGINAL
    n_class = [0.,1.,2.]
    data_alternated = []
    
    #print(size_train,size_each_class)
    
    index = 0
    for x in range(len(data)):   
        c = n_class[index]
        for i in data:            
            if(i[-1] == c):
                #print("IGUAL",i[-1],c,n_class[-1])
                i[-1] = i[-1] * -1
                data_alternated.append(i)
                if(c == n_class[-1]):
                    c = n_class[0]
                    index = -1
                
                index = index + 1
                c = n_class[index]

    data_alternated = np.array(data_alternated)
    data_alternated[:,-1] *= -1
    
    return data_alternated 

def normalize_data(f,has_target=True):
    
    x = np.array(f)
    x_normed = (x - x.min(axis=0))/ (x.max(axis=0) - x.min(axis=0))
    
    #SUBSTITUIO OS VALORES ALVO DA ULTIMA COLUNA NOS DADOS NORMALIZADOS
    if(has_target):
        x_normed[:,-1] = f[:,-1]

    return x_normed
    
def execute(data,learn_rate,momentum_rate,file_result,p_train):

    inputs = data[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    targets = data[:,-1] #COPIAR ULTIMA COLUNA
    
    
    train_data = ClassificationDataSet(4, 1,nb_classes=3)
    test_data = ClassificationDataSet(4, 1,nb_classes=3)
    
    size = int(len(inputs) * p_train)
    for n in range(0, size):
        #print(targets[n])
        train_data.addSample( inputs[n], [targets[n]])
        
    for n in range(size, len(inputs)):
        #print(targets[n])
        test_data.addSample( inputs[n], [targets[n]])
    
        
    train_data._convertToOneOfMany( )
    test_data._convertToOneOfMany( )
    
    fnn = buildNetwork(train_data.indim, 2, train_data.outdim)
    trainer = BackpropTrainer(fnn, train_data, learningrate=learn_rate,momentum=momentum_rate,verbose=False)
    
    epochs = 0
    for i in range(300):    
        epochs += 1
        trainer.train()  
        
    #print (trainer.testOnClassData())
    #print (trainer.testOnData())   
        
    cont = 0
    for test in test_data:
        r = fnn.activate(test[0])
        cls = convert(r)
        print(cls,test[1])
        if((cls == test[1]).all()):
            cont += 1
        
    
    print(cont)
    
    error = cont / len(test_data)
    
    line_result = str(momentum_rate)+"\t"+str(learn_rate)+"\t"+str(error)+"\t"+str(epochs)+"\t"+str(p_train)
    
    f.write(line_result+"\n")
    f.flush()
    
    

if __name__ == "__main__":
    
    f = open('resultados_2n.txt', 'a')    
    f.write("momentum\tlearn_rate\tacuracia\tepochs\tp_train\n")

    dic = {'Iris-setosa\n': 0, 'Iris-versicolor\n': 1, 'Iris-virginica\n': 2}    
    
    filename = "iris.txt"
    file = read_file(filename)
    file = change_class_name(file,dic)
    file = str_to_number(file)
    file_array = np.array(file)
    data = normalize_data(file_array)
    
    data = order_data(data)
    
    data = data[::-1]#INTERTENDO A ORDEM DOS ITENS    
    
    learn_rates = [0.1, 0.15, 0.2]
    momentums = [0.1, 0.15, 0.2]
    p_trains = [0.75, 0.8, 0.9]
    
    for p_train in p_trains:
        for learn_rate in learn_rates:
            for momentum in momentums:   
                for i in range(10):
                    execute(data,learn_rate,momentum,f,p_train)
        
        
        
    f.close()
        
        
        