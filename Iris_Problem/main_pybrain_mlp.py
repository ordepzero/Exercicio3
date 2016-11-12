# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 05:57:41 2016

@author: PeDeNRiQue
"""

import numpy as np

from pybrain.utilities import percentError
from pybrain.datasets  import ClassificationDataSet
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer

def put_file_int_array(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([float(x) for x in line.split()])   
    return np.array(array);
    
    
def read_file(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([x for x in line.split(",")])   
    return np.array(array)
    
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
    
    P_TRAIN = 0.75
    size_total = len(data)
    size_train = int(size_total*P_TRAIN)
    size_each_class = int(size_train / len(n_class))
    
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
    
    
def train_test_data(data,p_train=0.75):
    '''
    size_total = len(data)
    size_train = int(size_total*p_train)
    train = data[:size_train]
    test  = data[size_train:]
       
    t = np.split(data,[size_train])   
       
    print("Size:",len(t[1]))
    '''
    return data,data
    
def convert_to_bin_class(value):
    if(value == 1.):
        return [0, 0, 1]
    elif(value == 2.):
        return [0, 1, 0]
    else:
        return [1, 0, 0]
        
def classification(targets,outputs):
    cont = 0    
    for i in range(len(targets)):
        if(targets[i] == outputs[i]):
            cont = cont + 1
    result = cont / len(targets)
    return result
            

def execute_mlp(n_neurons,data_size,learn_rate,momentum_rate,f):
    
    dic = {'Iris-setosa\n': 0, 'Iris-versicolor\n': 1, 'Iris-virginica\n': 2}    
    
    filename = "iris.txt"
    file = read_file(filename)
    file = change_class_name(file,dic)
    file = str_to_number(file)
    file_array = np.array(file)
    data = normalize_data(file_array)
    
    data = order_data(data)
    
    data = data[::-1]#INTERTENDO A ORDEM DOS ITENS    
    
    inputs = data[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    targets = data[:,-1] #COPIAR ULTIMA COLUNA
    
    train_data_temp,test_data_temp = train_test_data(data,data_size)
    
    train_data = ClassificationDataSet(4, nb_classes=3)#TAMANHO DA ENTRADA, NUMERO DE CLASSES
    test_data  = ClassificationDataSet(4, nb_classes=3)#TAMANHO DA ENTRADA, NUMERO DE CLASSES
    
    cont = 0
    for n in range(0, len(train_data_temp)):
        train_data.addSample( train_data_temp[n][:-1], [train_data_temp[n][-1]])
        #print(train_data.getSample(cont))
        #cont = cont + 1
    
    for n in range(0, len(test_data_temp)):
        test_data.addSample( test_data_temp[n][:-1], [test_data_temp[n][-1]])
    
    train_data._convertToOneOfMany( )
    test_data._convertToOneOfMany( )  
    '''
    print ("Number of training patterns: ", len(train_data))
    print ("Input and output dimensions: ", train_data.indim, train_data.outdim)
    print ("First sample (input, target, class):")
    print (test_data['input'][0], test_data['target'][0], test_data['class'][0])
    '''

    network = FeedForwardNetwork()

    inLayer = SigmoidLayer(train_data.indim)
    first_hiddenLayer = SigmoidLayer(n_neurons)
    second_hiddenLayer= SigmoidLayer(n_neurons)
    outLayer = SigmoidLayer(train_data.outdim)
    
    network.addInputModule(inLayer)
    network.addModule(first_hiddenLayer)
    network.addModule(second_hiddenLayer)
    network.addOutputModule(outLayer)
    
    in_to_hidden = FullConnection(inLayer, first_hiddenLayer)
    hidden_to_hidden = FullConnection(first_hiddenLayer,second_hiddenLayer)
    hidden_to_out = FullConnection(second_hiddenLayer, outLayer)
    
    network.addConnection(in_to_hidden)
    network.addConnection(hidden_to_hidden)
    network.addConnection(hidden_to_out)
    
    network.sortModules()
    
    trainer = BackpropTrainer( network, dataset=train_data, momentum=momentum_rate, verbose=False, weightdecay=learn_rate)
    
    for i in range(1):
        trainer.trainEpochs(1000)
    
    result = trainer.testOnClassData(test_data,return_targets=True)
    #result = classification(result[1],result[0])    
    print(result)
    f.write(str(result))
    f.flush()
    
    
    
if __name__ == "__main__":
    
    momentum = [0.75]
    learn_rate = [0.25]
    data_size = [75]  
    n_neurons = [10]
    
    f = open('teste.txt', 'a')    
    f.write("momentum\tlearn_rate\tdata_size\tn_neurons\tacuracia\n")
    

    for m in momentum:
        for lr in learn_rate:
            for ds in data_size:
                for nn in n_neurons:
                    for x in range(5):
                        line = str(m)+"\t"+str(lr)+"\t"+str(ds)+"\t"+str(nn)+"\t"
                        print(line)
                        f.write(line)
                        execute_mlp(nn,ds,lr,m,f)
                        f.write("\n")
                        a = 5
    
    
    f.close()