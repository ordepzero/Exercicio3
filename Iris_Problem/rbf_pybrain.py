from sklearn.cluster import KMeans
import numpy as np
import math
import random
from pybrain.datasets  import ClassificationDataSet
from pybrain.structure import LinearLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer

def logistic(value):
    #print("_",value)
    return 1/(1+math.exp(-value))

def linear_function(value):
    if(value > 1):
        return 1
    elif(value < 0):
        return 0
    else:
        return value
    
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
    

def normalize_data(f,has_target=True):
    
    x = np.array(f)
    x_normed = (x - x.min(axis=0))/ (x.max(axis=0) - x.min(axis=0))
    
    #SUBSTITUIO OS VALORES ALVO DA ULTIMA COLUNA NOS DADOS NORMALIZADOS
    if(has_target):
        x_normed[:,-1] = f[:,-1]

    return x_normed
    
def mean_sq_dist(center,entries):
    result = sum([math.pow(d-k,2) for d,k in zip(center,entries)])
    return result

def weighted_sum(inputs, weights,variations):
    results = []
    for weight,var in zip(weights,variations):
        #print("<",inputs, weight,var,">")
        result = radial_basis_function(sum(inputs * weight),var)
        results.append(result)
        #print(inputs , x,inputs * x,sum(inputs * x),result)
    return results
    
def radial_basis_function(u,variation):
    u = u * u;
    variation  = variation * variation
    value = u / (2*variation)
    return math.exp(-value)


def calculate_variance(points,center):
    total = 0
    for i in range(len(points)):
        for j in range(len(points[i])):            
            total += math.pow(points[i][j]-center[j],2)
            #print(points[i][j],center[j],points[i][j]-center[j])
    return total/len(points)

def caculate_error(inputs,weights,outputs, targets):
    #print(inputs,weights,outputs, targets)
    
    new_weights = []
    
    for i in range(len(targets)):    
        delta = (targets[i]-outputs[i])*(outputs[i] *(1 - outputs[i]))        
        nws = []
        for j in range(len(weights[i])):
            w = weights[i][j] + (0.1 * delta * inputs[j])
            nws.append(w)    
            
        new_weights.append(nws) 
        
    return new_weights
    

def convert(values):
    position = 0;
       
    for i in range(len(values)):
        if(values[i] > values[position]):
            position = i
    result = [0]*3
    result[position] = 1
    return result
    
def is_equal(outputs, targets):
    for i in range(len(outputs)):
        if(outputs[i] != targets[i]):
            return False
    return True

def order_data(values): 
    new_order = []       
    while True:
        position = random.randint(0, len(values)-1)
        #print(len(values),position)    
        #print(values[position])
        new_order.append(values[position])
        values = np.delete(values, position, 0)
        
        #print(".",t)
        if(len(values) == 0):
            break    
    return new_order
    
def classification(targets,outputs):
    cont = 0    
    for i in range(len(targets)):
        if(targets[i] == outputs[i]):
            cont = cont + 1
    result = cont / len(targets)
    return result
    
if __name__ == "__main__":
    
    n_classes = 3   
    n_output = 3
    output_weights = [[random.random() for y in range(n_classes+1)] for x in range(n_output)]
    
    dic = {'Iris-setosa\n': 0, 'Iris-versicolor\n': 1, 'Iris-virginica\n': 2}    
    
    filename = "iris2.txt"
    file = read_file(filename)
    file = change_class_name(file,dic)
    file = str_to_number(file)
    file_array = np.array(file)
    data = normalize_data(file_array)
    #print(data)
    #data = np.array(order_data(data))
    #print("#####")
    #print(data)
    #print(len(data))
    data_test = data[90:150]
    data = data[:90]
     
    results = [[0 for x in range(2)] for y in range(n_classes)] 
    kmeans = KMeans(n_clusters = n_classes, random_state=0).fit(data[:,:-1])

    #DETERMINANDO OS CENTROS
    centers = kmeans.cluster_centers_
    
    points = [[] for x in range(n_classes)]
    #CALCULANDO A VARIANCIA DE CADA UMA DAS FUNÇÕES
    for x in range(len(data)):
        positions = kmeans.predict([data[x][:-1]])
        position = positions[0]
        results[position][0] = results[position][0] + 1
        MSD = mean_sq_dist(centers[position],data[x][:-1] )
        results[position][1] = results[position][1] + MSD 
        points[position].append(data[x][:-1])
        
    #CALCULO DA VARIANCIA DE CADA FUNÇÃO
    variations = [0 for y in range(n_classes)]
        #print(statistics.pvariance(points[x]))
        #variations[x] = results[x][1] / results[x][0]
    
        
    
    #print("---",centers)
    
    variations = [calculate_variance(point,center) for point,center in zip(points,centers)]    
    #print(variations)
    entries = []
    for d in range(len(data)):
        entry = weighted_sum(data[d][:-1],centers,variations)
        entries.append(entry)
        #print(entry)
    #print(len(entries))
    
    
    train_data = ClassificationDataSet(n_classes, 1,nb_classes=n_classes)
    test_data = ClassificationDataSet(n_classes, 1,nb_classes=n_classes)   
    
    for n in range(0, len(entries)):
        train_data.addSample( entries[n], [data[n][-1]])
        
    entries = []
    for d in range(len(data_test)):
        entry = weighted_sum(data_test[d][:-1],centers,variations)
        entries.append(entry)
        
    for n in range(0, len(entries)):
        test_data.addSample( entries[n], [data_test[n][-1]])

    train_data._convertToOneOfMany( )
    test_data._convertToOneOfMany( )
    
    network = FeedForwardNetwork()
    inLayer = LinearLayer(3)    
    outLayer = LinearLayer(3)
    network.addInputModule(inLayer)
    network.addOutputModule(outLayer)
    in_to_out = FullConnection(inLayer, outLayer)
    network.addConnection(in_to_out)
    
    
    network.sortModules()

    trainer = BackpropTrainer( network, dataset=train_data, verbose=False)
    for i in range(1):
        trainer.trainEpochs(1000)
    
    result = trainer.testOnClassData(test_data,return_targets=True)
    result = classification(result[1],result[0]) 
    print(result)
    
    
    
    
    
    
    