from sklearn.cluster import KMeans
import numpy as np
import math
import random
from pybrain.datasets  import ClassificationDataSet

def logistic(value):
    #print("_",value)
    return 1/(1+math.exp(-value))
    
def linear_function(value):
    if(value > 1):
        return 1
    elif(value < 0):
        return 0
        
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
    
def mean_sq_dist(center,entries):
    result = sum([math.pow(d-k,2) for d,k in zip(center,entries)])
    return result


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
def convert_to_int(values):
    for i in range(len(values)):
        if(i == 1):
            return i
            
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
        #delta = (targets[i]-outputs[i])*(outputs[i] *(1 - outputs[i]))      
        delta = targets[i]-outputs[i]
        nws = []
        for j in range(len(weights[i])):
            w = weights[i][j] + (0.2 * delta * inputs[j])
            nws.append(w)    
            
        new_weights.append(nws) 
        
    return new_weights
    
def pseudo_samples(data):
    entries = []
    for d in range(len(data)):
        entry = weighted_sum(data[d][:-1],centers,variations)
        entries.append(entry)
        
    return entries
    
if __name__ == "__main__":
    
    n_classes = 4 
    n_output = 3
    output_weights = [[random.random() for y in range(n_classes+1)] for x in range(n_output)]
    
    dic = {'Iris-setosa\n': 0, 'Iris-versicolor\n': 1, 'Iris-virginica\n': 2}    
    
    filename = "iris2.txt"
    file = read_file(filename)
    file = change_class_name(file,dic)
    file = str_to_number(file)
    file_array = np.array(file)
    #data = normalize_data(file_array)
    data = file_array
    
    data_test = data[90:150]
    data = data[:90]    
    
    #print(data[:,:-1])
    
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
    
    variations = [calculate_variance(point,center) for point,center in zip(points,centers)]    
        
    print(centers,variations)
    entries = pseudo_samples(data)
    
    
    train_data = ClassificationDataSet(n_classes, 1,nb_classes=n_output)
    
    for n in range(0, len(entries)):
        train_data.addSample( entries[n], [data[n][-1]])

    train_data._convertToOneOfMany( )
    
    
    for epochs in range(6):
        rights = 0
        cont = 0
        for i in range(len(train_data["input"])):
            #print("<")
            results = []
            for j in range(len(output_weights)):
                add_bias = [1]
                add_bias.extend(train_data["input"][i])
                #print(add_bias)
                total = 0
                
                for x in range(len(add_bias)):#CALCULANDO CADA SAÍDA
                    total += add_bias[x] * output_weights[j][x]
                    #print("<",entries[i][x],output_weights[j][x],">")
                result = linear_function(total)
                results.append(result)
                #print(add_bias,result,train_data["target"][i][j])
                #output_weights[j] = caculate_error(add_bias,output_weights[j],result, train_data["target"][i][j])
                #print(i,result,train_data["target"][i][j])
                
            r = is_equal(convert(results),train_data["target"][i])
            if(r == True):
                rights += 1
            else:                
                cont += 1
                if(epochs < 4):
                    output_weights = caculate_error(add_bias,output_weights,results, train_data["target"][i])

            #print(i,convert(results),train_data["target"][i],rights)
        print("Acertos:", rights,"Ajustes:", cont)
     
     
    entries = pseudo_samples(data_test)     
        
    
    rights = 0
    for i in range(len(entries)):
        #print("<")
        results = []
        for j in range(len(output_weights)):
            add_bias = [1]
            add_bias.extend(entries[i])
            #print(add_bias)
            total = 0
            
            for x in range(len(add_bias)):#CALCULANDO CADA SAÍDA
                total += add_bias[x] * output_weights[j][x]
                #print("<",entries[i][x],output_weights[j][x],">")
            result = linear_function(total)
            results.append(result)
            #print(add_bias,result,train_data["target"][i][j])
            #output_weights[j] = caculate_error(add_bias,output_weights[j],result, train_data["target"][i][j])
            #print(i,result,train_data["target"][i][j])
        conv = convert_to_int(convert(results))
        if(conv == data_test[i][-1]):
            rights += 1
      #print(i,convert(results),train_data["target"][i],rights)
    print("Acertos:", rights)
    
    
    