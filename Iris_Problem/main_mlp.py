# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 23:16:44 2016

@author: PeDeNRiQue
"""

import math
import numpy as np

class Weight:
    def __init__(self):
        self.value = 0
 
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
    
if __name__ == "__main__":
    
    data = put_file_int_array("samples.txt"," ")
    data = separate_input_target(data,1)
    
    print(data)
