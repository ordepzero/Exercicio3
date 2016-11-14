# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 04:39:23 2016

@author: PeDeNRiQue
"""
import numpy as np
import math
import random

t = np.array([
 [ 0.27777778,  0.70833333,  0.08474576,  0.04166667 , 0.        ],
 [ 0.19444444,  0.54166667,  0.06779661,  0.04166667,  0.        ],
 [ 0.75      , 0.5        , 0.62711864 , 0.54166667  ,1.        ],
 [ 0.58333333 , 0.5        , 0.59322034,  0.58333333 , 1.        ],
  [ 0.69444444,  0.41666667,  0.76271186,  0.83333333 , 2.       ],
 [ 0.38888889 , 0.20833333 , 0.6779661  , 0.79166667 , 2.        ],
 [ 0.41666667 , 0.33333333 , 0.69491525 , 0.95833333 , 2.        ]])
 
while True:
    position = random.randint(0, len(t)-1)
    print(len(t),position)    
    print(t[position])
    t = np.delete(t, position, 0)
    
    #print(".",t)
    if(len(t) == 0):
        break