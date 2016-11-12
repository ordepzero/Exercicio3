# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 04:39:23 2016

@author: PeDeNRiQue
"""

import math

L = [[1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12], [13, 14, 15, 16, 17], [18, 19, 20]]
features, classes = zip(*[(s[:-1], [s[-1]]) for s in L])

print(L[:2])
#print(L[2:])

v1 = 0.5370495669980353
v2 = 0

print(math.pow(v1-v2,2)/2)

x1 = [0]*4
x2 = [0]*5

if(x1 == x2):
    print("IGUAL")