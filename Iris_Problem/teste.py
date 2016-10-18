# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 04:39:23 2016

@author: PeDeNRiQue
"""

import math

out = 0.75136507
target = 0.01

delta = -(target- out)*out*(1-out)

print(delta)