### DATA PROCESSING ###

'''
3-dimensional space - each row is a description of a cell nucleus in three dimensions

radius1 = mean radius
radius2 = radius SE
radius3 = worst radius
'''

import csv
import numpy as np
import pandas as pd

'''
data = pd.read_csv('../dataset/breast-cancer-wisconsin.data')
print("data")
print(data)

data2 = pd.read_csv('../dataset/Index')
print("data2")
print(data2)

data4 = pd.read_csv('../dataset/wpbc.data')
print("data4")
print(data4)
'''

header_list = ['id', 'diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1',
               'compactness1', 'concavity1', 'concave-points1', 'symmetry1', 'fractal-dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2',
               'compactness2', 'concavity2', 'concave-points2', 'symmetry2', 'fractal-dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3',
               'compactness3', 'concavity3', 'concave-points3', 'symmetry3', 'fractal-dimension3']
data = pd.read_csv('../dataset/wdbc.data',
                   names=header_list)
print(data)
# print(data['id'])  # column id
# print(data.iloc[1]) #row 1
