### DATA PROCESSING ###

'''
3-dimensional space - each row is a description of a cell nucleus in three dimensions

radius_mean = mean radius
radius_SE = radius SE
radius_worst = worst radius
'''

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
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

    header_list = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                'compactness_mean', 'concavity_mean', 'concave-points_mean', 'symmetry_mean', 'fractal-dimension_mean', 'radius_SE', 'texture_SE', 'perimeter_SE', 'area_SE', 'smoothness_SE',
                'compactness_SE', 'concavity_SE', 'concave-points_SE', 'symmetry_SE', 'fractal-dimension_SE', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                'compactness_worst', 'concavity_worst', 'concave-points_worst', 'symmetry_worst', 'fractal-dimension_worst']
    data = pd.read_csv('../dataset/wdbc.data',
                    names=header_list)
    print(data)
    # print(data['id'])  # column id
    # print(data.iloc[1]) #row 1


    # Splitting data
    X = data.iloc[:, 2:].values
    y = data['diagnosis'].values
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.21, random_state=42)

    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3))
    model.fit(X_train, y_train)
    print(model.score(X_valid, y_valid))

if __name__ == '__main__':
    main()