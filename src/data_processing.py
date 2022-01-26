# data and idea from: http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29
### Just for practice - not my data


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
from mpl_toolkits import mplot3d
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


    ##### Diagnosis #####
    # b, m = data['diagnosis'].value_counts(dropna=False)
    # x = ['M', 'B']
    # y = [m, b]
    # plt.figure()
    # plt.bar(x, y)
    # plt.title('Diagnosis')
    # plt.xlabel('Diagnosis')
    # plt.ylabel('Count')
    # plt.tight_layout()
    # # plt.show()
    # # plt.savefig('diagnosis.png')

    ##### Mean Symmetry #####
    # plt.subplot()
    # symmetry_mean_M = data.loc[data['diagnosis'] == 'M']
    # symmetry_mean_B = data.loc[data['diagnosis'] == 'B']
    # plt.plot(symmetry_mean_B['symmetry_mean'].dropna(), 'b*')
    # plt.plot(symmetry_mean_M['symmetry_mean'].dropna(), 'r.', alpha=0.5)
    # plt.legend(['Benign', 'Malignant'])
    # plt.title('Mean Symmetry')
    # # # plt.xlabel('Latitude')
    # # # plt.ylabel('Longitude')
    # plt.tight_layout()
    # plt.show()
    # # # plt.savefig('LonLat.png')

    ##### Mean Radius #####
    # plt.subplot()
    # radius_mean_M = data.loc[data['diagnosis'] == 'M']
    # radius_mean_B = data.loc[data['diagnosis'] == 'B']
    # plt.plot(radius_mean_B['radius_mean'].dropna(), 'b*')
    # plt.plot(radius_mean_M['radius_mean'].dropna(), 'r.', alpha=0.5)
    # plt.legend(['Benign', 'Malignant'])
    # plt.title('Mean Radius')
    # # # plt.xlabel('Latitude')
    # # # plt.ylabel('Longitude')
    # plt.tight_layout()
    # plt.show()
    # # # plt.savefig('mean_radius.png')

    ##### Radius #####
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    radius_mean_M = data.loc[data['diagnosis'] == 'M']
    radius_mean_B = data.loc[data['diagnosis'] == 'B']
    ax.scatter3D(radius_mean_B['radius_mean'].dropna(), radius_mean_B['radius_SE'].dropna(), radius_mean_B['radius_worst'].dropna(), c=radius_mean_B['radius_worst'].dropna(), alpha=0.6, cmap='Blues')
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.scatter3D(radius_mean_M['radius_mean'].dropna(), radius_mean_M['radius_SE'].dropna(), radius_mean_M['radius_worst'].dropna(), c=radius_mean_M['radius_worst'].dropna(), alpha=0.3, cmap='Reds')
    plt.legend(['Benign', 'Malignant'])
    plt.title('Radius')
    # # plt.xlabel('Latitude')
    # # plt.ylabel('Longitude')
    # # plt.zlabel('idk')
    plt.tight_layout()
    plt.show()
    # # plt.savefig('radius.png')

    ##### Mean Texture #####
    # plt.subplot()
    # texture_mean_M = data.loc[data['diagnosis'] == 'M']
    # texture_mean_B = data.loc[data['diagnosis'] == 'B']
    # plt.plot(texture_mean_B['texture_mean'].dropna(), 'b*')
    # plt.plot(texture_mean_M['texture_mean'].dropna(), 'r.', alpha=0.5)
    # plt.legend(['Benign', 'Malignant'])
    # plt.title('Mean Texture')
    # # # plt.xlabel('Latitude')
    # # # plt.ylabel('Longitude')
    # plt.tight_layout()
    # plt.show()
    # # # plt.savefig('LonLat.png')

    ##### Smoothness #####
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    smoothness_M = data.loc[data['diagnosis'] == 'M']
    smoothness_B = data.loc[data['diagnosis'] == 'B']
    ax.scatter3D(smoothness_B['smoothness_mean'].dropna(), smoothness_B['smoothness_SE'].dropna(), smoothness_B['smoothness_worst'].dropna(), c=smoothness_B['smoothness_worst'].dropna(), alpha=0.3, cmap='Blues')
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.scatter3D(smoothness_M['smoothness_mean'].dropna(), smoothness_M['smoothness_SE'].dropna(), smoothness_M['smoothness_worst'].dropna(), c=smoothness_M['smoothness_worst'].dropna(), alpha=0.5, cmap='Reds')
    plt.legend(['Benign', 'Malignant'])
    plt.title('Smoothness')
    # # plt.xlabel('Latitude')
    # # plt.ylabel('Longitude')
    # # plt.zlabel('idk')
    plt.tight_layout()
    plt.show()
    # # plt.savefig('smoothness.png')

    ##### Concavity #####
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    concavity_M = data.loc[data['diagnosis'] == 'M']
    concavity_B = data.loc[data['diagnosis'] == 'B']
    ax.scatter3D(concavity_B['concavity_mean'].dropna(), concavity_B['concavity_SE'].dropna(), concavity_B['concavity_worst'].dropna(), c=concavity_B['concavity_worst'].dropna(), alpha=0.9, cmap='Blues')
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.scatter3D(concavity_M['concavity_mean'].dropna(), concavity_M['concavity_SE'].dropna(), concavity_M['concavity_worst'].dropna(), c=concavity_M['concavity_worst'].dropna(), alpha=0.8, cmap='Reds')
    plt.legend(['Benign', 'Malignant'])
    plt.title('Concavity')
    # # plt.xlabel('Latitude')
    # # plt.ylabel('Longitude')
    # # plt.zlabel('idk')
    plt.tight_layout()
    plt.show()
    # # plt.savefig('concavity.png')

if __name__ == '__main__':
    main()