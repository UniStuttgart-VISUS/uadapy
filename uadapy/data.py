from sklearn import datasets
import numpy as np
import uadapy.distribution as ua

def load_iris_normal():
    """
    Uses the iris dataset and fits a normal distribution
    :return:
    """
    iris = datasets.load_iris()
    dist = []
    for c in np.unique(iris.target):
        dist.append(ua.distribution(np.array(iris.data[iris.target == c]), "Normal"))
    return dist

def load_iris():
    """
    Uses the iris dataset and fits a normal distribution
    :return:
    """
    iris = datasets.load_iris()
    dist = []
    for c in np.unique(iris.target):
        dist.append(ua.distribution(np.array(iris.data[iris.target == c])))
    return dist
