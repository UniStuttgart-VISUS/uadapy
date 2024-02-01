from sklearn import datasets
import numpy as np
import imuncertain as ua

def load_iris():
    """
    Uses the iris dataset and fits a normal distribution
    :return:
    """
    iris = datasets.load_iris()
    dist = []
    for c in np.unique(iris.target):
        dist.append(ua.distribution.distribution(np.array(iris.data[iris.target == c]), "Normal"))
    return dist
