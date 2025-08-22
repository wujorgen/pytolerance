import numpy as np
from sklearn.datasets import load_iris


iris = load_iris(as_frame=True)
sepal_width = iris.data["sepal width (cm)"].values


def test_normal_TL():
    pass


def test_TL_sepal_width():
    pass
    # upper 99/75: 3.441
    # lower 99/25: 2.663
    # 2-sided 98/50: 2.728 to 3.398
    # IQR = 2.757 to 3.333
    # MLE params: -3.7157, 5.5918, 1.6488, 1.9400
