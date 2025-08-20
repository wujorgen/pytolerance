import numpy as np
from sklearn.datasets import load_iris
from pytolerance import Distribution, MLE


iris = load_iris(as_frame=True)


def test_distribution_types():
    random_data = np.random.rand(5)
    dist0 = Distribution(random_data)
    dist1 = Distribution(random_data, "n")
    dist2 = Distribution(random_data, "e")
    # dist3 = Distribution(random_data, "w")
    dist4 = Distribution(random_data, "j")
    # TODO: check that an unavailable distribution raises an Error


def test_MLE_johnson():
    benchmark_sol_x = np.array([-2.995e+00,  5.097e+00,  1.891e+00,  1.840e+00])
    sepal_width = iris.data["sepal width (cm)"].values
    dist = Distribution(sepal_width, type="j")
    assert dist.type == "johnsonsu"
    assert dist.x0 == [0, 1, 0, 1]
    sol = MLE.fit_dist(dist, verbose=False)
    assert np.all(np.isclose(sol.x, benchmark_sol_x, rtol=1e-3))


# def test_MLE_normal():

# def test_MLE_exponential():