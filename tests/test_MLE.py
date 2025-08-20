import numpy as np
from sklearn.datasets import load_iris
from pytolerance import Distribution, MLE

"""
The tests here use the demonstration at: https://reliability.readthedocs.io/en/latest/How%20does%20Maximum%20Likelihood%20Estimation%20work.html
"""

iris = load_iris(as_frame=True)

def test_MLE_johnson():
    benchmark_sol_x = np.array([-2.995e+00,  5.097e+00,  1.891e+00,  1.840e+00])
    sepal_width = iris.data["sepal width (cm)"].values
    dist = Distribution(sepal_width, type="j")
    assert dist.type == "johnsonsu"
    assert dist.x0 == [0, 1, 0, 1]
    sol = MLE.fit_dist(dist, verbose=False)
    assert np.all(np.isclose(sol.x, benchmark_sol_x, rtol=1e-3))


# def test_MLE_normal():


def test_MLE_exponential():
    data = np.array([27, 64, 3, 18, 8])
    dist = Distribution(data, type="exponential")
    assert np.isclose(dist.ll([0, 10]), -23.51292546497023, rtol=1e-2)
    dist.constraints = [
                    {"type": "eq", "fun": lambda x: x[0]},
                ]
    sol = MLE.fit_dist(dist, verbose=False)
    assert np.all(np.isclose(sol.x, [0, 1/0.0416667], rtol=1e-3))


def test_MLE_exponential_right_censor():
    data = np.array([17, 5, 12])
    right_censored_data = np.array([20, 25])
    dist = Distribution(data, right_data=right_censored_data, type="exponential")
    assert np.isclose(dist.ll([0, 10]), -14.807755278982137, rtol=1e-2)
    dist.constraints = [
                {"type": "eq", "fun": lambda x: x[0]},
            ]
    sol = MLE.fit_dist(dist, verbose=False)
    assert np.all(np.isclose(sol.x, [0, 1/0.0379747], rtol=1e-3))
    

def test_MLE_weibull_right_censor():
    data = np.array([17, 5, 12])
    right_censored_data = np.array([20, 25])


if __name__ == "__main__":
    test_MLE_exponential()
    test_MLE_exponential_right_censor()