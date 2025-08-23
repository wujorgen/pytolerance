import numpy as np
from sklearn.datasets import load_iris
from pytolerance import Distribution, MLE, NormalDistribution

"""
The tests here use the demonstration at: https://reliability.readthedocs.io/en/latest/How%20does%20Maximum%20Likelihood%20Estimation%20work.html
"""

iris = load_iris(as_frame=True)
sepal_width = iris.data["sepal width (cm)"].values


def test_MLE_normal():
    dataset = NormalDistribution(data=sepal_width)
    dataset.fit()
    assert dataset._sol.success, "MLE failed!"
    assert np.all(np.isclose(dataset._sol.x, [sepal_width.mean(), sepal_width.std()], rtol=1e-3))


def test_MLE_johnson():  # TODO: rewrite
    benchmark_sol_x = np.array([-2.995e+00,  5.097e+00,  1.891e+00,  1.840e+00])
    dist = Distribution(sepal_width, type="j")
    assert dist.type == "johnsonsu"
    assert dist.x0 == [0, 1, 0, 1]
    sol = MLE.fit_dist(dist, verbose=False)
    assert np.all(np.isclose(sol.x, benchmark_sol_x, rtol=1e-3))


def test_MLE_exponential():  # TODO: rewrite
    data = np.array([27, 64, 3, 18, 8])
    dist = Distribution(data, type="exponential")
    lam = 0.1
    scale = 1/lam
    assert np.isclose(dist.ll([0, scale]), -23.51292546497023, rtol=1e-2)
    dist.constraints = [
                    {"type": "eq", "fun": lambda x: x[0]},
                ]
    sol = MLE.fit_dist(dist, verbose=False)
    assert np.all(np.isclose(sol.x, [0, 1/0.0416667], rtol=1e-3))


def test_MLE_exponential_right_censor():  # TODO: rewrite
    data = np.array([17, 5, 12])
    right_censored_data = np.array([20, 25])
    lam = 0.1
    scale = 1/lam
    dist = Distribution(data, right_data=right_censored_data, type="exponential")
    assert np.isclose(dist.ll([0, scale]), -14.807755278982137, rtol=1e-2)
    dist.constraints = [
                {"type": "eq", "fun": lambda x: x[0]},
            ]
    sol = MLE.fit_dist(dist, verbose=False)
    assert np.all(np.isclose(sol.x, [0, 1/0.0379747], rtol=1e-3))
    

def test_MLE_weibull_right_censor():  # TODO: rewrite
    data = np.array([17, 5, 12])
    right_censored_data = np.array([20, 25])
    dist = Distribution(data, right_data=right_censored_data, type="w")
    alpha = 15
    beta = 2
    c = beta
    loc = 0
    scale = alpha
    print(dist.ll([c, loc, scale]))
    assert np.isclose(dist.ll([c, loc, scale]), -13.8324, rtol=1e-3)
    dist.constraints = [
                {"type": "eq", "fun": lambda x: x[1]},
            ]
    sol = MLE.fit_dist(dist, verbose=False)
    assert np.isclose(-1 * sol.fun, -12.4823)
    assert np.all(np.isclose(sol.x, [1.57474, 0, 23.0653], rtol=1e-3))


if __name__ == "__main__":
    test_MLE_normal()
    #test_MLE_exponential()
    #test_MLE_exponential_right_censor()
    #test_MLE_weibull_right_censor()