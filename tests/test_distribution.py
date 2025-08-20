import numpy as np
from pytolerance import Distribution


def test_distribution_types():
    random_data = np.random.rand(5)
    dist0 = Distribution(random_data)
    dist1 = Distribution(random_data, "n")
    dist2 = Distribution(random_data, "e")
    # dist3 = Distribution(random_data, "w")
    dist4 = Distribution(random_data, "j")
    # TODO: check that an unavailable distribution raises an Error
