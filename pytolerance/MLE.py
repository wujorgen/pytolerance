from numpy.typing import ArrayLike
from scipy.optimize import minimize

from . import Distribution


class MLE:
    @staticmethod
    def fit(dist: Distribution | ArrayLike, verbose: bool = False):
        """Fits the target distribution using Maximum Likelihood Estimation.

        :param dist: pytolerance.Distribution
        :return sol.x:
        :raise Error:
        """
        if type(dist) == Distribution:
            pass
            # MLE.fit_dist()
        else:
            # MLE.fit_data()
            pass

    @staticmethod
    def fit_dist(dist: Distribution | ArrayLike, method:str="SLSQP", verbose: bool = False):
        sol = minimize(dist.nll, x0=dist.x0, method=method, constraints=dist.constraints)
        if verbose:
            print(sol)
        return sol

    @staticmethod
    def fit_data(data, handle):
        pass
