from functools import partial

from numpy.typing import ArrayLike
from scipy.optimize import minimize

from . import Distribution


class TL:
    @staticmethod
    def fit_upper(dist, solver: str = "SLSQP", verbose: bool = False):
        pass

    @staticmethod
    def fit_lower(dist, solver: str = "SLSQP", verbose: bool = False):
        pass

    @staticmethod
    def fit_double():
        pass

    @staticmethod
    def fit_and_find_best():
        pass


# TODO
"""
def ll_target_cons(data, ll_target, params):
    return ll_target - ll(data, params)


def obj_upper(coverage, params):
    return -1 * scipy.stats.norm.ppf(coverage, *params)


def obj_lower(coverage, params):
    return scipy.stats.norm.ppf(coverage, *params)


def obj_double(lower_coverage, upper_coverage, params):
    upper_quantile = scipy.stats.norm.ppf(upper_coverage, *params)
    lower_quantile = scipy.stats.norm.ppf(lower_coverage, *params)
    return lower_quantile - upper_quantile  # Minimize this <-> maximize interval width, which is upper_quantile - lower_quantile
    """
