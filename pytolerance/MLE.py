from numpy.typing import ArrayLike
from scipy.optimize import minimize


class MLE:
    @staticmethod
    def fit(
        data,
        left_censored_data,
        right_censored_data,
        left_interval_data,
        right_interval_data,
        conventional_form,
        distribution="normal",
        find_best: bool = False,
        verbose: bool = False,
    ):
        """Fits a continuous distribution to data using Maximum Likelihood Estimation.

        :param data:
        :param left_censored_data:
        :param right_censored_data:
        :param left_interval_data:
        :param right_interval_data:
        :param distribution: normal (default),
        :param find_best: False (default) or True. If True, all available distributions
                            will be evaluated and the one with the highest likelihood
                            will be returned.
        :param conventional_form:
        """
        pass

    @staticmethod
    def fit_dist(dist, solver: str = "SLSQP", verbose: bool = False):
        sol = minimize(
            dist.nll, x0=dist.x0, method=solver, constraints=dist.constraints
        )
        if verbose:
            print(sol)
        return sol

    @staticmethod
    def mle(dist, solver: str = "SLSQP", verbose: bool = False):
        """Performs Maximum Likelihood Estimation on instance of BaseDistribution."""
        sol = minimize(
            dist.nll, x0=dist._MLEx0, method=solver, constraints=dist._MLEconstraints
        )
        if verbose:
            print(sol)
        return sol
