import numpy as np
import scipy
from numpy.typing import ArrayLike

from .__info__ import AVAILABLE_DISTRIBUTIONS, EPS


class Distribution:
    """Convenience class that mostly wraps around a scipy.stats.distribution."""

    def __init__(
        self,
        data: ArrayLike,
        left_data: ArrayLike | None = None,
        right_data: ArrayLike | None = None,
        interval_data_left: ArrayLike | None = None,
        interval_data_right: ArrayLike | None = None,
        type="normal",
    ):
        """
        :param data: array-like, containing all the data
        :param left_data: left-censored data
        :param right_data: right-censored data
        :param interval_data: interval-censored data
        :param type: [n]ormal (default), [e]xponential, [j]ohnsonsu, [w]eibull, TODO
        """
        self.type = type.lower()
        if self.type in [member.name for member in AVAILABLE_DISTRIBUTIONS]:
            self.type = getattr(AVAILABLE_DISTRIBUTIONS, self.type).value
        elif self.type in [member.value for member in AVAILABLE_DISTRIBUTIONS]:
            pass
        else:
            raise TypeError("The selected distribution is not available.")
        self.data = data
        self.left_data = left_data
        self.right_data = right_data
        self.interval_data_left = interval_data_left
        self.interval_data_right = interval_data_right
        # the following case structure sets:
        # - dist, which is used as the handle for the scipy.stats.<distribution>
        # - x0, which is the initial guess for the MLE routine
        # - constraints, which is the constraints on the distribution parameters
        # - theta, which are the specific parameters for each distribution.
        #   scipy uses different conventions from the common forms,
        #   so we have to convert the values back
        self.dist = scipy.stats.norm
        self.x0 = [0, 1]
        self.constraints = []
        self.constraints_tl = []
        self.theta = {}
        match self.type:
            case "exponential":
                self.dist = scipy.stats.expon
                self.constraints = [
                    {"type": "eq", "fun": lambda x: x[0]},
                ]
            case "johnsonsu":
                self.dist = scipy.stats.johnsonsu
                self.constraints = [
                    {"type": "ineq", "fun": lambda x: x[1] + EPS},
                    {"type": "ineq", "fun": lambda x: x[3] + EPS},
                ]
                self.x0 = [0, 1, 0, 1]
            case "weibull":
                self.dist = scipy.stats.weibull_min
                self.constraints = [
                    {"type": "eq", "fun": lambda x: x[1]},
                ]
                self.x0 = [1, 0, 1]
            case _:  # normal
                pass  # defined before match statement

    def ll(self, theta):
        """Returns log-likelihood.

        :param theta:
        """
        tmp = self.dist.logpdf(self.data, *theta)
        if self.right_data is not None:
            tmp = np.append(tmp, self.dist.logsf(self.right_data, *theta))
        if self.left_data is not None:
            tmp = np.append(tmp, self.dist.logcdf(self.right_data, *theta))
        if self.interval_data_left is not None and self.interval_data_right is not None:
            tmp = np.append(
                tmp,
                self.dist.logcdf(self.interval_data_right, *theta)
                - self.dist.logcdf(self.interval_data_left, *theta),
            )
        return np.sum(tmp)

    def nll(self, theta):
        """Returns negative log-likelihood.

        :param theta:
        """
        return -1 * self.ll(theta)


class FlexDistribution:
    """TODO: we may want to re-implement the distribution as a data collection that can switch it's distribution type on the fly."""
