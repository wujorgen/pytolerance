import numpy as np
import scipy
from numpy.typing import ArrayLike

from .__info__ import AVAILABLE_DISTRIBUTIONS, eps


class Distribution:
    """Convenience class that mostly wraps around a scipy.stats.distribution."""

    def __init__(
        self,
        data: ArrayLike,
        left_data: ArrayLike | None = None,
        right_data: ArrayLike | None = None,
        interval_data: ArrayLike | None = None,
        type="normal",
    ):
        """
        :param data: array-like, containing all the data
        :param left_data:
        :param right_data:
        :param interval_data:
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
        self.interval_data = interval_data
        # the following case structure sets:
        # - dist, which is used as the handle for the scipy.stats.<distribution>
        # - x0, which is the initial guess for the MLE routine
        # - constraints, which is the constraints on the distribution parameters
        self.dist = scipy.stats.norm
        self.x0 = [0, 1]
        self.constraints = []
        self.constraints_tl = []
        match self.type:
            case "exponential":
                self.dist = scipy.stats.expon
                # self.constraints = [
                #    {"type": "eq", "fun": lambda x: x[0]},
                # ]
            case "johnsonsu":
                self.dist = scipy.stats.johnsonsu
                self.constraints = [
                    {"type": "ineq", "fun": lambda x: x[1] + eps},
                    {"type": "ineq", "fun": lambda x: x[3] + eps},
                ]
                self.x0 = [0, 1, 0, 1]
            case "weibull":
                pass
            case _:  # normal
                pass  # defined before match statement

    def ll(self, theta):
        """Returns log-likelihood.

        :param theta:
        """
        tmp = self.dist.logpdf(self.data, *theta)
        if self.right_data is not None:
            tmp = np.append(tmp, self.dist.logsf(self.right_data, *theta))
        return np.sum(tmp)

    def nll(self, theta):
        """Returns negative log-likelihood.

        :param theta:
        """
        return -1 * self.ll(theta)
