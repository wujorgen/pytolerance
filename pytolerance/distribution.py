import numpy as np
from numpy.typing import ArrayLike
import scipy

from .__info__ import AVAILABLE_DISTRIBUTIONS, eps


class Distribution:
    """Convenience class that mostly wraps around a scipy.stats.distribution."""
    def __init__(self, data:ArrayLike, type="normal"):
        """
        :param data: array-like, containing all the data
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
        # the following case structure sets:
        # - dist, which is used as the handle for the scipy.stats.<distribution>
        # - x0, which is the initial guess for the MLE routine
        # - constraints, which is the constraints required for the distribution parameters
        self.dist = scipy.stats.norm
        self.x0 = [0, 1]
        self.constraints = []
        self.constraints_tl = []
        match self.type:
            case "exponential":
                self.dist = scipy.stats.expon
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

    def ll(self, theta:ArrayLike):
        """Returns log-likelihood.

        :param theta:
        """
        return np.sum(self.dist.logpdf(self.data, *theta))

    def nll(self, theta:ArrayLike):
        """Returns negative log-likelihood.

        :param theta:
        """
        return -1 * self.ll(theta)
