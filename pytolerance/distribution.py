from abc import ABC, abstractmethod

import numpy as np
import scipy
from numpy.typing import ArrayLike
from scipy.stats._distn_infrastructure import rv_continuous

from .__info__ import AVAILABLE_DISTRIBUTIONS, EPS
from .MLE import MLE


class BaseDistribution(ABC):
    """Abstract Base Class for Distribution objects.
    Must be extended to a specific distribution family."""

    def __init__(
        self,
        data,
        left_censored_data=None,
        right_censored_data=None,
        left_interval_data=None,
        right_interval_data=None,
        conventional_form: bool = False,
    ):
        """Constructor for Distribution class.

        :param data:
        :param left_censored_data:
        :param right_censored_data:
        :param left_interval_data:
        :param right_interval_data:
        :param conventional_form:
        """
        self.data = data
        self.left_censored_data = left_censored_data
        self.right_censored_data = right_censored_data
        self.left_interval_data = left_interval_data
        self.right_interval_data = right_interval_data
        self.conventional_form = conventional_form
        self._sol = None
        self._params = {}
        self._cache_TL = {}  # [U/L/D][confidence]/[coverage]

    @property
    @abstractmethod
    def dist(self) -> rv_continuous:
        """Returns a reference to the SciPy stats family."""

    @property
    @abstractmethod
    def params(self) -> dict:
        """Returns a dict of the distribution parameters."""

    @property
    @abstractmethod
    def _MLEconstraints(self) -> list[dict]:
        """Returns constraints on parameter feasibility needed for MLE."""

    @property
    @abstractmethod
    def _MLEx0(self) -> list:
        """Returns the initial guess for MLE."""

    def _calc_ll_target(self, confidence, sided=1, verbose: bool = False):
        """Decrements the log likelihood for tolerance limit calculation.

        :param confidence: between 0 and 1
        :param sided: 1 or 2
        """
        if self._sol is None:
            pass  # TODO: throw error or run MLE
        ll_mle = self._sol.fun * -1  # optimization problem minimizes nll
        alpha_eff = 2 * (1 - confidence) / sided
        if verbose:
            print("alpha effective:", alpha_eff)
        chi2val = scipy.stats.chi2.ppf(1 - alpha_eff, df=1)
        ll_target = ll_mle - chi2val / 2
        return ll_target

    def ll(self, theta):
        """Returns log-likelihood.

        :param theta:
        :return:
        """
        tmp = self.dist.logpdf(self.data, *theta)
        if self.right_censored_data is not None:
            tmp = np.append(tmp, self.dist.logsf(self.right_censored_data, *theta))
        if self.left_censored_data is not None:
            tmp = np.append(tmp, self.dist.logcdf(self.left_censored_data, *theta))
        if self.left_interval_data is not None and self.right_interval_data is not None:
            tmp = np.append(
                tmp,
                self.dist.logcdf(self.right_interval_data, *theta)
                - self.dist.logcdf(self.left_interval_data, *theta),
            )
        return np.sum(tmp)

    def nll(self, theta):
        """Returns negative log-likelihood.

        :param theta:
        :return:
        """
        return -1 * self.ll(theta)

    def fit(self, verbose: bool = False):
        """Fits distribution to provided data via Maximum Likelihood Estimation."""
        self._sol = MLE.mle(self, "SLSQP", verbose)
        # TODO: need logic to try other solvers/constraints in event of failure

    def tl_upper(self, confidence, coverage):
        """Calculates and caches the upper one-sided tolerance limit.
        This is done by targeting a decremented log likelihood while maximizing the
        percentile associated with the desired coverage.

        :param confidence: in range 0 and 100
        :param coverage: in range 0 and 100
        """
        ll_target = self._calc_ll_target(confidence, 1)

    def tl_lower(self, confidence, coverage):
        """Calculates and caches the lower one-sided tolerance limit.
        This is done by targeting a decremented log likelihood while minimizing the
        percentile associated with the desired coverage.

        :param confidence: in range 0 and 100
        :param coverage: in range 0 and 100
        """
        ll_target = self._calc_ll_target(confidence, 1)

    def tl_double(self, confidence, coverage_low, coverage_high):
        """Calculates and caches the two-sided tolerance limit.
        This is done by targeting a decremented log likelihood while maximizing the
        difference between the low and high percentiles.

        :param confidence: in range 0 and 100
        :param coverage: in range 0 and 100
        """
        ll_target = self._calc_ll_target(confidence, 2)


class NormalDistribution(BaseDistribution):
    def __init__(
        self,
        data,
        left_censored_data=None,
        right_censored_data=None,
        left_interval_data=None,
        right_interval_data=None,
        conventional_form: bool = False,
    ):
        super().__init__(
            data,
            left_censored_data,
            right_censored_data,
            left_interval_data,
            right_interval_data,
            conventional_form=conventional_form,
        )

    @property
    def dist(self) -> rv_continuous:
        """Returns a reference to the SciPy stats family."""
        return scipy.stats.norm

    @property
    def params(self) -> dict:
        """Returns a dict of the distribution parameters."""
        return self.params

    @property
    def _MLEconstraints(self) -> list:
        """Returns constraints on parameter feasibility needed for MLE.
        The normal distribution does not have any requirements."""
        return []

    @property
    def _MLEx0(self) -> list:
        """Returns the initial guess for MLE."""
        return [0, 1]

    def tl_upper(self, confidence, coverage):
        """Calculates and caches the upper one-sided tolerance limit.
        An analytical solution is available for the normal distribution
        when censored data is not present."""
        if (
            self.left_censored_data is None
            and self.right_censored_data is None
            and self.left_interval_data is None
            and self.right_interval_data is None
        ):
            n = self.data.size
            non_centrality = np.sqrt(n) * scipy.stats.norm.ppf(coverage)
            k = (1 / np.sqrt(n)) * scipy.stats.nct.ppf(
                confidence, n - 1, non_centrality
            )
            return self._sol.x[0] + k * self._sol.x[1]
        else:
            return super().tl_upper(confidence, coverage)

    def tl_lower(self, confidence, coverage):
        """Calculates and caches the lower one-sided tolerance limit.
        An analytical solution is available for the normal distribution
        when censored data is not present."""
        if (
            self.left_censored_data is None
            and self.right_censored_data is None
            and self.left_interval_data is None
            and self.right_interval_data is None
        ):
            n = self.data.size
            non_centrality = np.sqrt(n) * scipy.stats.norm.ppf(coverage)
            k = (1 / np.sqrt(n)) * scipy.stats.nct.ppf(
                confidence, n - 1, non_centrality
            )
            return self._sol.x[0] - k * self._sol.x[1]
        else:
            return super().tl_lower(confidence, coverage)


class ExponentialDistribution(BaseDistribution):
    pass


class JohnsonSUDistribution(BaseDistribution):
    pass


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
