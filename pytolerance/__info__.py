from enum import Enum


class AVAILABLE_DISTRIBUTIONS(Enum):
    n = "normal"
    e = "exponential"
    # WEIB = "weibull"
    j = "johnsonsu"


eps = 1e-5
