from enum import Enum


class AVAILABLE_DISTRIBUTIONS(Enum):
    n = "normal"
    e = "exponential"
    w = "weibull"
    j = "johnsonsu"


EPS = 1e-5
