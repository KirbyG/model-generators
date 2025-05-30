from scipy.stats import poisson
Po = poisson.pmf
import numpy as np
from itertools import chain
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Set, Callable, Tuple
from itertools import chain, combinations