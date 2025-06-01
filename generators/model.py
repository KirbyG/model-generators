from abc import ABC, abstractmethod
import numpy as np
from typing import List

class Model(ABC):
    def mask(self, dirs):
        arr = np.zeros(self.rank, dtype='int')
        for d in dirs:
            arr[self.dimensions.index(d.dim)] = d.sign
        return arr

    dimensions: List

    @abstractmethod
    def __init__(self, dimensions, shape, transitions, verbose=False):
        self.dimensions = dimensions
        self.shape = shape
        self.transitions = transitions
        self.verbose = verbose

        self.rank = len(dimensions)
    
    @abstractmethod
    def empty_world(self):
        pass

    @abstractmethod
    def run(self, initial_condition, T):
        pass

    @abstractmethod
    def time_series(self, axis):
        pass

    @abstractmethod
    def distribution(self, axis):
        pass