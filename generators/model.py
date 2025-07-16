from abc import ABC, abstractmethod
import numpy as np
from typing import List

class Model(ABC):
    dimensions: List

    @abstractmethod
    def __init__(self, dimensions, shape, transitions, verbose=False):
        self.dimensions = dimensions
        self.shape = np.array(shape)
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

    @abstractmethod
    def animation(self, axes):
        pass