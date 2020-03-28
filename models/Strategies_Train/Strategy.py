from abc import ABC, abstractmethod
import numpy as np
import config
import Data

class Strategy(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def applyStrategy(self, data : Data.Data, **kwargs):
        pass