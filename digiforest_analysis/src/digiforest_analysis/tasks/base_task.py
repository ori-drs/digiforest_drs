from abc import abstractmethod
from digiforest_analysis.utils.timing import Timer
import numpy as np


class BaseTask:
    @abstractmethod
    def __init__(self, **kwargs):
        self._timer = Timer()
        self._viz_center = np.zeros(3)
        self._viz_zoom = kwargs.get("viz_zoom", 0.8)
        self._debug_level = kwargs.get("debug_level", 0)

    @abstractmethod
    def _process(self, **kwargs):
        pass

    def process(self, **kwargs):
        with self._timer(f"{self.__class__.__name__}"):
            output = self._process(**kwargs)
        print(self._timer, end="")
        return output

    @property
    def timer(self):
        return self._timer

    @property
    def viz_center(self):
        return self._viz_center

    @property
    def viz_zoom(self):
        return self._viz_zoom

    @viz_center.setter
    def viz_center(self, viz_center):
        self._viz_center = viz_center
