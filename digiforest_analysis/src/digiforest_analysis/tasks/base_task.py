from abc import abstractmethod
from digiforest_analysis.utils.timing import Timer


class BaseTask:
    @abstractmethod
    def __init__(self, **kwargs):
        self._timer = Timer()

    @abstractmethod
    def _process(self, **kwargs):
        pass

    def process(self, **kwargs):
        with self._timer(f"{self.__class__.__name__}"):
            output = self._process(**kwargs)
        print(self._timer)
        return output

    @property
    def timer(self):
        return self._timer
