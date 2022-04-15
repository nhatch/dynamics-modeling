from abc import ABC, abstractproperty
from typing import List

class ModelWithSpec(ABC):
    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def dataset_name(self) -> str:
        pass

    @abstractproperty
    def x_features(self) -> List[str]:
        pass

    @abstractproperty
    def y_features(self) -> List[str]:
        pass
