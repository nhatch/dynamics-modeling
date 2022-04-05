from abc import ABC, abstractproperty


class NamedDataset(ABC):
    @abstractproperty
    def name(self) -> str:
        pass
