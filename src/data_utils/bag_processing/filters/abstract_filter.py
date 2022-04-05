from abc import ABC, abstractmethod, abstractproperty
from typing import List, Set
from rospy import AnyMsg, Time

class AbstractFilter(ABC):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def topics(self) -> List[Set[str]]:
        """
        Returns a list of topics that this filter subscribes to.
        """
        pass

    @abstractproperty
    def should_log(self) -> bool:
        """
        Whether one should currently log.
        This assumes that reader is traversing bag sequentially so last message received in callback is the last message that reader had access to.
        """
        pass

    @abstractmethod
    def callback(self, msg: AnyMsg, ts: Time, topic: str):
        """
        Callback to process subscribed message.
        It may (or may not) change the current state of the filter (returned by should_log).

        Args:
            msg (AnyMsg): Message from topic that this filter subscribed to.
            ts (Time): Timestamp of the message.
            topic (str): Topic from which message was received.
        """
        pass

    @abstractmethod
    def end_bag(self):
        """
        Method for clearing up and reseting the state.
        """
        pass
