from collections import defaultdict
from typing import DefaultDict, Dict, List, Set

from .abstract_filter import AbstractFilter


def get_filters_topics(
    filters: List[AbstractFilter],
    bag_topics: Set[str],
    format_kwargs: Dict[str, str],
) -> DefaultDict[str, List[AbstractFilter]]:
    """
    Given Filters get all of the topics that are needed for the filter to work that also exist in the bag.

    Raises:
        ValueError: If topics in the bag do not satisfy topic requirements for **any** of the filters.
    """
    result = defaultdict(list)

    for filter_ in filters:
        for topic_set in filter_.topics:
            topic_set = {x.format(**format_kwargs) for x in topic_set}
            if topic_set.issubset(bag_topics):
                for topic in topic_set:
                    result[topic].append(filter_)
                break
            raise ValueError(
                "Filter {} has no set of topics that it can use in the bag.".format(
                    filter_.__class__.__name__
                )
            )

    return result


def flip_filter(filter_: AbstractFilter) -> AbstractFilter:
    """
    Given a filter, return a new filter that is the same as the original, but with should_log flipped.
    """

    class FlipFilter(AbstractFilter):
        def __init__(self, filter_: AbstractFilter):
            self.filter_ = filter_
            self._msg_received = False

        @property
        def topics(self) -> List[Set[str]]:
            return self.filter_.topics

        def callback(self, msg, ts, topic):
            self._msg_received = True
            self.filter_.callback(msg, ts, topic)

        def end_bag(self):
            self.filter_.end_bag()

        @property
        def should_log(self) -> bool:
            if not self._msg_received:
                return False
            return not self.filter_.should_log

    return FlipFilter(filter_)
