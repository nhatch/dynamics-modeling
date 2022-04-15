from collections import defaultdict
from email.policy import default
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
                "Filter {} has no set of topics that it can use in the bag.".format(filter_.__class__.__name__)
            )

    return result
