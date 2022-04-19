"""
Module containing transforms for bag topics and converting them into corresponding numpy arrays.

Sorting matters here, since some transforms might overlap in features/topics.

isort:skip_file
"""
from collections import defaultdict
from inspect import isclass
from typing import Dict, List, Set

from .abstract_transform import AbstractTransform
from . import (
    ground_truth_transform,
    input_transform,
    autorally_ground_truth,
    autorally_state,
    odom_transform,
)


def get_topics_and_transforms(
    features: List[str], bag_topics: Set[str], format_map: Dict[str, str]
) -> Dict[str, List[AbstractTransform]]:
    result = defaultdict(list)
    features_to_find = set(features)

    # NOTE: Ordering matters here a lot!
    for callback_module in [
        ground_truth_transform,
        input_transform,
        odom_transform,
        autorally_ground_truth,
        autorally_state,
    ]:
        for obj_str in dir(callback_module):
            # Faster filter for built-in tools and the abstract class
            if obj_str.startswith("__") or obj_str == "AbstractTransform":
                continue

            obj = getattr(callback_module, obj_str)
            if isclass(obj) and issubclass(obj, AbstractTransform):
                # Check if it feature matches any of the requested features
                if obj.feature in features_to_find:
                    for topic in obj.topics:
                        topic = topic.format_map(format_map)

                        if topic in bag_topics:
                            # Topic exists in a bag, and feature is requested.

                            # Instantiate and append object to result
                            result[topic].append(obj(features))
                            features_to_find.discard(obj.feature)
                            break
    return result
