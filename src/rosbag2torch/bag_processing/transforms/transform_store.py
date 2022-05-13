from collections import defaultdict
from typing import Dict, List, Optional, Set, cast

from .abstract_transform import AbstractTransform
from .autorally_ground_truth import AutorallyGroundTruth
from .autorally_state import AutorallyState
from .ground_truth_transform import GroundTruthTransform
from .input_transform import InputTransform
from .odom_transform import OdomTransform


class __TransformStore:
    __transforms: List[AbstractTransform] = [
        GroundTruthTransform(),
        InputTransform(),
        AutorallyGroundTruth(),
        AutorallyState(),
        OdomTransform(),
    ]

    def __new__(cls):
        # Singleton
        if not hasattr(cls, "instance"):
            cls.instance = super(__TransformStore, cls).__new__(cls)
        return cls.instance

    @classmethod
    def get_topics_and_transforms(
        cls,
        bag_topics: Set[str],
        format_map: Dict[str, str],
        transforms: Optional[List[AbstractTransform]] = None,
        features: Optional[List[str]] = None,
    ) -> Optional[Dict[str, List[AbstractTransform]]]:
        # ^ is XOR in python
        assert (transforms is None) ^ (
            features is None
        ), "One (and only one) of the transforms or features must be specified"

        if transforms is None:
            features = cast(List[str], features)  # mypy type hinting

            transforms = []
            features_set = set(features)
            for tf in cls.__transforms:
                if tf.feature in features_set:
                    # tf.topics is a list of sets of topics that this transform is interested in.
                    # Each set is standalone, and can be used to perform a transform into a feature.
                    for tf_topics_set in tf.topics:
                        tf_topics_set_formatted = {
                            topic.format_map(format_map) for topic in tf_topics_set
                        }
                        if tf_topics_set_formatted.issubset(bag_topics):
                            transforms.append(tf)
                            features_set.remove(tf.feature)
                            break
            if len(features_set) > 0:
                return None

        result: Dict[str, List[AbstractTransform]] = defaultdict(list)
        tfs_processed = 0
        for tf in transforms:
            for tf_topics_set in tf.topics:
                tf_topics_set_formatted = {
                    topic.format_map(format_map) for topic in tf_topics_set
                }
                if tf_topics_set_formatted.issubset(bag_topics):
                    for topic in tf_topics_set_formatted:
                        result[topic].append(tf)
                    tfs_processed += 1
                    break

        # If any of the transforms were not processed, then we return None
        if tfs_processed != len(transforms):
            return None

        return result

    @classmethod
    def register_transform(cls, transform: AbstractTransform):
        cls.__transforms.append(transform)


# Expose public functions
get_topics_and_transforms = __TransformStore.get_topics_and_transforms
register_transform = __TransformStore.register_transform
