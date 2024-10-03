from typing import Union

from .bounding_boxes import BboxAggregation
from .segmentation_masks import MaskAggregation
from .time_intervals import IntervalAggregation


def build_aggreagtion(
    kind: str,
) -> Union[BboxAggregation, MaskAggregation, IntervalAggregation]:
    if kind == "bbox":
        return BboxAggregation()
    if kind == "mask":
        return MaskAggregation()
    if kind == "interval":
        return IntervalAggregation()
    raise NotImplementedError(
        f"Select aggregation type: `bbox`, `mask`, `interval`. Got: {kind}"
    )


__all__ = [
    "BboxAggregation",
    "MaskAggregation",
    "IntervalAggregation",
    "build_aggreagtion",
]
