from typing import Optional

from aggme.aggregation.bounding_boxes import BboxAggregation
from aggme.utils.dataclass import IntervalMarkup, MarkupGroup


class IntervalAggregation(BboxAggregation):
    ANNOTATION_TYPE = "interval"

    def __init__(
        self,
        cluster_method_name: str = "dbscan",
        cluster_method_params: Optional[dict] = None,
    ):
        cluster_method_params = (
            cluster_method_params if cluster_method_params is not None else dict()
        )

        super().__init__(cluster_method_name, cluster_method_params)
        self.iou_func = IntervalMarkup.iou_1d
        self.cluster_method = self.CLUSTER_METHODS[cluster_method_name].set_params(
            **cluster_method_params
        )

    @staticmethod
    def _res_to_group(
        res,
        group_name: str,
        method: str = "",
    ) -> Optional[MarkupGroup]:
        if res is None:
            return None
        res_group = MarkupGroup(
            name=group_name, data=[], dimension=None, relative=False
        )
        for label_with_markup in res:
            label = label_with_markup[0]
            data = label_with_markup[1]
            markup = IntervalMarkup(
                annotator=f"Result{method}", coordinates=data, label=label
            )
            res_group.data.append(markup)
        return res_group
