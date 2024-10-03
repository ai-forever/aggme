import math
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from aggme.abstracts.aggregation import AbstractAggregation
from aggme.utils.dataclass import MarkupGroup, MaskMarkup


class MaskAggregation(AbstractAggregation):
    def __init__(
        self,
        cluster_method_name: str = "mean_shift",
        cluster_method_params: Optional[dict] = None,
    ):
        cluster_method_params = (
            cluster_method_params if cluster_method_params is not None else dict()
        )
        super().__init__(cluster_method_name, cluster_method_params)
        self.method = MaskMarkup.polygon_iou

    @staticmethod
    def get_group_ious(
        group_by_label: List[MaskMarkup],
        threshold: float,
    ) -> Tuple[float, list, list]:
        """Get average IoU for all masks in group by label

        Parameters
        ----------
        group_by_label: List[MaskMarkup]
            Markups grouped by one label
        threshold: float
            IoU threshold to accept / reject markups

        Returns
        -------
        Tuple[float, list, list]
            result average IoU, accepted annotators, rejected annotators

        """
        rejected = []
        accepted = []
        total_iou = []

        pair = [pp for pp in combinations(group_by_label, 2)]
        for aa, bb in pair:
            iou = aa.iou(bb)
            aa.add_iou(iou)
            bb.add_iou(iou)
            total_iou.append(iou)

        for markup in group_by_label:
            if markup.get_avg_iou() < threshold:
                rejected.append(markup.annotator)
            else:
                accepted.append(markup.annotator)
        iou = sum(total_iou) / len(total_iou)
        return iou, accepted, rejected

    def hard_aggregation(
        self,
        group: MarkupGroup,
        threshold: float = 0.5,
    ) -> Optional[MarkupGroup]:
        """Hard aggregation for markups.

        Parameters
        ----------
        group : MarkupGroup
            Group of markups
        threshold : float
            IoU threshold
        """
        group.reset()
        group.fill_missing()

        label_with_markups = self._aggregate(group, threshold)

        return self._res_to_group(label_with_markups, group.name, method="_hard")

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
            markup = MaskMarkup(annotator=f"Result{method}", mask=data, label=label)
            res_group.data.append(markup)
        return res_group

    def _aggregate(
        self,
        group: MarkupGroup,
        threshold: float = 0.5,
        confidence: float = 1.0,
    ) -> Optional[list]:
        """Aggregate markups: check all centroids -> find markups -> match by IoU.

        Parameters
        ----------
        group : MarkupGroup
            Group of annotations with markups
        threshold : float
            IoU threshold
        confidence: float
            Markups confidence

        """
        if confidence <= 0:
            raise ValueError("Confidence should be greater than 0!")

        lenghts_by_labels = [len(x) for x in group.get_groups_by_labels()]
        if len(group.get_groups_by_annotators()) < self.MIN_ANNOTATIONS:  # noqa
            return
        if len(set(lenghts_by_labels)) != 1:
            return
        result = []
        for group_by_labels in group.get_groups_by_labels():
            iou, accepted, rejected = self.get_group_ious(group_by_labels, threshold)

            if iou >= threshold:
                mask = np.average([markup.mask for markup in group_by_labels], axis=0)
                mask[mask < confidence] = 0
                mask[mask >= confidence] = 1
                result.append((group_by_labels[0].label, mask))
            else:
                return
        return result

    def soft_aggregation(
        self,
        group: MarkupGroup,
        threshold: float = 0.5,
        confidence: float = 0.7,
    ) -> Optional[tuple]:
        """Soft aggregation for markups.

        Parameters
        ----------
        group : MarkupGroup
            Group of annotations with markups
        threshold : float
            Similarity metric threshold
        confidence: float
            Markups confidence
        Returns
        -------
        list
            Aggregation, accepted and rejected assignments
        """
        group.reset()
        if len(group.get_annotators()) < self.MIN_ANNOTATIONS:
            return self._res_to_group(None, group.name, method="_soft"), [], []

        labels_with_markup, all_accepted, all_rejected = [], [], []

        lengths = [len(g) for g in group.get_groups_by_annotators()]
        required_amount = math.ceil(len(lengths) * confidence)
        group.fill_missing()

        for group_to_drop in group.get_groups_by_labels():
            drop_group = MarkupGroup(
                name=group.name,
                data=group_to_drop,
                dimension=group.dimension,
                relative=group.relative,
            )
            drop_num = 0
            label_with_markup, accepted, rejected = [], [], []
            aggregated = False
            while len(group.get_annotators()) - drop_num >= required_amount:
                label_with_markup, accepted, rejected = self.drop_aggregation(
                    drop_group,
                    threshold,
                    confidence,
                    drop_num=drop_num,
                )
                drop_num += 1
                if label_with_markup:
                    all_rejected += rejected
                    aggregated = True
                    break

            if not aggregated:
                return self._res_to_group(None, group.name, method="_soft"), [], []
            if isinstance(label_with_markup, MarkupGroup):
                label_with_markup = [
                    (label_with_markup.data[0].label, label_with_markup.data[0].mask)
                ]
            labels_with_markup.extend(label_with_markup)

        all_accepted = set(group.get_annotators()) - set(all_rejected)
        return (
            self._res_to_group(labels_with_markup, group.name, method="_soft"),
            all_accepted,
            all_rejected,
        )
