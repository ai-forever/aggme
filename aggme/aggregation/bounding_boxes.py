import math
from collections import defaultdict
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from aggme.abstracts.aggregation import AbstractAggregation
from aggme.utils.dataclass import BboxMarkup, MarkupGroup


class BboxAggregation(AbstractAggregation):
    ANNOTATION_TYPE: str = "bboxes"

    def __init__(
        self,
        cluster_method_name: str = "mean_shift",
        cluster_method_params: Optional[dict] = None,
    ):
        cluster_method_params = (
            cluster_method_params if cluster_method_params is not None else dict()
        )
        super().__init__(cluster_method_name, cluster_method_params)
        self.cluster_method = self.CLUSTER_METHODS[cluster_method_name].set_params(
            **cluster_method_params
        )
        self.iou_func = BboxMarkup.iou_coco

    def _aggregate(
        self,
        annotation_group: MarkupGroup,
        threshold: float = 0.5,
        confidence: float = 1.0,
    ) -> Optional[list]:
        """Aggregate markups: check all centroids -> find markups -> match by IoU.

        Parameters
        ----------
        annotation_group: MarkupGroup
            Group of markups
        threshold: float
            IoU threshold
        confidence: float
            Markups confidence

        """

        lenghts = [len(mark) for mark in annotation_group.get_groups_by_annotators()]
        if len(set(lenghts)) == 1:
            if lenghts[0] == 0:
                return

            if centroids := self._check_by_centroids(annotation_group, confidence):
                markups, _, _, _ = centroids
                if (
                    sorted([len(mark) for mark in markups.values()])[-1]
                    < self.MIN_ANNOTATIONS
                ):
                    return
                label_with_markup = BboxMarkup.check_label_with_markups(
                    self.iou_func, markups, threshold
                )

                if len(label_with_markup) > 0:
                    return label_with_markup

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
            markup = BboxMarkup(
                annotator=f"Result{method}", coordinates=data, label=label
            )
            res_group.data.append(markup)
        return res_group

    def _markups_filter(
        self,
        group: MarkupGroup,
        confidence: float,
    ) -> Optional[tuple]:
        """Filter for bad markups.

        Parameters
        ----------
        group : MarkupGroup
            Group of annotations
        confidence: float
            Markups confidence

        Returns
        -------
        list
            Remaining annotates, rejected assignments and failed flag

        """

        rejected = []

        # Search for points and duplicates (or close to duplicate) in markups
        points = group.get_points()
        duplicates = group.get_duplicates(self.iou_func)

        # Delete points and duplicates
        additional_rejected = group.drop_markups(points) + group.drop_markups(
            duplicates
        )
        rejected += additional_rejected

        # Search and delete unmatched markups
        _, failed, _, unmatched_markups = self._check_by_centroids(
            group,
            confidence,
        )

        if unmatched_markups:
            additional_rejected = group.drop_markups(unmatched_markups)
            rejected += additional_rejected
        return group, rejected, failed

    def _check_labels(
        self,
        markup_group: MarkupGroup,
        confidence: float,
    ) -> Optional[tuple]:
        """Check markups label matching.

        Parameters
        ----------
        markup_group : MarkupGroup
            Group of markups
        confidence: float
            Markups confidence

        Returns
        -------
        tuple
            Changed markups, rejected and failed flag

        """
        failed = False
        changed = False
        rejected = []
        items_with_markups, _, matched_groups_ids, _ = self._check_by_centroids(
            markup_group,
            confidence,
        )
        for group_id in matched_groups_ids:
            group = items_with_markups[group_id]
            keys = list(map(lambda x: x.label, group))

            if len(set(keys)) != 1:

                changed = True
                values, counts = np.unique(np.array(keys), return_counts=True)
                presence = max(counts) / len(group)
                absence = 1 - presence

                if presence < confidence and absence < confidence:
                    failed = True

                elif presence >= confidence:
                    key = [
                        values[i]
                        for i in range(len(values))
                        if counts[i] == max(counts)
                    ][0]
                    # Change key if needed
                    for markup in markup_group.data:
                        if markup in group:
                            if markup.label != key:
                                # change
                                markup.set_label(key)
                                rejected.append(markup.annotator)
        if changed:
            return markup_group, rejected, failed
        return markup_group, rejected, failed

    def _check_count(
        self,
        markup_group: MarkupGroup,
        required_amount: int,
        required_num_markups: int,
        threshold: float,
        confidence: float,
    ) -> Optional[tuple]:
        """Check markups count.

        Parameters
        ----------
        markup_group : MarkupGroup
            Group of markups
        required_amount: int
            Amount of required markups -- for one group
        required_num_markups: int
            Amount of required markups in each markup -- for one annotator
        threshold : float
            IoU threshold
        confidence: float
            Markups confidence

        Returns
        -------
        list
            Changed markups and rejected

        """
        rejected = []
        added_boxes = []
        items_with_markups, _, matched_groups_ids, _ = self._check_by_centroids(
            markup_group,
            confidence,
        )
        for group_id in matched_groups_ids:
            group = items_with_markups[group_id]
            # Check if group lacks markups and in group >= required_amount markups
            if required_amount <= len(group) < len(markup_group.get_annotators()):
                added_markup = BboxMarkup.check_label_with_markups(
                    self.iou_func, {group_id: group}, threshold
                )

                for user_answers in markup_group.get_groups_by_annotators():
                    # Search markups[j] in groups
                    intersection = [i for i in user_answers if i in group]

                    if len(user_answers) < required_num_markups and not intersection:
                        annotator = user_answers[0].annotator
                        rejected.append(annotator)
                        if len(added_markup) > 0:
                            label = added_markup[0][0]
                            coords = added_markup[0][1]

                            new_box = BboxMarkup(label, coords, annotator)
                            added_boxes.append(new_box)
        new_group = MarkupGroup(
            name=markup_group.name,
            data=markup_group.data + added_boxes,
            dimension=markup_group.dimension,
            relative=markup_group.relative,
            point_threshold=markup_group.point_threshold,
            duplicate_threshold=markup_group.duplicate_threshold,
        )

        return new_group, rejected

    @staticmethod
    def _check_matching(
        matched_markups: dict,
        annotators_len: int,
        confidence: float,
    ) -> Optional[Tuple[List, List, bool]]:
        """Check markups matching.

        Parameters
        ----------
        matched_markups : dict
            Markup groups after matching: {0: [[label, markup], [label, markup], ...], 1: ...}
        annotators_len: int
            Length of unique annotators
        confidence: float
            Markups confidence

        Returns
        -------
        list
            Unmatched markups, matched flag and failed flag
        """
        failed = False
        unmatched_markups = []
        matched_groups = []

        for key, markups in matched_markups.items():
            if confidence == 1.0:
                if len(markups) != annotators_len:
                    return unmatched_markups, matched_groups, True
                else:
                    matched_groups.append(key)

            else:
                presence = len(markups) / annotators_len
                absence = 1 - presence
                if presence < confidence and absence < confidence:
                    failed = True
                elif presence < confidence <= absence:
                    unmatched_markups.extend(markups)
                elif presence >= confidence:
                    matched_groups.append(key)

        return unmatched_markups, matched_groups, failed

    def _check_by_centroids(
        self,
        group: MarkupGroup,
        confidence: float,
    ) -> Optional[tuple]:
        """Find groups of annotations.

        Parameters
        ----------
        group : MarkupGroup
            Group of markups for one item
        confidence: float
            Markups confidence

        Returns
        -------
        Tuple
            Dict of matched markups or unmatched markups (with / without failed)
        """

        # Find centers
        centers = {}
        for i, markup in enumerate(group.data):
            centers[i] = markup.get_center()

        # Sorting centers with distances
        centers = dict(sorted(centers.items(), key=lambda item: item[1]))

        # Grouping annotation
        matched_markups = defaultdict(list)
        used_keys = []
        for key, val in centers.items():
            if not matched_markups:
                if key not in used_keys:
                    matched_markups[val].append(group.data[key])
                    used_keys.append(key)
            else:
                added = False
                for kk, vv in matched_markups.items():
                    if group.data[key].check_matching(kk):
                        if key not in used_keys:
                            matched_markups[kk].append(group.data[key])
                            used_keys.append(key)
                            added = True

                if not added:
                    if key not in used_keys:
                        matched_markups[val].append(group.data[key])
                        used_keys.append(key)

        # Search unmatched markups and check after distances
        unmatched_markups, matched_groups, failed = self._check_matching(
            matched_markups,
            len(group.get_annotators()),
            confidence,
        )

        if failed or unmatched_markups:
            matched_markups = defaultdict(list)
            if self.ANNOTATION_TYPE == "interval":
                self.cluster_method.fit(np.reshape(list(centers.values()), (-1, 1)))
            else:
                self.cluster_method.fit(list(centers.values()))
            groups = self.cluster_method.labels_
            for ii, idx in enumerate(centers):
                matched_markups[groups[ii]].append(group.data[idx])

        # Search unmatched markups and check after chosen cluster method
        unmatched_markups, matched_groups, failed = self._check_matching(
            matched_markups,
            len(group.get_annotators()),
            confidence,
        )

        if confidence < 1.0:
            return (
                matched_markups,
                failed,
                matched_groups,
                list(map(lambda x: x, unmatched_markups)),
            )

        return matched_markups, None, None, None

    def hard_aggregation(
        self,
        group: MarkupGroup,
        threshold: float = 0.5,
    ) -> Optional[MarkupGroup]:
        """Hard aggregation method.

        Parameters
        ----------
        group : MarkupGroup
            Group of annotations
        threshold : float
            Similarity metric threshold (for IoU, Dice etc.)
        """
        group.reset()

        if len(group.get_annotators()) >= self.MIN_ANNOTATIONS:
            return self._res_to_group(
                self._aggregate(group, threshold), group.name, method="_hard"
            )

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
            Group of annotations
        threshold : float
            IoU threshold
        confidence: float
            Markups confidence
        Returns
        -------
        list
            Aggregation, accepted and rejected assignments and failed flag
        """
        group.reset()
        if len(group.get_annotators()) < self.MIN_ANNOTATIONS:
            return self._res_to_group(None, group.name, method="_soft"), [], []

        label_with_markup, accepted, rejected = None, [], []

        lengths = [len(g) for g in group.get_groups_by_annotators()]
        required_amount = math.ceil(len(lengths) * confidence)
        _, _, matched_groups_ids, _ = self._check_by_centroids(
            group, confidence=confidence
        )
        required_num_markups = len(matched_groups_ids)
        annotators = group.get_annotators()

        # Checking counts
        if len(set(lengths)) != 1:
            group, additional_rejected = self._check_count(
                group,
                required_amount,
                required_num_markups,
                threshold,
                confidence,
            )
            rejected += additional_rejected

        # Filtering points, duplicates and unmatched markups
        filtered_group, additional_rejected, failed = self._markups_filter(
            group, confidence=confidence
        )
        rejected += additional_rejected

        if failed:
            return (
                self._res_to_group(label_with_markup, group.name, method="_soft"),
                set(),
                set(),
            )

        # Checking labels
        checked_group, additional_rejected, failed = self._check_labels(
            filtered_group, confidence
        )
        rejected += additional_rejected

        if failed:
            return (
                self._res_to_group(label_with_markup, group.name, method="_soft"),
                set(),
                set(),
            )

        if label_with_markup := self._aggregate(
            checked_group,
            threshold,
            confidence,
        ):
            accepted += checked_group.get_annotators()
        else:
            if not checked_group.data:
                return (
                    self._res_to_group(label_with_markup, group.name, method="_soft"),
                    set(),
                    set(),
                )
            while (
                max(
                    list(
                        map(
                            lambda x: len(x),
                            self._check_by_centroids(
                                checked_group,
                                confidence,
                            )[0].values(),
                        )
                    )
                )
                > self.MIN_ANNOTATIONS
            ):

                items_with_markups, _, _, _ = self._check_by_centroids(
                    checked_group, confidence
                )
                drops = []

                for _, item in items_with_markups.items():
                    if len(item) <= self.MIN_ANNOTATIONS:
                        continue
                    pairs = combinations(item, 2)
                    ious = sorted(
                        [
                            (self.iou_func(markup1, markup2), markup1, markup2)
                            for markup1, markup2 in pairs
                        ],
                        key=lambda x: x[0],
                    )
                    if ious:
                        drop1 = ious[0][1]
                        drop2 = ious[0][2]
                        for iou, mark_1, mark_2 in ious[1:]:
                            if (mark_1 == drop1) or (mark_2 == drop1):
                                drops.append(drop1)
                                break
                            elif (mark_1 == drop2) or (mark_2 == drop2):
                                drops.append(drop2)
                                break

                checked_group.drop_markups(drops)

                if label_with_markup := self._aggregate(
                    checked_group,
                    threshold,
                    confidence,
                ):
                    accepted = set(annotators) - set(rejected)
                    return (
                        self._res_to_group(
                            label_with_markup, group.name, method="_soft"
                        ),
                        accepted,
                        rejected,
                    )

        if label_with_markup is None:
            accepted = set()
            rejected = set()
        elif not label_with_markup:
            accepted = set(group.get_annotators())
            rejected = set()
        else:
            accepted = set(accepted)
            rejected = set(rejected)
            accepted -= accepted & rejected
        return (
            self._res_to_group(label_with_markup, group.name, method="_soft"),
            accepted,
            rejected,
        )
