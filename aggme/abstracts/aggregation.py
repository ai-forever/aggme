from abc import ABC, abstractmethod
from itertools import combinations
from typing import Callable, Optional, Tuple

from sklearn.cluster import DBSCAN, MeanShift
from tqdm import tqdm

from aggme.utils.dataclass import MarkupGroup
from aggme.utils.processing import AnnotationData


class AbstractAggregation(ABC):
    CLUSTER_METHODS = {
        "mean_shift": MeanShift(n_jobs=8),
        "dbscan": DBSCAN(n_jobs=8),
    }
    ANNOTATION_TYPE = None
    DEFAULT_CLUSTER_METHOD = "mean_shift"

    def __init__(
        self,
        cluster_method_name: str = "mean_shift",
        cluster_method_params: dict = {"bandwidth": 0.05},
        min_annotations: int = 3,
    ):
        self.MIN_ANNOTATIONS = min_annotations
        self.cluster_method = self.CLUSTER_METHODS[cluster_method_name].set_params(
            **cluster_method_params
        )
        self.method: Callable

    @abstractmethod
    def _aggregate(
        self,
        group: MarkupGroup,
        threshold: float = 0.5,
        confidence: float = 1.0,
    ):
        pass

    @staticmethod
    def _res_to_group(
        res,
        group_name: str,
        method: str = "",
    ) -> Optional[MarkupGroup]:
        raise NotImplementedError

    @staticmethod
    def get_group_by_name(
        results: dict,
        group_name: str,
    ) -> Tuple:
        """Get group by name

        Parameters
        ----------
        results: dict
            Aggregation results
        group_name: str
            Searched group name

        Yields
        -------
        Tuple:
            method, group
        """
        keys = results.keys()
        for method in keys:
            for group in results[method]:
                if group[0].name == group_name:
                    return method, group
        return None, None

    def get_aggregation_results(
        self,
        data: AnnotationData,
        threshold: float,
        confidence: float,
    ):
        """Apply all aggregation methods alternately

        Parameters
        ----------
        data: AnnotationData
            All data to aggregate. Do not forget to apply .load_markups_iter() method
        threshold: float
            IoU threshold
        confidence: float
            Markups confidence

        Yields
        -------
        Tuple:
            Initial group, resulted group, accepted annotators, rejected annotators
        """
        results = {"hard": [], "drop": [], "soft": [], "fail": []}

        for group in tqdm(data.groups):
            result = self.hard_aggregation(
                group=group,
                threshold=threshold,
            )
            accepted = set(group.get_annotators())
            rejected = set()
            if result:
                results["hard"].append((group, result, accepted, rejected))
            else:
                result, accepted, rejected = self.drop_aggregation(
                    group=group,
                    threshold=threshold,
                )
                if result:
                    results["drop"].append((group, result, accepted, rejected))
                else:
                    result, accepted, rejected = self.soft_aggregation(
                        group=group,
                        threshold=threshold,
                        confidence=confidence,
                    )
                    if result:
                        results["soft"].append((group, result, accepted, rejected))
                    else:
                        results["fail"].append((group, None, None, None))
        return results

    def drop_aggregation(
        self,
        group: MarkupGroup,
        threshold: float = 0.5,
        confidence: float = 1.0,
        drop_num: int = 1,
    ) -> Tuple:
        """Select aggregation for bboxes.

        Parameters
        ----------
        group : MarkupGroup
            Group of annotations
        threshold : float
            IoU threshold
        confidence: float
            Markups confidence
        drop_num : int
            Number of items to drop

        """
        drop_marks, accepted, rejected = None, None, None
        group.reset()
        if (
            combs := len(group.get_groups_by_annotators())
        ) > self.MIN_ANNOTATIONS:  # noqa
            if combs - drop_num >= self.MIN_ANNOTATIONS:
                assignments = list(map(lambda x: x.annotator, group.data))

                for comb in combinations(
                    group.get_groups_by_annotators(), combs - drop_num
                ):
                    data = []
                    for subgroup in comb:
                        for mark in subgroup:
                            data.append(mark)

                    check_group = MarkupGroup(
                        name=group.name,
                        data=data,
                        dimension=group.dimension,
                        relative=group.relative,
                    )
                    drop_marks = self._aggregate(check_group, threshold, confidence)
                    if drop_marks:
                        accepted = set(
                            list(map(lambda x: x.annotator, check_group.data))
                        )
                        rejected = set(assignments) - set(accepted)
                        break

        return (
            self._res_to_group(drop_marks, group.name, method="_drop"),
            accepted,
            rejected,
        )

    def soft_aggregation(
        self,
        group: MarkupGroup,
        threshold: float = 0.5,
        confidence: float = 0.7,
    ):
        """Soft aggregation for bboxes.

        Parameters
        ----------
        group : MarkupGroup
            Group of markups
        threshold : float
            Similarity metric threshold
        confidence: float
            Markups confidence
        Returns
        -------
        list
            Aggregation, accepted and rejected assignments and failed flag
        """
        raise NotImplementedError

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
        raise NotImplementedError
