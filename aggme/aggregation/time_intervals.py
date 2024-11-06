import json
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
    def convert_and_save_markups(
        results: dict,
        file_name: str = None,
    ) -> list:
        """Convert to json and save markups

        Parameters
        ----------
        results: dict
            Aggregation results
        file_name: str = None
            File name to save

        Yields
        -------
        list
        """
        keys = results.keys()
        markup_results = []
        for method in keys:
            if method == "fail":
                continue
            for group in results[method]:
                original_group, result_group, accepted, rejected = group
                markup_result = []
                for i, markup in enumerate(result_group.data):
                    left, right = markup.expand_markup()
                    data_class = {
                        "left": left,
                        "right": right,
                        "label": markup.label,
                    }
                    markup_result.append(data_class)
                markup_results.append(
                    {
                        "markup": markup_result,
                        "image": group[0].name,
                        "method": method,
                    }
                )

        if file_name:
            json_data = json.dumps(markup_results, indent=4)
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(json_data)

        return markup_results

    @staticmethod
    def markups_comparison(
        results1: dict,
        results2: dict,
        threshold: float = 1.0,
    ) -> list:
        """Markups comparison

        Parameters
        ----------
        results1: dict
            Aggregation results
        results2: dict
            Aggregation results
        threshold: float = 1.0
            IoU threshold

        Yields
        -------
        list
        """
        comparison_results = []
        for element1 in results1:
            image1 = element1["image"]
            element2 = None
            # ищем совпадение по image
            for element2 in results2:
                image2 = element2["image"]
                if image1 == image2:
                    break
                element2 = None
            # сравниваем найденный image
            if element2:
                classes1 = [markup["label"] for markup in element1["markup"]]
                classes2 = [markup["label"] for markup in element2["markup"]]
                count_match = len(classes1) == len(classes2)
                class_count_match = len(set(classes1)) == len(set(classes2))

                good_indexes = []
                markup_data1 = element1["markup"].copy()
                markup_data2 = element2["markup"].copy()
                for index1, markup1 in enumerate(markup_data1):
                    max_iou = -1
                    max_iou_index2 = -1
                    for index2, markup2 in enumerate(markup_data2):
                        assert markup1["right"] - markup1["left"] >= 0
                        assert markup2["right"] - markup2["left"] >= 0
                        first_begin = (
                            markup1["left"]
                            if markup1["left"] <= markup2["left"]
                            else markup2["left"]
                        )
                        second_begin = (
                            markup1["left"]
                            if first_begin != markup1["left"]
                            else markup2["left"]
                        )
                        first_end = (
                            markup1["right"]
                            if markup1["right"] <= markup2["right"]
                            else markup2["right"]
                        )
                        second_end = (
                            markup1["right"]
                            if first_end != markup1["right"]
                            else markup2["right"]
                        )
                        if (intersection := abs(first_end - second_begin)) == 0:
                            iou = 0
                        else:
                            iou = round(
                                intersection
                                / float(abs(second_end - first_begin) + 1e-10),
                                2,
                            )
                        assert 0.0 <= iou <= 1.0
                        if max_iou < iou:
                            max_iou = iou
                            max_iou_index2 = index2

                    if max_iou >= threshold:
                        good_indexes.append(index1)
                        del markup_data2[max_iou_index2]

                iou_match_percent = int((len(good_indexes) / len(markup_data1)) * 100)

                comparison_result = {
                    "image": image1,
                    "count_match": count_match,
                    "class_count_match": class_count_match,
                    "iou_match_percent": iou_match_percent,
                }
                comparison_results.append(comparison_result)
            else:
                comparison_result = {
                    "image": image1,
                    "count_match": None,
                    "class_count_match": None,
                    "iou_match_percent": None,
                }
                comparison_results.append(comparison_result)
        return comparison_results

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
