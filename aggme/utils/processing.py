"""
TEMPLATE = {  # BBoxes
    "sample_id": {
        "markups": {
            "annotator_id": [
                {"label_1": [[x0, y0, h0, w0], [x1, y1, h1, w1]], "label_2": [[x2, y2, h2, w2]]},
                {"label_1": [[x3, y3, h3, w3], [x4, y4, h4, w4]]}
            ]
        },
        "dimension": dimension,
    }
}
TEMPLATE = {  # Masks
    "sample_id": {
        "markups": {
            "annotator_id": [
                 {"label_1": [*mask*], "label_2": [*mask*]},  # mask - binary np.array()
         ]
         },
        "dimension": dimension,
    }
}
"""

import os
from ast import literal_eval
from collections import defaultdict
from typing import Any, Callable, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from aggme.utils.dataclass import BboxMarkup, IntervalMarkup, MarkupGroup, MaskMarkup

default_label = "default_label"


def literal_eval_to_markups(annotation: Union[str, List]) -> Optional[List]:
    """Convert annotations to list if possible

    Parameters
    ----------
    annotation : list, str

    Returns
    -------
    list
        List of markups with or without labels

    """
    if isinstance(annotation, list):
        markup = annotation
    elif isinstance(annotation, str):
        markup = literal_eval(annotation)
    else:
        markup = []

    if not markup:
        return None
    return markup


def toloka_bboxes_to_template(
    df: pd.DataFrame,
    markup_col="OUTPUT:markup",
    sample_name_col="INPUT:image",
    annotators_col="ASSIGNMENT:assignment_id",
):
    data = {}
    dimension = None
    for ind, v in df.groupby(sample_name_col):
        raw_markups = v[markup_col].apply(literal_eval_to_markups).tolist()
        annotators = v[annotators_col].tolist()
        markups = defaultdict(list)
        for i, temp in enumerate(raw_markups):
            if temp is None:
                continue
            tmp = defaultdict(list)
            for markup in temp:
                x = markup.get("left")
                y = markup.get("top")
                w = markup.get("width")
                h = markup.get("height")
                label = markup.get("label")
                tmp[label].append([x, y, w, h])

            markups[annotators[i]].append(tmp)

        sample_id = os.path.splitext(os.path.basename(ind))[0]

        if sample_id not in data.keys():
            data[sample_id] = {"markups": markups, "dimension": dimension}
        else:
            data[sample_id]["markups"].update(markups)
            data[sample_id]["dimension"] = dimension
    return data


def toloka_masks_to_template(
    df: pd.DataFrame,
    markup_col="OUTPUT:markup",
    sample_name_col="INPUT:image",
    annotators_col="ASSIGNMENT:assignment_id",
    dimension_col="dimension",
):
    for column in [markup_col, sample_name_col, annotators_col, dimension_col]:
        if column not in df.columns:
            raise KeyError(f"No {column} in passed DataFrame")

    data = {}
    for ind, v in df.groupby(sample_name_col):
        raw_markups = v[markup_col].apply(literal_eval_to_markups).tolist()
        annotators = v[annotators_col].tolist()
        dimensions = v[dimension_col].tolist()
        dimension = literal_eval(dimensions[0])
        markups = defaultdict(list)
        for i, temp in enumerate(raw_markups):
            tmp = defaultdict(list)
            for markup in temp:
                label = markup.get("label")
                polygon = MaskMarkup.polygons_to_keypoints(markup)
                tmp[label].append(MaskMarkup.polygon_to_mask(polygon, dimension))

            for key in tmp.keys():
                tmp[key] = [np.sum(tmp[key], axis=0)]

            markups[annotators[i]].append(tmp)

        sample_id = os.path.splitext(os.path.basename(ind))[0]

        if sample_id not in data.keys():
            data[sample_id] = {"markups": markups, "dimension": dimension}
        else:
            data[sample_id]["markups"].update(markups)
            data[sample_id]["dimension"] = dimension
    return data


def synthetic_bboxes_to_template(markups_data: list):
    def mark_to_template(markup: list):
        res = defaultdict(list)
        for label, coords in markup:
            res[label].append(coords)
        return res

    data = {}
    params = []
    for markups in markups_data:
        sample_id = markups[0]
        assignments = markups[1]
        annotates = markups[2]
        params.append(markups[3])

        tmp = defaultdict(list)
        for i, mark in enumerate(annotates):
            tmp[assignments[i]].append(mark_to_template(mark))

        data[sample_id] = {}
        data[sample_id]["dimension"] = None
        data[sample_id]["markups"] = tmp
    return data


def abc_intervals_to_template(
    df: pd.DataFrame,
    sample_name_col="INPUT:video",
    markup_col="OUTPUT:video",
    annotators_col="ASSIGNMENT:user_id",
):
    data = {}
    dimension = None
    for ind, v in df.groupby(sample_name_col):
        raw_markups = v[markup_col].apply(literal_eval_to_markups).tolist()
        annotators = v[annotators_col].tolist()
        markups = defaultdict(list)
        for i, temp in enumerate(raw_markups):
            tmp = defaultdict(list)
            broken = False
            for markup in temp:
                x = markup.get("begin_frame")
                y = markup.get("end_frame")
                if (x is None) or (y is None):
                    broken = True
                label = markup.get("label")
                if (label is None) or (label == "") or (isinstance(label, list)):
                    label = default_label
                tmp[label].append([x, y])
            if broken:
                continue
            markups[annotators[i]].append(tmp)

        sample_id = os.path.splitext(os.path.basename(ind))[0]

        if sample_id not in data.keys():
            data[sample_id] = {"markups": markups, "dimension": dimension}
        else:
            data[sample_id]["markups"].update(markups)
            data[sample_id]["dimension"] = dimension
    return data


class AnnotationData:
    """
    Top-level utils to load markups
    """

    groups: List[MarkupGroup]
    groups_iterable: Iterable[MarkupGroup] = None
    markup_type: str
    point_threshold: float = 0.001
    duplicate_threshold: float = 0.85

    markup_class_choice = {
        "bboxes": BboxMarkup,
        "mask": MaskMarkup,
        "interval": IntervalMarkup,
    }

    def __init__(
        self,
        markup_type: str,
        point_threshold: Optional[float] = None,
        duplicate_threshold: Optional[float] = None,
    ):
        if markup_type not in self.markup_class_choice.keys():
            raise KeyError(
                f"No such markup type <<{markup_type}>>."
                f"Select from {list(self.markup_class_choice.keys())}"
            )
        self.markup_type = markup_type
        self.point_threshold = (
            point_threshold if point_threshold is not None else self.point_threshold
        )
        self.duplicate_threshold = (
            duplicate_threshold
            if duplicate_threshold is not None
            else self.duplicate_threshold
        )

    def load_markups(self, input_: Any, process_func: Callable):
        data = process_func(input_)
        self.groups = self._preprocess(data)

    def load_markups_iter(self, input_: Any, process_func: Callable):
        data = process_func(input_)
        self.groups_iterable = self._preprocess_iter(data)

    def _preprocess(self, data: dict) -> List[MarkupGroup]:
        result = []
        for sample_id, val in data.items():
            marks = []
            for annotator, markups in val["markups"].items():
                for markup in markups:
                    for class_id, coordinates in markup.items():
                        for coords in coordinates:
                            marks.append(self._get_markup(class_id, coords, annotator))
            dimension = val["dimension"]
            result.append(
                MarkupGroup(
                    name=sample_id,
                    data=marks,
                    relative=False,
                    dimension=dimension,
                    point_threshold=self.point_threshold,
                    duplicate_threshold=self.duplicate_threshold,
                )
            )
        return result

    def _preprocess_iter(self, data: dict):
        for sample_id, val in data.items():
            marks = []
            for annotator, markups in val["markups"].items():
                for markup in markups:
                    for class_id, coordinates in markup.items():
                        for coords in coordinates:
                            marks.append(self._get_markup(class_id, coords, annotator))
            dimension = val["dimension"]
            group = MarkupGroup(
                name=sample_id,
                data=marks,
                relative=False,
                dimension=dimension,
                point_threshold=self.point_threshold,
                duplicate_threshold=self.duplicate_threshold,
            )
            yield group

    def _get_markup(
        self,
        class_id,
        coordinates,
        annotator,
    ) -> Union[MaskMarkup, BboxMarkup, IntervalMarkup]:
        return self.markup_class_choice[self.markup_type](
            class_id, coordinates, annotator
        )
