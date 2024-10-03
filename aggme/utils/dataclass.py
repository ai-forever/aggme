from dataclasses import dataclass
from itertools import combinations
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from typing_extensions import Self


class UnknownStructureException(Exception):
    """Raises when wrong structure passed"""


@dataclass
class BboxMarkup:
    """Dataclass for single bbox markup

    Parameters
    ----------
    label: str
        label of current markup
    coordinates: tuple
        x, y, w, h
    annotator: str
        assignment id / network name / custom name

    """

    label: Optional[str]
    annotator: Optional[str]
    x: Union[int, float]
    y: Union[int, float]
    w: Union[int, float]
    h: Union[int, float]

    def __init__(
        self,
        label: str,
        coordinates: Union[tuple, list],
        annotator: str,
    ):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.w = coordinates[2]
        self.h = coordinates[3]
        self.label = label
        self.annotator = annotator

    def __repr__(self):
        return (
            f"//annotator: {self.annotator}\n"
            f"label: {self.label};\n"
            f"x: {round(self.x, 10)}, y: {round(self.y, 10)}, w: {round(self.w, 10)}, h: {round(self.h, 10)}\\\\\n\n"
        )

    def __eq__(self, other: Self):
        return (
            (self.expand_markup() == other.expand_markup())
            & (self.annotator == other.annotator)
            & (self.label == other.label)
        )

    @staticmethod
    def check_label_with_markups(
        iou_func: Callable,
        items_with_markups: dict,
        threshold: float = 0.5,
    ) -> list:
        """

        Parameters
        ----------
        iou_func: Callable
            IOU calculation function
        items_with_markups: dict
            Markups data
        threshold: float
            IOU threshold

        Returns
        -------
        list
            labels and markups for them if current markups fits with IOU metric

        """

        def check_similarity(
            method: Callable,
            markups: list,
            threshold: float,
        ) -> bool:
            # Get all pairs from list and calc IoUs:
            pair = [pp for pp in combinations(markups, 2)]
            ious = [method(aa, bb) for aa, bb in pair]

            # Pass if IoU > threshlod
            if len(ious):
                return sum(ious) / len(ious) > threshold
            else:
                return False

        label_with_markup = []
        for _, item in items_with_markups.items():
            if check_similarity(iou_func, item, threshold):
                keys = list(map(lambda x: x.label, item))
                markups = list(map(lambda x: x.expand_markup(), item))

                markup = np.array(markups)
                markup = list((markup.sum(axis=0) / markup.shape[0]).round(8))

                values, counts = np.unique(np.array(keys), return_counts=True)
                key = [
                    values[i] for i in range(len(values)) if counts[i] == max(counts)
                ][0]
                label_with_markup.append((key, markup))
            else:
                return []

        return label_with_markup

    @staticmethod
    def bbox_coco_to_voc(bbox: list) -> list:
        """Convert COCO format [x, y, w, h] to PASCAL VOC [x1, y1, x2, y2]"""
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    @staticmethod
    def iou_coco(bbox1, bbox2) -> float:
        """Intersection over Union

        Parameters
        ----------
        bbox1 : list
            First bounding box: [x, y, w, h]
        bbox2 : list
            Second bounding box: [x, y, w, h]

        Returns
        -------
        float
            Intersection over Union

        """

        assert bbox1.w >= 0
        assert bbox1.h >= 0
        assert bbox2.w >= 0
        assert bbox2.h >= 0

        area1, area2 = bbox1.w * bbox1.h, bbox2.w * bbox2.h

        x1 = max(bbox1.x, bbox2.x)
        y1 = max(bbox1.y, bbox2.y)
        x2 = min(bbox1.x + bbox1.w, bbox2.x + bbox2.w)
        y2 = min(bbox1.y + bbox1.h, bbox2.y + bbox2.h)

        if (intersection := abs(max((x2 - x1, 0)) * max((y2 - y1), 0))) == 0:
            return 0

        iou = intersection / float(area1 + area2 - intersection + 1e-10)

        assert 0.0 <= iou <= 1.0
        return iou

    @staticmethod
    def bbox_voc_to_coco(bbox: list) -> list:
        """Convert VOC format [x1, y1, x2, y2] to PASCAL COCO [x, y, w, h]"""
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]

    def set_label(self, label: str):
        """Set label name for markup

        Parameters
        ----------
        label: str
            markup label

        """
        self.label = label

    def set_data(self, other: Self):
        """Set new coordinates (x, y, w, h) from another markup

        Parameters
        ----------
        other: BboxMarkup | IntervalMarkup
            other markup

        """
        self.x, self.y, self.w, self.h = other.expand_markup()

    def expand_markup(
        self,
    ) -> Tuple[int | float, int | float, int | float, int | float]:
        """Get markup coordinates

        Returns
        -------
        coordinates tuple - x, y, w, h

        """
        return self.x, self.y, self.w, self.h

    def get_center(self) -> Tuple[int | float, int | float]:
        """Get center coordinates for bbox markup

        Returns
        -------
        center coordinates - xc, yc

        """
        center = (self.x + self.w / 2, self.y + self.h / 2)
        return center

    def check_matching(self, kk: tuple) -> bool:
        """Check if another coordinates is inside current bbox

        Parameters
        ----------
        kk: tuple
            xd, yd coordinates

        Returns
        -------
        bool
            True if kk(xd, yd) point is inside of box, False otherwise

        """
        if (self.x < kk[0] < (self.x + self.w)) and (
            self.y < kk[1] < (self.y + self.h)
        ):
            return True
        return False

    def point_detection(self, point_threshold: float) -> bool:
        """Check if current bbox is point

        Parameters
        ----------
        point_threshold: float
            Threshold to detect points. The lower the value, the smaller the points will be detected
        Returns
        -------
        bool
            True if markup is a point, False otherwise

        """
        if self.w < point_threshold or self.h < point_threshold:
            return True
        else:
            return False


@dataclass
class MaskMarkup:
    """Dataclass for single mask markup

    Parameters
    ----------
    label: str
        label of current markup
    mask: np.array
        Binary markup mask
    annotator:
        assignment id / network name / custom name

    """

    label: str
    annotator: str
    mask: np.array
    _ious: list

    def __init__(
        self,
        label: str,
        mask: np.array,
        annotator: str,
    ):
        self.label = label
        self.mask = mask
        self.annotator = annotator
        self._ious = []

    def __eq__(self, other):
        return (
            (self.mask == other.mask).all()
            & (self.annotator == other.annotator)
            & (self.label == other.label)
        )  # noqa

    @staticmethod
    def polygon_iou(polygon1: List, polygon2: List) -> float:
        """
        Intersection over Union for polygon with shapely
        Do not mix points inside polygons. It can change shape of
        polygon and intersection area.

        Parameters
        ----------
        polygon1 : list
            First polygon: [(x1, y1), ..., (xn, yn)]
        polygon2 : list
            Second polygon: [(x1, y1), ..., (xn, yn)]

        Returns
        -------
        float
            Intersection over Union

        """
        assert (
            len(polygon1) < 3
        ), "Polygon should contain at least 3 points, but {} found".format(
            len(polygon1)
        )
        assert (
            len(polygon2) < 3
        ), "Polygon should contain at least 3 points, but {} found".format(
            len(polygon2)
        )

        p1 = Polygon(polygon1).buffer(0)
        p2 = Polygon(polygon2).buffer(0)
        return p1.intersection(p2).area / p1.union(p2).area

    @staticmethod
    def polygons_to_keypoints(markup: dict) -> list:
        """Polygon from Toloka API as list of keypoints.

        Parameters
        ----------
        markup : dict
            Polygon as a dict, includes list of points

        Returns
        -------
        list
            List of coordinates for polygon

        """
        if (shape := markup.get("shape")) == "polygon":
            polygon = []
            for point in markup.get("points"):
                x, y = point.get("left"), point.get("top")
                polygon.append((x, y))
            return polygon
        else:
            raise UnknownStructureException(f"Need polygon structure, but got {shape}")

    @staticmethod
    def polygon_to_mask(polygon: list, size):
        background = Image.new("L", (size[0], size[1]), 0)
        points = [(point[0] * size[0], point[1] * size[1]) for point in polygon]
        ImageDraw.Draw(background).polygon(points, outline=1, fill=1)
        mask = np.array(background)
        return mask

    @staticmethod
    def get_binary_mask(mask: np.ndarray, threshold: float):
        """Convert mask to binary via threshold"""
        return ((mask >= threshold) * 255).astype(np.uint8)

    def expand_markup(self):
        return self.mask

    def get_avg_iou(self) -> float:
        """Get average IOU received from all caclulations

        Returns
        -------
        float
            average IOU

        """
        res = 0
        if len(self._ious):
            res = sum(self._ious) / len(self._ious)
            self._ious = []
        return res

    def add_iou(self, added: float):
        """Add IOU with other mask to all IOUs of current markup

        Parameters
        ----------
        added: float
            Added float number (IOU)

        """
        self._ious.append(added)

    def iou(self, other: Self) -> float:
        """Get IoU compare with other mask markup

        Parameters
        ----------
        other: MaskMarkup
            Other mask markup

        Returns
        -------
        float
            iou score

        """
        if np.max(self.mask) == np.max(other.mask) == 0:
            return 1
        intersection = np.logical_and(self.mask, other.mask)
        union = np.logical_or(self.mask, other.mask)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score


@dataclass
class IntervalMarkup(BboxMarkup):
    """Dataclass for single interval markup

    Parameters
    ----------
    label: str
        label of current markup
    coordinates: tuple
        start, end
    annotator: str
        assignment id / network name / custom name

    """

    def __init__(
        self,
        label,
        coordinates,
        annotator,
    ):
        coordinates = [coordinates[0], coordinates[1], None, None]
        super().__init__(label, coordinates, annotator)

    def __repr__(self):
        return (
            f"//annotator: {self.annotator}\n"
            f"label: {self.label};\n"
            f"left: {round(self.x, 10)}, right: {round(self.y, 10)}\\\\\n\n"
        )

    @staticmethod
    def iou_1d(interval1, interval2) -> float:
        """Intersection over Union

        Parameters
        ----------
        interval1 : IntervalMarkup
            First interval
        interval2 : IntervalMarkup
            Second interval

        Returns
        -------
        float
            Intersection over Union

        """
        assert interval1.y - interval1.x >= 0
        assert interval2.y - interval2.x >= 0

        # Another method to compute union
        # length1, length2 = interval1[1] - interval1[0], interval2[1] - interval2[0]
        # union = length1 + length2 - intersection

        first_begin = interval1.x if interval1.x <= interval2.x else interval2.x
        second_begin = interval1.x if first_begin != interval1.x else interval2.x

        first_end = interval1.y if interval1.y <= interval2.y else interval2.y
        second_end = interval1.y if first_end != interval1.y else interval2.y

        if (intersection := abs(first_end - second_begin)) == 0:
            return 0

        union = abs(second_end - first_begin)
        iou = intersection / float(union + 1e-10)

        assert 1.0 >= iou >= 0.0
        return iou

    @staticmethod
    def convert_interval_annotation(annotation: list) -> list:
        """Convert single string with annotations to list

        Parameters
        ----------
        annotation : list

        Returns
        -------
        list
            List of intervals with or without labels

        """
        if not annotation:
            return []

        result_annotation = []
        for mark in annotation:
            data = [mark.get("label"), (mark.get("begin"), mark.get("end"))]
            result_annotation.append(data)
        return result_annotation

    def set_data(self, other: Self):
        """Set new coordinates (start, end) from another markup

        Parameters
        ----------
        other: IntervalMarkup
            other markup

        """
        self.x, self.y = other.expand_markup()

    def get_center(self) -> int | float:
        """Get center coordinates for bbox markup

        Returns
        -------
        int | float
            center coordinate

        """
        begin, end = self.expand_markup()
        center = begin + (end - begin) // 2
        return center

    def check_matching(self, kk: int | float) -> bool:
        """Check if another coordinate is inside current interval

        Parameters
        ----------
        kk: int | float
            point coordinate

        Returns
        -------
        bool
            True if kk point is inside of interval, False otherwise

        """
        begin, end = self.expand_markup()
        if begin < kk < end:
            return True
        return False

    def point_detection(self, point_threshold: float) -> bool:
        """Check if current bbox is point

        Parameters
        ----------
        point_threshold: float
            Threshold to detect points. The lower the value, the smaller the points will be detected
        Returns
        -------
        bool
            True if markup is a point, False otherwise

        """
        begin, end = self.expand_markup()
        if end - begin < point_threshold:
            return True
        return False

    def expand_markup(self) -> Tuple[int | float, int | float]:
        """Get markup coordinates

        Returns
        -------
        coordinates tuple - start, end

        """
        return self.x, self.y


class MarkupGroup:
    """Dataclass for group of markups

    Parameters
    ----------
    name: str
        Name of current group. E.g. "example" (.jpg)
    data: list
        (BboxMarkup | MaskMarkup | IntervalMarkup) values
    dimension: tuple
        Item's dimension
    relative: bool
        Define if markups in group are having relative coordinates

    """

    name: str
    markup_type = {
        "bbox": BboxMarkup,
        "mask": MaskMarkup,
        "interval": IntervalMarkup,
    }
    data: List[BboxMarkup | MaskMarkup | IntervalMarkup]

    dimension: Optional[Tuple[int, int]]
    relative: bool
    #     _deleted_data: List[BboxMarkup | MaskMarkup | IntervalMarkup] = list()
    point_threshold: float = 0.001
    duplicate_threshold: float = 0.85

    def __init__(
        self,
        name: str = None,
        data: List[BboxMarkup | MaskMarkup | IntervalMarkup] = None,
        relative: bool = None,
        dimension: Optional[Tuple[int, int]] = None,
        point_threshold: float = 0.001,
        duplicate_threshold: float = 0.85,
    ):
        self.name = name
        self.data = data
        self.relative = relative
        self.dimension = dimension
        self.point_threshold = point_threshold
        self.duplicate_threshold = duplicate_threshold
        self._deleted_data = []
        if self.name is None:
            raise ValueError

    def __repr__(self):
        return f"MarkupGroup object\nname: {self.name}\ndata_len: {len(self.data)}\nannotators_len: {len(self.get_annotators())}"

    def _remove(self, mark: BboxMarkup | MaskMarkup | IntervalMarkup) -> bool:
        """Remove markup from group

        Parameters
        ----------
        mark: BboxMarkup | MaskMarkup | IntervalMarkup
            markup to delete

        Returns
        -------
        bool
            True if markup was in the group and was deleted, False otherwise

        """
        for i, mark_ in enumerate(self.data):
            if mark_ == mark:
                self._deleted_data.append(self.data.pop(i))
                return True
        return False

    def get_duplicates(
        self,
        iou_func,
    ) -> List[BboxMarkup | MaskMarkup | IntervalMarkup]:
        """Get duplicated markups in group

        Parameters
        ----------
        iou_func: Callable
            Function to calculate IOU for current markup type

        Returns
        -------
        duplicates: list
            [{iou_1: (markup_1, markup_2), iou_2: (markup_2, markup_3)}, ...]

        """
        duplicates = []  # annotators
        ious = {}
        users = list(set(map(lambda x: x.annotator, self.data)))
        for user in users:
            pairs = [
                pp
                for pp in combinations(
                    list(filter(lambda x: x.annotator == user, self.data)), 2
                )
            ]
            for pair in pairs:
                ious[iou_func(pair[0], pair[1])] = (pair[0], pair[1])

            for iou in ious.keys():
                if iou >= self.duplicate_threshold:
                    duplicates.append(ious[iou][0])
        return duplicates

    def get_points(self) -> List[BboxMarkup | MaskMarkup | IntervalMarkup]:
        points = []
        for mark in self.data:
            if mark.point_detection(point_threshold=self.point_threshold):
                points.append(mark)
        return points

    def get_groups_by_annotators(
        self,
    ) -> List[List[BboxMarkup | MaskMarkup | IntervalMarkup]]:
        """Groups of markups selected by annotators

        Returns
        -------
        list
            Selected by annotators groups list

        """
        users = list(set(map(lambda x: x.annotator, self.data)))
        groups = []
        for user in users:
            group = []
            for mark in self.data:
                if mark.annotator == user:
                    group.append(mark)
            groups.append(group)
        return groups

    def get_groups_by_labels(self) -> list:
        """Groups of markups selected by labels

        Returns
        -------
        list
            Selected by labels groups list

        """
        labels = list(set(map(lambda x: x.label, self.data)))
        groups = []
        for label in labels:
            group = []
            for mark in self.data:
                if mark.label == label:
                    group.append(mark)
            groups.append(group)
        return groups

    def drop_markups(
        self,
        markups: List[BboxMarkup | MaskMarkup | IntervalMarkup],
    ) -> List[str]:
        """Remove some markups from group

        Parameters
        ----------
        markups: List[BboxMarkup | MaskMarkup | IntervalMarkup]
            Markups to drop
        Returns
        -------
        list
            Annotators which was deleted from group

        """
        rejected = []
        for drop in markups:
            self._remove(drop)
            rejected.append(drop.annotator)

        return rejected  # annotator

    def fill_missing(self, type_: str = "mask"):
        """Add missing markups by labels to each annotator.
            Only works if num_annotators < num_labels

        Parameters
        ----------
        type_: str = "mask"
            markup type, optional between "mask", "bbox" and "interval"

        """
        if type_ != "mask":
            raise NotImplementedError

        if len(set([len(x) for x in self.get_groups_by_labels()])) > 1:
            labels = set([x[0].label for x in self.get_groups_by_labels()])
            for annotator_markups in self.get_groups_by_annotators():
                if len(annotator_markups) < len(labels):
                    missing_labels = labels - set([x.label for x in annotator_markups])
                    for missing_label in list(missing_labels):
                        self.data.append(
                            self.markup_type[type_](
                                label=missing_label,
                                annotator=annotator_markups[0].annotator,
                                mask=np.zeros(self.dimension),
                            )
                        )

    def get_annotators(self) -> List[str]:
        """Get all annotators from group

        Returns
        -------
        list[str]
            Annotators list

        """
        return list(set(map(lambda x: x.annotator, self.data)))

    def get_labels(self) -> List[str]:
        """Get all labels in group

        Returns
        -------
        list
            All labels in group
        """
        return list(set(map(lambda x: x.label, self.data)))

    def reset(self):
        self.data.extend(self._deleted_data)
        self._deleted_data = []

    def convert_to(self, to: str | Callable = "coco") -> list:
        """Convert group data to coco, voc, relative, etc. types

        Parameters
        ----------
        to: [str, Callable]
            To which type convert data ('voc', 'coco', YOUR_PROCESS_FUNC: Callable)

            Example of function `to`:
            def to_custom(data):
                ## data == markup.expand_markup()
                return list(map(lambda x: x * 10, data))
        Returns
        -------
        list
            converted labels with markups

        """

        def process_(x):
            return x

        if len(self.data) == 0:
            raise ValueError("No data to convert")
        if isinstance(to, str):
            if isinstance(self.data[0], BboxMarkup):
                if to == "coco":
                    process = process_
                elif to == "voc":
                    process = MarkupGroup.bbox_coco_to_voc
                else:
                    raise NotImplementedError(
                        f"No such method `{to}`. "
                        f"Please choose existing method "
                        f"or provide custom function"
                    )
            elif isinstance(self.data[0], MaskMarkup):
                raise NotImplementedError("No methods implemented for masks")
            elif isinstance(self.data[0], IntervalMarkup):
                raise NotImplementedError("No methods implemented for intervals")
        else:
            process = to

        res = list(map(lambda x: (x.label, process(x.expand_markup())), self.data))
        return res

    # def draw(
    #         self,
    #         accepted: Optional[list] = None,
    #         rejected: Optional[list] = None,
    #         res_hard: Optional[list] = None,
    #         res_soft: Optional[list] = None,
    #         res_drop: Optional[list] = None,
    #         mode: str = "full",
    # ):
    #     if len(self.data) == 0:
    #         raise ValueError("No data to visualize")
    #     if isinstance(self.data[0], BboxMarkup):
    #         draw_bbox_group(self, accepted=accepted, rejected=rejected, res_hard=res_hard, res_soft=res_soft)
    #     elif isinstance(self.data[0], MaskMarkup):
    #         draw_mask_group(self, accepted=accepted, rejected=rejected, res_hard=res_hard, res_soft=res_soft)
    #     else:
    #         raise NotImplementedError
