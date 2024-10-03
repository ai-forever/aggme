import copy

import matplotlib.pyplot as plt
import numpy as np

from aggme.utils.dataclass import MaskMarkup


def print_boxes_result(annotates, aggregated):
    def get_color_by_mode(mode):
        if mode == "no gesture":
            color_ = "green"
        elif mode == "gesture":
            color_ = "red"
        else:
            color_ = "white"
        return color_

    annotations = copy.deepcopy(annotates[2])
    assignments = annotates[1]
    annotations.append(aggregated)

    fig, axes = plt.subplots(1, len(annotations), figsize=(24, 5))
    for i in range(len(annotations)):
        ann = annotations[i]

        if not ann:
            ann = [("", [0, 0, 0, 0])]

        for j in range(len(ann)):
            color = get_color_by_mode(ann[j][0])
            rectangle = plt.Rectangle(
                (ann[j][1][0], ann[j][1][1]),
                ann[j][1][2],
                ann[j][1][3],
                fc="white",
                ec=color,
                alpha=0.6,
                lw=2,
            )
            axes[i].add_patch(rectangle)
            if i == len(annotations) - 1:
                axes[i].set_title("aggregation")
            else:
                axes[i].set_title(f"annotation {i + 1}\n {assignments[i][-5:]}")
        axes[i].grid()
    plt.show()


def print_masks_result(annotates, aggregated, image, dimension):
    annotations = annotates[2]
    assignments = annotates[1]

    fig, axes = plt.subplots(1, len(annotations) + 1, figsize=(24, 5))
    for i in range(len(annotations)):
        ann = annotations[i]

        if not ann:
            mask = np.zeros((dimension[1], dimension[0]))
        else:
            masks = [MaskMarkup.polygon_to_mask(mark[1], dimension) for mark in ann]
            mask = sum(masks)

        axes[i].set_title(f"annotation {i + 1}\n {assignments[i][-5:]}")

        mask[mask % 2 == 1] = 1
        mask[mask % 2 == 0] = 0

        axes[i].imshow(image)
        axes[i].imshow(mask, alpha=0.3)

    axes[-1].imshow(image)
    axes[-1].set_title("aggregation")
    if aggregated:
        axes[-1].imshow(aggregated[0][1] >= 0.6, alpha=0.3)

    plt.show()


def print_boxes(annotations):
    plt.axes()
    colors = ["red", "blue", "black", "green", "black", "gray", "pink"]
    for i in range(len(annotations)):
        ann = annotations[i]

        if not ann:
            ann = [("", [0, 0, 0, 0])]
            rectangle = plt.Rectangle(
                (ann[0][1][0], ann[0][1][1]),
                ann[0][1][2],
                ann[0][1][3],
                fc="white",
                ec="white",
                alpha=0.6,
            )
            plt.gca().add_patch(rectangle)

        else:
            for j in range(len(ann)):
                rectangle = plt.Rectangle(
                    (ann[j][1][0], ann[j][1][1]),
                    ann[j][1][2],
                    ann[j][1][3],
                    fc="white",
                    ec=colors[i],
                    alpha=0.6,
                )
                plt.gca().add_patch(rectangle)

    plt.axis("scaled")
    plt.grid()
    plt.show()


def print_masks(annotations, image, dimension):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    masks = []
    for i in range(len(annotations)):
        ann = annotations[i]

        if not ann:
            mask = np.zeros((dimension[1], dimension[0]))
        else:
            masks = [MaskMarkup.polygon_to_mask(mark[1], dimension) for mark in ann]
            mask = sum(masks)

        mask[mask % 2 == 1] = 1
        mask[mask % 2 == 0] = 0

        masks.append(mask)

    axes[0].imshow(image)
    axes[1].imshow(image)
    axes[1].imshow(sum(masks), alpha=0.3)


def get_result(
    annotations,
    assignments,
    aggregator,
    markup_type,
    iou_threshold,
    image=None,
    dimension=None,
):
    if markup_type == "bbox":
        print_boxes(annotations)
    elif markup_type == "mask":
        print_masks(annotations, image, dimension)

    label_with_bbox, accepted, rejected = aggregator.soft_markup_aggregation(
        annotations, assignments, iou_threshold, dimension=dimension
    )

    if label_with_bbox is not None:
        print("aggregated markups: ", label_with_bbox)
        print("accepted: ")
        print("\n".join("{}".format(item) for item in accepted))
        print("rejected: ")
        print("\n".join("{}".format(item) for item in rejected))

        return label_with_bbox
