from typing import Any, Callable

import matplotlib.pyplot as plt

from aggme.utils.dataclass import BboxMarkup, IntervalMarkup, MarkupGroup, MaskMarkup


def draw_bbox_group(
    group,
    result=None,
    accepted=None,
    rejected=None,
):
    if group is None:
        print(None)
        return
    if accepted is None:
        accepted = []
    if rejected is None:
        rejected = []
    try:
        nusers = len(group.get_annotators())
        if result is None:
            fig, axes = plt.subplots(1, nusers, figsize=(24, 5))
        else:
            fig, axes = plt.subplots(1, nusers + 1, figsize=(24, 5))
            axes[nusers].set_title(result.data[0].annotator)
            axes[nusers].set_facecolor("bisque")
        i = 0
        for group_by_users in group.get_groups_by_annotators():
            for markup in group_by_users:
                x, y, w, h = markup.expand_markup()
                if markup.label == "gesture":
                    rectangle = plt.Rectangle(
                        (x, y), w, h, fc="white", ec="red", alpha=0.6, lw=2
                    )
                else:
                    rectangle = plt.Rectangle(
                        (x, y), w, h, fc="white", ec="blue", alpha=0.6, lw=2
                    )
                try:
                    axes[i].add_patch(rectangle)
                except TypeError:
                    axes.add_patch(rectangle)

                if markup.annotator in accepted:
                    color = "green"
                elif markup.annotator in rejected:
                    color = "red"
                else:
                    color = "black"
                try:
                    axes[i].set_title(markup.annotator[-7:], c=color)
                except TypeError:
                    axes.set_title(markup.annotator[-7:], c=color)
            i += 1

        # draw result
        if result is not None:
            for i, markup in enumerate(result.data):
                x, y, w, h = markup.expand_markup()
                if markup.label == "gesture":
                    rectangle = plt.Rectangle(
                        (x, y), w, h, fc="white", ec="red", alpha=0.6, lw=2
                    )
                else:
                    rectangle = plt.Rectangle(
                        (x, y), w, h, fc="white", ec="blue", alpha=0.6, lw=2
                    )
                axes[nusers].add_patch(rectangle)

        plt.show()
    except Exception as e:
        print(group)
        raise e


def draw_mask_group(
    group,
    result=None,
    accepted=None,
    rejected=None,
):
    if group is None:
        print(None)
        return
    if accepted is None:
        accepted = []
    if rejected is None:
        rejected = []
    try:
        nusers = len(group.get_annotators())
        nlabels = len(group.get_labels())
        if nusers == 1:
            nusers += 1
        if nlabels == 1:
            nlabels += 1

        labels = {}
        for i, markups in enumerate(group.get_groups_by_labels()):
            labels[markups[0].label] = i

        if result is None:
            fig, axes = plt.subplots(nlabels, nusers, figsize=(20, 10))
        else:
            fig, axes = plt.subplots(nlabels, nusers + 1, figsize=(20, 10))
            axes[0][nusers].set_title(result.data[0].annotator)

        for i, markups in enumerate(group.get_groups_by_annotators()):
            if markups[0].annotator in accepted:
                color = "green"
            elif markups[0].annotator in rejected:
                color = "red"
            else:
                color = "black"

            axes[0][i].set_title(markups[0].annotator[-7:], c=color)
            for markup in markups:
                j = labels[markup.label]
                axes[j][i].imshow(markup.mask, aspect="auto")
                axes[j][i].set_xticks([])
                axes[j][i].set_yticks([])
                axes[j][0].set_ylabel(markup.label)

        # draw result
        if result is not None:
            for markup in result.data:
                axes[labels[markup.label]][nusers].imshow(
                    markup.mask, aspect="auto", cmap="jet"
                )
                axes[labels[markup.label]][nusers].set_xticks([])
                axes[labels[markup.label]][nusers].set_yticks([])
        plt.show()
    except Exception as e:
        print(group)
        raise e


def draw_interval_group(
    group,
    result=None,
    accepted=None,
    rejected=None,
):
    if accepted is None:
        accepted = []
    if rejected is None:
        rejected = []

    n_answers = 1 if result else 0
    n_labels = len(group.get_labels())

    fig, ax = plt.subplots(1, n_labels, figsize=(4 * n_labels, 1 + n_answers))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    if n_labels == 1:
        ax = [ax]

    legend = {}
    for iax, label_group in enumerate(group.get_groups_by_labels()):
        label = label_group[0].label
        # draw group itself
        for i, mark in enumerate(label_group):
            if mark.annotator in accepted:
                color = "green"
            elif mark.annotator in rejected:
                color = "red"
            else:
                color = "black"

            ax[iax].plot(
                *[[mark.x, mark.y], [i, i]],
                marker="o",
                label=str(mark.annotator)[-7:],
                c=color,
            )
            legend[i] = str(mark.annotator)[-7:]

        if result:
            corr_mark = None
            for mark in result.data:
                if mark.label == label:
                    corr_mark = mark
                    break
            if corr_mark is not None:
                ax[iax].plot(
                    *[[corr_mark.x, corr_mark.y], [-2, -2]],
                    marker="o",
                    label=mark.annotator,
                    c="blue",
                )
                legend[-2] = mark.annotator

        ax[iax].set_yticks(list(legend.keys()), list(legend.values()))
        ax[iax].set_ybound(-1 - 3 * n_answers, len(legend.keys()))
        ax[iax].set_xbound()
        ax[iax].set_xlabel(label)


def draw_single_markup(markup):
    if isinstance(markup, BboxMarkup):
        x, y, w, h = markup.expand_markup()
        rectangle = plt.Rectangle((x, y), w, h, fc="white", ec="red", alpha=0.6, lw=2)
        fig, ax = plt.subplots()
        ax.add_patch(rectangle)
    elif isinstance(markup, MaskMarkup):
        plt.imshow(markup.mask)
    elif isinstance(markup, IntervalMarkup):
        left = markup.x
        right = markup.y
        fig, ax = plt.subplots(1, 1, figsize=(4, 1))
        coords = [[left, right], [0, 0]]
        ax.plot(*coords, marker="o", label=str(markup.annotator)[-7:])
        ax.get_yaxis().set_ticks([])
        plt.show()
    else:
        raise NotImplementedError


class Visualizer:
    def __init__(self, data: Any):
        self.data = data
        self._draw_func = self._get_default_draw_function()

    def _get_default_draw_function(self) -> Callable:
        if (
            (type(self.data) is BboxMarkup)
            or (type(self.data) is MaskMarkup)
            or (type(self.data) is IntervalMarkup)
        ):
            draw_func = draw_single_markup
        elif isinstance(self.data, MarkupGroup):
            if type(self.data.data[0]) is BboxMarkup:
                draw_func = draw_bbox_group
            elif type(self.data.data[0]) is MaskMarkup:
                draw_func = draw_mask_group
            elif type(self.data.data[0]) is IntervalMarkup:
                draw_func = draw_interval_group
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(
                f"possible types are:"
                f" {type(BboxMarkup)} / {type(MaskMarkup)} /"
                f" {type(MarkupGroup)} / {type(IntervalMarkup)}."
                f" Provided: {type(self.data)}"
            )
        return draw_func

    def set_draw_function(self, f: Callable):
        self._draw_func = f

    def draw(self, data: Any, *args, **kwargs):
        self._draw_func(data, *args, **kwargs)
