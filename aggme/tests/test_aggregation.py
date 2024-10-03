import pytest

from aggme.aggregation.bounding_boxes import BboxAggregation
from aggme.aggregation.segmentation_masks import MaskAggregation
from aggme.tests.correct_answers import (
    input_bbox_drop,
    input_bbox_hard,
    input_bbox_soft,
    input_mask_drop,
    input_mask_hard,
    input_mask_soft,
    output_bbox_drop,
    output_bbox_hard,
    output_bbox_soft,
    output_mask_drop,
    output_mask_hard,
    output_mask_soft,
)


@pytest.mark.parametrize("inp, out", zip(input_bbox_hard, output_bbox_hard))
def test_bbox_hard_aggregation(inp, out):
    aggregator = BboxAggregation()
    att_id, assigns_id, annotates, params = inp
    assert aggregator.hard_aggregation(annotates, **params) == out


@pytest.mark.parametrize("inp, out", zip(input_bbox_soft, output_bbox_soft))
def test_bbox_soft_aggregation(inp, out):
    aggregator = BboxAggregation()
    att_id, assigns_id, annotates, params = inp
    assert aggregator.soft_aggregation(annotates, assigns_id, **params) == out


@pytest.mark.parametrize("inp, out", zip(input_bbox_drop, output_bbox_drop))
def test_bbox_drop_aggregation(inp, out):
    aggregator = BboxAggregation()
    att_id, assigns_id, annotates, params = inp
    assert aggregator.drop_aggregation(annotates, assigns_id, **params) == out


@pytest.mark.parametrize("inp, out", zip(input_mask_hard, output_mask_hard))
def test_mask_hard_aggregation(inp, out):
    aggregator = MaskAggregation()
    att_id, assigns_id, annotates, cloud_path, dimension, params = inp
    res = aggregator.hard_aggregation(annotates, dimension=dimension, **params)
    assert res == out


@pytest.mark.parametrize("inp, out", zip(input_mask_soft, output_mask_soft))
def test_mask_soft_aggregation(inp, out):
    aggregator = MaskAggregation()
    att_id, assigns_id, annotates, cloud_path, dimension, params = inp
    res = aggregator.soft_aggregation(
        annotates, assigns_id, dimension=dimension, **params
    )
    if res[0] is None:
        if out[0] is None:
            assert True
        else:
            assert False

    elif out[0] is None:
        assert False
    else:
        assert (res[0][0][1] == out[0][0][1]).all()
        assert res[1] == out[1]
        assert res[2] == out[2]


@pytest.mark.parametrize("inp, out", zip(input_mask_drop, output_mask_drop))
def test_mask_drop_aggregation(inp, out):
    aggregator = MaskAggregation()
    att_id, assigns_id, annotates, cloud_path, dimension, params = inp
    res = aggregator.drop_aggregation(
        annotates, assigns_id, dimension=dimension, **params
    )
    if res[0] is None:
        if out[0] is None:
            assert True
        else:
            assert False

    elif out[0] is None:
        assert False
    else:
        assert (res[0][0][1] == out[0][0][1]).all()
        assert res[1] == out[1]
        assert res[2] == out[2]
