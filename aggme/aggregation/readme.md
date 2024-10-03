## Aggregation module
### About algorithms
We present algorithms for aggregating annotations. Framework provides 3 types of markups:


Each class can aggregate with 3 different ways:
- `hard_aggregation` - No heuristics, just checking on IOU.
- `drop_aggregation` - Trying to force-drop bad annotations.
- `soft_aggregation` - Trying to apply heuristics to delete human mistakes. Delete duplicates, find points, change wrong labels, etc.


### Aggregate annotations
To start **aggregating** results with concrete method you need to import class, suitable for your type of data:

```python
from aggme.aggregation import BboxAggregation

# load data
ann_data = ...
group = ann_data.groups[0]  # each group grouped by image/video/audio name

aggregator = BboxAggregation()
result_hard = aggregator.hard_aggregation(group)
result_drop = aggregator.drop_aggregation(group)
result_soft = aggregator.soft_aggregation(group)
```

To aggregate data with all existing methods you can simply use `.get_aggregation_results()` method.

The main idea of this method is to apply next algorithm after previously failed. Priority of our algorithms are simple:
- `hard`
- if first did not worked - `drop`
- if second did not worked - `soft`
- if none worked - markup was not able to aggregate, goes to `fail`. Try to set up hyperparameters of the aggregator.
```python
results = aggregator.get_aggregation_results(group)
```

### Set up hyperparameters
Aggregation classes can be parametrised.
Base setup starts with defining `cluster_method` and its parameters
```python
from aggme.aggregation import BboxAggregation
aggregator = BboxAggregation(
    cluster_method_name='mean_shift',
    cluster_method_params={'bandwith': 0.05}
)
```
For now, available methods are `[mean_shift, dbscan]`

Next parameters are passing to `hard/soft/drop_aggregation()` methods directly.

- `threshold` parameter defines IoU threshold. For Bboxes, it means how much of square of one box covers another
- `confidence` parameter defines `annotator`'s completeness for current markups. For example, if we have 3 different answers
from different annotators, confidence < 0.66 will allow aggregate results from only 2 of 3 annotators (not working with `hard_aggregation`)

See examples in [AggMe usage examples](https://gitlab.ai.cloud.ru/rndcv/aggregator/-/tree/main/examples) to get more information about functionality.
