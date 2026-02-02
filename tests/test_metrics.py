from mondAI.metrics.mae import MAE


def test_metric_reflexivity() -> None:
    metric = MAE()
    y_true = [1, 2, 3]
    assert metric.compute(y_true, y_true) == 0.0
