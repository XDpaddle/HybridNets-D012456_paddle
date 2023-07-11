"""Various metrics based on Type I and Type II errors.

References:
    https://en.wikipedia.org/wiki/Confusion_matrix


Example:

    .. code-block:: python

        import segmentation_models_pytorch as smp

        # lets assume we have multilabel prediction for 3 classes
        output = torch.rand([10, 3, 256, 256])
        target = torch.rand([10, 3, 256, 256]).round().long()

        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multilabel', threshold=0.5)

        # then compute metrics with required reduction (see metric docs)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

"""
# import torch
import warnings
from typing import Optional, List, Tuple, Union

import paddle

__all__ = [
    "get_stats",
    "fbeta_score",
    "f1_score",
    "iou_score",
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "positive_predictive_value",
    "negative_predictive_value",
    "false_negative_rate",
    "false_positive_rate",
    "false_discovery_rate",
    "false_omission_rate",
    "positive_likelihood_ratio",
    "negative_likelihood_ratio",
]


###################################################################################################
# Statistics computation (true positives, false positives, false negatives, false positives)
###################################################################################################

def get_stats(output, target, mode, ignore_index = None, threshold = None, num_classes = None):
    """Compute true positive, false positive, false negative, true negative 'pixels'
    for each image and each class.

    Args:
        output (Union[torch.LongTensor, torch.FloatTensor]): Model output with following
            shapes and types depending on the specified ``mode``:

            'binary'
                shape (N, 1, ...) and ``torch.LongTensor`` or ``torch.FloatTensor``

            'multilabel'
                shape (N, C, ...) and ``torch.LongTensor`` or ``torch.FloatTensor``

            'multiclass'
                shape (N, ...) and ``torch.LongTensor``

        target (torch.LongTensor): Targets with following shapes depending on the specified ``mode``:

            'binary'
                shape (N, 1, ...)

            'multilabel'
                shape (N, C, ...)

            'multiclass'
                shape (N, ...)

        mode (str): One of ``'binary'`` | ``'multilabel'`` | ``'multiclass'``
        ignore_index (Optional[int]): Label to ignore on for metric computation.
            **Not** supproted for ``'binary'`` and ``'multilabel'`` modes.  Defaults to None.
        threshold (Optional[float, List[float]]): Binarization threshold for
            ``output`` in case of ``'binary'`` or ``'multilabel'`` modes. Defaults to None.
        num_classes (Optional[int]): Number of classes, necessary attribute
            only for ``'multiclass'`` mode. Class values should be in range 0..(num_classes - 1).
            If ``ignore_index`` is specified it should be outside the classes range, e.g. ``-1`` or
            ``255``.

    Raises:
        ValueError: in case of misconfiguration.

    Returns:
        Tuple[torch.LongTensor]: true_positive, false_positive, false_negative,
            true_negative tensors (N, C) shape each.

    """

    if paddle.is_floating_point(target):
        raise ValueError(f"Target should be one of the integer types, got {target.dtype}.")

    if paddle.is_floating_point(output) and threshold is None:
        raise ValueError(
            f"Output should be one of the integer types if ``threshold`` is not None, got {output.dtype}."
        )

    if paddle.is_floating_point(output) and mode == "multiclass":
        raise ValueError(f"For ``multiclass`` mode ``target`` should be one of the integer types, got {output.dtype}.")

    if mode not in {"binary", "multiclass", "multilabel"}:
        raise ValueError(f"``mode`` should be in ['binary', 'multiclass', 'multilabel'], got mode={mode}.")

    if mode == "multiclass" and threshold is not None:
        raise ValueError("``threshold`` parameter does not supported for this 'multiclass' mode")

    if output.shape != target.shape:
        raise ValueError(
            "Dimensions should match, but ``output`` shape is not equal to ``target`` "
            + f"shape, {output.shape} != {target.shape}"
        )

    if mode != "multiclass" and ignore_index is not None:
        raise ValueError(f"``ignore_index`` parameter is not supproted for '{mode}' mode")

    if mode == "multiclass" and num_classes is None:
        raise ValueError("``num_classes`` attribute should be not ``None`` for 'multiclass' mode.")

    if ignore_index is not None and 0 <= ignore_index <= num_classes - 1:
        raise ValueError(
            f"``ignore_index`` should be outside the class values range, but got class values in range "
            f"0..{num_classes - 1} and ``ignore_index={ignore_index}``. Hint: if you have ``ignore_index = 0``"
            f"consirder subtracting ``1`` from your target and model output to make ``ignore_index = -1``"
            f"and relevant class values started from ``0``."
        )

    if mode == "multiclass":
        tp, fp, fn, tn = _get_stats_multiclass(output, target, num_classes, ignore_index)
    else:
        if threshold is not None:
            output = paddle.where(output >= threshold, 1, 0)
            target = paddle.where(target >= threshold, 1, 0)
        tp, fp, fn, tn = _get_stats_multilabel(output, target)

    return tp, fp, fn, tn


@paddle.no_grad()
def _get_stats_multiclass(output, target, num_classes, ignore_index):

    batch_size, *dims = output.shape
    num_elements = paddle.cast(paddle.prod(paddle.to_tensor(dims)), dtype=paddle.int64)

    if ignore_index is not None:
        ignore = target == ignore_index
        output = paddle.where(ignore, paddle.full_like(output, -1), output)
        target = paddle.where(ignore, paddle.full_like(target, -1), target)
        ignore_per_sample = ignore.reshape([batch_size, -1]).sum(axis=1)

    tp_count = paddle.cast(paddle.zeros([batch_size, num_classes]), dtype=paddle.int64)
    fp_count = paddle.cast(paddle.zeros([batch_size, num_classes]), dtype=paddle.int64)
    fn_count = paddle.cast(paddle.zeros([batch_size, num_classes]), dtype=paddle.int64)
    tn_count = paddle.cast(paddle.zeros([batch_size, num_classes]), dtype=paddle.int64)

    for i in range(batch_size):
        target_i = target[i]
        output_i = output[i]
        mask = output_i == target_i
        matched = paddle.where(mask, target_i, paddle.full_like(target_i, -1))
        tp = paddle.histogram(paddle.cast(matched, dtype=paddle.float32), bins=num_classes, min=0, max=num_classes - 1)
        fp = paddle.histogram(paddle.cast(output_i, dtype=paddle.float32), bins=num_classes, min=0, max=num_classes - 1) - tp
        fn = paddle.histogram(paddle.cast(target_i, dtype=paddle.float32), bins=num_classes, min=0, max=num_classes - 1) - tp
        tn = num_elements - tp - fp - fn
        if ignore_index is not None:
            tn = tn - ignore_per_sample[i]
        tp_count[i] = paddle.cast(tp, dtype=paddle.int64)
        fp_count[i] = paddle.cast(fp, dtype=paddle.int64)
        fn_count[i] = paddle.cast(fn, dtype=paddle.int64)
        tn_count[i] = paddle.cast(tn, dtype=paddle.int64)

    return tp_count, fp_count, fn_count, tn_count


@paddle.no_grad()
def _get_stats_multilabel(output, target):

    batch_size, num_classes, *dims = target.shape
    output = output.reshape([batch_size, num_classes, -1])
    target = target.reshape([batch_size, num_classes, -1])

    tp = (output * target).sum(2)
    fp = output.sum(2) - tp
    fn = target.sum(2) - tp
    tn = paddle.prod(paddle.to_tensor(dims)) - (tp + fp + fn)

    return tp, fp, fn, tn


###################################################################################################
# Metrics computation
###################################################################################################


def _handle_zero_division(x, zero_division):
    nans = paddle.isnan(x)
    if paddle.any(nans) and zero_division == "warn":
        warnings.warn("Zero division in metric calculation!")
    value = zero_division if zero_division != "warn" else 0
    value = paddle.to_tensor(value, dtype=x.dtype)
    # x = paddle.where(nans, value, x) paddle.full_like(x, fill_value=value)
    x = paddle.where(nans, paddle.full_like(x, fill_value=value), x)
    return x


def _compute_metric(
    metric_fn,
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division="warn",
    **metric_kwargs,
) -> float:

    if class_weights is None and reduction is not None and "weighted" in reduction:
        raise ValueError(f"Class weights should be provided for `{reduction}` reduction")

    class_weights = class_weights if class_weights is not None else 1.0
    class_weights = paddle.to_tensor(class_weights)
    class_weights = class_weights / class_weights.sum()

    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)

    elif reduction == "macro" or reduction == "weighted":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score * class_weights).mean()

    elif reduction == "micro-imagewise":
        tp = tp.sum(1)
        fp = fp.sum(1)
        fn = fn.sum(1)
        tn = tn.sum(1)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = score.mean()

    elif reduction == "macro-imagewise" or reduction == "weighted-imagewise":
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score.mean(0) * class_weights).mean()

    elif reduction == "none" or reduction is None:
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)

    else:
        raise ValueError(
            "`reduction` should be in [micro, macro, weighted, micro-imagewise,"
            + "macro-imagesize, weighted-imagewise, none, None]"
        )

    return score


# Logic for metric computation, all metrics are with the same interface


def _fbeta_score(tp, fp, fn, tn, beta=1):
    beta_tp = (1 + beta ** 2) * tp
    beta_fn = (beta ** 2) * fn
    score = beta_tp / (beta_tp + beta_fn + fp)
    return score


def _iou_score(tp, fp, fn, tn):
    return tp / (tp + fp + fn)


def _accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)


def _sensitivity(tp, fp, fn, tn):
    return tp / (tp + fn)


def _specificity(tp, fp, fn, tn):
    return tn / (tn + fp)


def _balanced_accuracy(tp, fp, fn, tn):
    return (_sensitivity(tp, fp, fn, tn) + _specificity(tp, fp, fn, tn)) / 2


def _positive_predictive_value(tp, fp, fn, tn):
    return tp / (tp + fp)


def _negative_predictive_value(tp, fp, fn, tn):
    return tn / (tn + fn)


def _false_negative_rate(tp, fp, fn, tn):
    return fn / (fn + tp)


def _false_positive_rate(tp, fp, fn, tn):
    return fp / (fp + tn)


def _false_discovery_rate(tp, fp, fn, tn):
    return 1 - _positive_predictive_value(tp, fp, fn, tn)


def _false_omission_rate(tp, fp, fn, tn):
    return 1 - _negative_predictive_value(tp, fp, fn, tn)


def _positive_likelihood_ratio(tp, fp, fn, tn):
    return _sensitivity(tp, fp, fn, tn) / _false_positive_rate(tp, fp, fn, tn)


def _negative_likelihood_ratio(tp, fp, fn, tn):
    return _false_negative_rate(tp, fp, fn, tn) / _specificity(tp, fp, fn, tn)


def fbeta_score(
    tp,
    fp,
    fn,
    tn,
    beta: float = 1.0,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """F beta score"""
    return _compute_metric(
        _fbeta_score,
        tp,
        fp,
        fn,
        tn,
        beta=beta,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def f1_score(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """F1 score"""
    return _compute_metric(
        _fbeta_score,
        tp,
        fp,
        fn,
        tn,
        beta=1.0,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def iou_score(tp, fp, fn, tn, reduction=None, class_weights=None, zero_division=1.0):
    """IoU score or Jaccard index"""  # noqa
    return _compute_metric(
        _iou_score,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def accuracy(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """Accuracy"""
    return _compute_metric(
        _accuracy,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def sensitivity(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """Sensitivity, recall, hit rate, or true positive rate (TPR)"""
    return _compute_metric(
        _sensitivity,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def specificity(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """Specificity, selectivity or true negative rate (TNR)"""
    return _compute_metric(
        _specificity,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def balanced_accuracy(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """Balanced accuracy"""
    return _compute_metric(
        _balanced_accuracy,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def positive_predictive_value(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """Precision or positive predictive value (PPV)"""
    return _compute_metric(
        _positive_predictive_value,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def negative_predictive_value(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """Negative predictive value (NPV)"""
    return _compute_metric(
        _negative_predictive_value,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_negative_rate(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """Miss rate or false negative rate (FNR)"""
    return _compute_metric(
        _false_negative_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_positive_rate(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """Fall-out or false positive rate (FPR)"""
    return _compute_metric(
        _false_positive_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_discovery_rate(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """False discovery rate (FDR)"""  # noqa
    return _compute_metric(
        _false_discovery_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_omission_rate(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """False omission rate (FOR)"""  # noqa
    return _compute_metric(
        _false_omission_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def positive_likelihood_ratio(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """Positive likelihood ratio (LR+)"""
    return _compute_metric(
        _positive_likelihood_ratio,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def negative_likelihood_ratio(
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
):
    """Negative likelihood ratio (LR-)"""
    return _compute_metric(
        _negative_likelihood_ratio,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


_doc = """

    Args:
        tp (torch.LongTensor): tensor of shape (N, C), true positive cases
        fp (torch.LongTensor): tensor of shape (N, C), false positive cases
        fn (torch.LongTensor): tensor of shape (N, C), false negative cases
        tn (torch.LongTensor): tensor of shape (N, C), true negative cases
        reduction (Optional[str]): Define how to aggregate metric between classes and images:

            - 'micro'
                Sum true positive, false positive, false negative and true negative pixels over
                all images and all classes and then compute score.

            - 'macro'
                Sum true positive, false positive, false negative and true negative pixels over
                all images for each label, then compute score for each label separately and average labels scores.
                This does not take label imbalance into account.

            - 'weighted'
                Sum true positive, false positive, false negative and true negative pixels over
                all images for each label, then compute score for each label separately and average
                weighted labels scores.

            - 'micro-imagewise'
                Sum true positive, false positive, false negative and true negative pixels for **each image**,
                then compute score for **each image** and average scores over dataset. All images contribute equally
                to final score, however takes into accout class imbalance for each image.

            - 'macro-imagewise'
                Compute score for each image and for each class on that image separately, then compute average score
                on each image over labels and average image scores over dataset. Does not take into account label
                imbalance on each image.

            - 'weighted-imagewise'
                Compute score for each image and for each class on that image separately, then compute weighted average
                score on each image over labels and average image scores over dataset.

            - 'none' or ``None``
                Same as ``'macro-imagewise'``, but without any reduction.

            For ``'binary'`` case ``'micro' = 'macro' = 'weighted'`` and
            ``'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'``.

            Prefixes ``'micro'``, ``'macro'`` and ``'weighted'`` define how the scores for classes will be aggregated,
            while postfix ``'imagewise'`` defines how scores between the images will be aggregated.

        class_weights (Optional[List[float]]): list of class weights for metric
            aggregation, in case of `weighted*` reduction is chosen. Defaults to None.
        zero_division (Union[str, float]): Sets the value to return when there is a zero division,
            i.e. when all predictions and labels are negative. If set to “warn”, this acts as 0,
            but warnings are also raised. Defaults to 1.

    Returns:
        torch.Tensor: if ``'reduction'`` is not ``None`` or ``'none'`` returns scalar metric,
            else returns tensor of shape (N, C)

    References:
        https://en.wikipedia.org/wiki/Confusion_matrix
"""

fbeta_score.__doc__ += _doc
f1_score.__doc__ += _doc
iou_score.__doc__ += _doc
accuracy.__doc__ += _doc
sensitivity.__doc__ += _doc
specificity.__doc__ += _doc
balanced_accuracy.__doc__ += _doc
positive_predictive_value.__doc__ += _doc
negative_predictive_value.__doc__ += _doc
false_negative_rate.__doc__ += _doc
false_positive_rate.__doc__ += _doc
false_discovery_rate.__doc__ += _doc
false_omission_rate.__doc__ += _doc
positive_likelihood_ratio.__doc__ += _doc
negative_likelihood_ratio.__doc__ += _doc

precision = positive_predictive_value
recall = sensitivity
