import cv2
import numpy as np
from utils.utils import postprocess, BBoxTransform, ClipBoxes
from typing import Optional, List
from functools import partial
# from utils.plot import display
from utils.constants import *
import warnings

# import torch
# import torch.nn as nn
# from torch.nn.modules.loss import _Loss
# import torch.nn.functional as F

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = paddle.minimum(paddle.unsqueeze(a[:, 3], axis=1), b[:, 2]) - paddle.maximum(paddle.unsqueeze(a[:, 1], axis=1), b[:, 0])
    ih = paddle.minimum(paddle.unsqueeze(a[:, 2], axis=1), b[:, 3]) - paddle.maximum(paddle.unsqueeze(a[:, 0], axis=1), b[:, 1])
    iw = paddle.clip(iw, min=0)
    ih = paddle.clip(ih, min=0)
    ua = paddle.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = paddle.clip(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua


    del area, iw, ih, ua, intersection


    return IoU


class _Loss(nn.Layer):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class FocalLoss(nn.Layer):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            # print(bbox_annotation)

            classification = paddle.clip(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if paddle.device.is_compiled_with_cuda():

                    alpha_factor = paddle.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * paddle.pow(focal_weight, gamma)

                    bce = -(paddle.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    regression_losses.append(paddle.to_tensor(0,dtype=dtype))
                    classification_losses.append(cls_loss.sum())
                else:

                    alpha_factor = paddle.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * paddle.pow(focal_weight, gamma)

                    bce = -(paddle.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    regression_losses.append(paddle.to_tensor(0, dtype=dtype))
                    classification_losses.append(cls_loss.sum())

                continue

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])
            
            IoU_max = paddle.max(IoU, axis=1)
            IoU_argmax = paddle.argmax(IoU, axis=1)

            # compute the loss for classification
            #targets = torch.ones_like(classification) * -1
            targets = paddle.zeros_like(classification)
            
            # if paddle.device.is_compiled_with_cuda():  # 删
            #     targets = targets.cuda()
            
            assigned_annotations = paddle.index_select(bbox_annotation, IoU_argmax, axis=0)
            
            positive_indices = paddle.full_like(IoU_max,False,dtype=paddle.bool) #torch.ge(IoU_max, 0.2) 
                        
            tensorA = (assigned_annotations[:, 2] - assigned_annotations[:, 0]) * (assigned_annotations[:, 3] - assigned_annotations[:, 1]) > 10 * 10
#             for idx,iou in enumerate(IoU_max):
#                 if tensorA[idx]: # Set iou threshold = 0.5
#                     if iou >= 0.5:
#                         positive_indices[idx] = True
# #                         targets[idx,:] = True
# #                     else:
# #                         positive_indices[idx] = False
#                 else:
#                     if iou >= 0.15:
#                         positive_indices[idx] = True
# #                     else:
# #                         positive_indices[idx] = False              
                                
# #             targets[torch.lt(IoU_max, 0.4), :] = 0

            a = paddle.logical_or(paddle.logical_and(tensorA,IoU_max >= 0.5),paddle.logical_and(~tensorA,IoU_max >= 0.15))
            positive_indices = paddle.to_tensor(positive_indices, dtype=paddle.int32)
            # positive_indices[paddle.logical_or(paddle.logical_and(tensorA,IoU_max >= 0.5),paddle.logical_and(~tensorA,IoU_max >= 0.15))] = 1
            index = paddle.where(paddle.logical_or(paddle.logical_and(tensorA,IoU_max >= 0.5),paddle.logical_and(~tensorA,IoU_max >= 0.15)))
            positive_indices_true = paddle.gather_nd(positive_indices, index)
            positive_indices_true = positive_indices_true + 1
            positive_indices = paddle.scatter_nd_add(positive_indices, index, positive_indices_true)
            positive_indices = paddle.to_tensor(positive_indices, dtype=paddle.bool)
            num_positive_anchors = positive_indices.sum()

#             for box in assigned_annotations[positive_indices, :]:
#                 xmin,ymin,xmax,ymax, cls = box
#                 print("WIDTH HEIGHT:", (xmax-xmin),"\t", (ymax-ymin))
#             for box in bbox_annotation:
#                 xmin,ymin,xmax,ymax, cls = box
#                 print("111 WIDTH HEIGHT:", (xmax-xmin),"\t", (ymax-ymin))
            

#             targets[positive_indices, :] = 0


            # targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1  # 删
            np_targets = targets.numpy()
            np_targets[positive_indices.numpy(), (assigned_annotations.numpy())[positive_indices.numpy(), 4].astype(np.int32)] = 1
            targets = paddle.to_tensor(np_targets, dtype=targets.dtype)

            alpha_factor = paddle.ones_like(targets) * alpha
            # if paddle.device.is_compiled_with_cuda():  # 删
            #     alpha_factor = alpha_factor.cuda()

            alpha_factor = paddle.where(paddle.equal(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = paddle.where(paddle.equal(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * paddle.pow(focal_weight, gamma)

            bce = -(targets * paddle.log(classification) + (1.0 - targets) * paddle.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = paddle.zeros_like(cls_loss)
            # if torch.cuda.is_available():  # 删除
            #     zeros = zeros.cuda()
            cls_loss = paddle.where(paddle.not_equal(targets, paddle.full_like(targets, -1.0, dtype=targets.dtype)), cls_loss, zeros)
 
            classification_losses.append(cls_loss.sum() / paddle.clip(paddle.to_tensor(num_positive_anchors, dtype=dtype), min=1.0))


            if positive_indices.sum() > 0:
                np_assigned_annotations = assigned_annotations.numpy()
                np_assigned_annotations = np_assigned_annotations[positive_indices.numpy(), :]
                assigned_annotations = paddle.to_tensor(np_assigned_annotations, dtype=assigned_annotations.dtype)

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = paddle.clip(gt_widths, min=1)
                gt_heights = paddle.clip(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = paddle.log(gt_widths / anchor_widths_pi)
                targets_dh = paddle.log(gt_heights / anchor_heights_pi)

                targets = paddle.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()  # paddle torch 一样 转置


                index = paddle.where(positive_indices)
                regression_true = paddle.gather_nd(regression, index)
                
                regression_diff = paddle.abs(targets - regression_true)

                regression_loss = paddle.where(
                    paddle.less_equal(regression_diff, paddle.full_like(regression_diff, 1.0 / 9.0, dtype=regression_diff.dtype)),
                    0.5 * 9.0 * paddle.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                # if paddle.device.is_compiled_with_cuda():  # 删
                #     regression_losses.append(paddle.tensor(0).to(dtype).cuda()) 
                # else:
                #     regression_losses.append(paddle.tensor(0).to(dtype))
                regression_losses.append(paddle.to_tensor(0, dtype=dtype))
            
        # debug
        imgs = kwargs.get('imgs', None)  # 看到这里
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(),
                              paddle.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
                              regressBoxes, clipBoxes,
                              0.25, 0.3)
            imgs = paddle.transpose(imgs, perm=[0, 2, 3, 1]).numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            # display(out, imgs, obj_list, imshow=False, imwrite=True)

        # del IoU  # 偶尔报错 UnboundLocalError: local variable 'IoU' referenced before assignment ？？
        # del IoU_argmax, IoU_max, a, alpha, alpha_factor, anchor, anchor_ctr_x, anchor_ctr_x_pi, anchor_ctr_y, anchor_ctr_y_pi
        # del anchor_heights, anchor_heights_pi, anchor_widths, anchor_widths_pi, anchors, annotations, assigned_annotations, batch_size
        # del bbox_annotation, dtype, gamma, gt_ctr_x, gt_ctr_y, gt_heights, gt_widths, index, np_assigned_annotations, np_targets
        # del num_positive_anchors, positive_indices, positive_indices_true, targets, targets_dh, targets_dw, targets_dx, targets_dy
        # del tensorA, zeros
        # paddle.device.cuda.empty_cache()

        return paddle.stack(classification_losses).mean(axis=0, keepdim=True), \
               paddle.stack(regression_losses).mean(axis=0, keepdim=True) * 50  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233


def focal_loss_with_logits(
    output: paddle.Tensor,
    target: paddle.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> paddle.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = paddle.to_tensor(target, dtype=output.dtype)

    # https://github.com/qubvel/segmentation_models.pytorch/issues/612
    # logpt = F.binary_cross_entropy(output, target, reduction="none")
    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = paddle.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        # focal_term[pt < reduced_threshold] = 1
        index = paddle.where(pt < reduced_threshold)
        focal_term_true = paddle.gather_nd(focal_term, index)
        focal_term_true = focal_term_true + 1
        focal_term = paddle.scatter_nd_add(focal_term, index, focal_term_true)

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = paddle.clip(focal_term.sum(), eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


class FocalLossSeg(_Loss):
    def __init__(
        self,
        mode: str,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
                should use `reduction="sum"`.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: paddle.Tensor, y_true: paddle.Tensor) -> paddle.Tensor:

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.reshape([-1])
            y_pred = y_pred.reshape([-1])

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:
            # print(y_true.shape)
            # print(y_pred.shape)
            num_classes = y_pred.shape[1]
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = paddle.to_tensor((y_true == cls), dtype=paddle.int32)
                cls_y_pred = y_pred[:, cls, ...]
                # print(cls_y_true.shape)
                # print(cls_y_pred.shape)
                # exit()

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss

def to_tensor(x, dtype=None) -> paddle.Tensor:
    if isinstance(x, paddle.Tensor):
        if dtype is not None:
            x = paddle.to_tensor(x, dtype=dtype)
        return x
    if isinstance(x, np.ndarray):
        x = paddle.to_tensor(x)
        if dtype is not None:
            x = paddle.to_tensor(x, dtype=dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = paddle.to_tensor(x)
        if dtype is not None:
            x = paddle.to_tensor(x, dtype=dtype)
        return x


def soft_dice_score(
    output: paddle.Tensor,
    target: paddle.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> paddle.Tensor:
    assert output.shape == target.shape
    if dims is not None:
        intersection = paddle.sum(output * target, axis=dims)
        cardinality = paddle.sum(output + target, axis=dims)
    else:
        intersection = paddle.sum(output * target)
        cardinality = paddle.sum(output + target)
    dice_score = paddle.clip((2.0 * intersection + smooth) / (cardinality + smooth), min=eps)
    return dice_score


class DiceLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=paddle.int32)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: paddle.Tensor, y_true: paddle.Tensor) -> paddle.Tensor:

        assert y_true.shape[0] == y_pred.shape[0]

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = (F.log_softmax(y_pred, axis=1)).exp()
            else:
                y_pred = (F.log_sigmoid(y_pred)).exp()

        bs = y_true.shape[0]
        num_classes = y_pred.shape[1]
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = paddle.reshape(y_true, shape=[bs, 1, -1])
            y_pred = paddle.reshape(y_pred, shape=[bs, 1, -1])
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = paddle.reshape(y_true, shape=[bs, -1])
            y_pred = paddle.reshape(y_pred, shape=[bs, num_classes, -1])

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
            
                y_true = F.one_hot(paddle.to_tensor((y_true * mask), dtype=paddle.int32), num_classes)  # N,H*W -> N,H*W, C
                y_true = paddle.transpose(y_true, perm=[0, 2, 1]) * mask.unsqueeze(1)  # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = paddle.transpose(y_true, perm=[0, 2, 1])  # N, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = paddle.reshape(y_true, shape=[bs, num_classes, -1])
            y_pred = paddle.reshape(y_pred, shape=[bs, num_classes, -1])
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(y_pred, paddle.to_tensor(y_true, dtype=y_pred.dtype), smooth=self.smooth, eps=self.eps, dims=dims)
        
        if self.log_loss:
            loss = -paddle.log(paddle.clip(scores, min=self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(axis=dims) > 0
        loss *= paddle.to_tensor(mask, dtype=loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> paddle.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)

def soft_tversky_score(
    output: paddle.Tensor,
    target: paddle.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> paddle.Tensor:
    assert output.shape == target.shape
    if dims is not None:
        intersection = paddle.sum(output * target, axis=dims)  # TP
        fp = paddle.sum(output * (1.0 - target), axis=dims)
        fn = paddle.sum((1 - output) * target, axis=dims)
    else:
        intersection = paddle.sum(output * target)  # TP
        fp = paddle.sum(output * (1.0 - target))
        fn = paddle.sum((1 - output) * target)

    tversky_score = paddle.clip((intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth), min=eps)
    return tversky_score

class TverskyLoss(DiceLoss):
    """Tversky loss for image segmentation task.
    Where TP and FP is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases

    Args:
        mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
        from_logits: If True assumes input is raw logits
        smooth:
        ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Positives)
        gamma: Constant that squares the error function. Defaults to ``1.0``

    Return:
        loss: torch.Tensor

    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0
    ):

        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(mode, classes, log_loss, from_logits, smooth, ignore_index, eps)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> paddle.Tensor:
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)
    

def legacy_get_string(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool = True) -> str:
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret