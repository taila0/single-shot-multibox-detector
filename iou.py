import numpy as np
from utils import ccwh2xyxy


def calculate_iou(pr_boxes, gt_boxes):
    """
    sample_bboxes : Ndarray, 2D array [x1, x2, y1, y2, x1, x2, y1, y2, ... ]
    sample_bboxes : Ndarray, 2D array [x1, x2, y1, y2, x1, x2, y1, y2, ... ]
    """

    # cxcywh -> xyxy
    pr_boxes = ccwh2xyxy(pr_boxes)
    # cxcywh -> xyxy
    gt_boxes = ccwh2xyxy(gt_boxes)

    # Get Area
    area_pr = (pr_boxes[:, 2] - pr_boxes[:, 0]) * (pr_boxes[:, 3] - pr_boxes[:, 1])
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # expand dims for using broadcasting
    # (N_achr, 4) -> (N_achr, 1, 4)
    expand_pr = np.expand_dims(pr_boxes, axis=1)

    # (N_gt, 4) -> (1, N_gt, 4)
    expand_gt = np.expand_dims(gt_boxes, axis=0)

    # search Maximum
    x1y1 = np.where(expand_pr[:, :, :2] > expand_gt[:, :, :2], expand_pr[:, :, :2], expand_gt[:, :, :2])

    # search Minimum
    x2y2 = np.where(expand_pr[:, :, 2:] < expand_gt[:, :, 2:], expand_pr[:, :, 2:], expand_gt[:, :, 2:])

    # get overlay area
    overlay_area = np.where(np.prod(np.maximum(x2y2 - x1y1, 0), axis=-1) > 0,
                            np.prod(np.maximum(x2y2 - x1y1, 0), axis=-1), 0)

    # expand dimension for broadcasting
    expand_area_pr = np.expand_dims(area_pr, axis=-1)

    iou = overlay_area / (expand_area_pr + area_gt - overlay_area)
    return iou

