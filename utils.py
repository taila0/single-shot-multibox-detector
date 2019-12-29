import numpy as np


def ccwh2xyxy(ccwh_boxes):
    cxs = ccwh_boxes[:, 0]
    cys = ccwh_boxes[:, 1]
    ws = ccwh_boxes[:, 2]
    hs = ccwh_boxes[:, 3]

    x1s = cxs - ws // 2
    x2s = cxs + ws // 2
    y1s = cys - hs // 2
    y2s = cys + hs // 2

    return np.stack([x1s, y1s, x2s, y2s], axis=-1)

