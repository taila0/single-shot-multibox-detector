from dataset import DetectionDataset
from delta import calculate_delta
from default_boxes import *
from utils import xywh2xyxy, draw_rectangles, images_with_rectangles, xyxy2xywh
from iou import calculate_iou

# Get sample image and object coordinates
trainset = DetectionDataset(data_type='train')
gt_img, gt_info = trainset[0]
gt_coords = gt_info.iloc[:, 1:5].values
gt_coords = xywh2xyxy(gt_coords)
gt_labels = gt_info.iloc[:, -1].values

fmap = tf.constant(shape=(2, 8, 8, 2), value=1)
h, w = fmap.get_shape()[1:3]
n_layer = 11
paddings = ['SAME'] * n_layer
strides = [1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2]
kernel_sizes = [3] * n_layer
center_xy = original_rectangle_coords((h, w), kernel_sizes, strides, paddings)[:, :2]

# get w, h
scales = [30]
ratios = [(1, 1),
          (1.5, 0.5),
          (1.2, 0.8),
          (0.8, 1.2),
          (1.4, 1.4)]

# 적용할 default box 의 크기
boxes = generate_default_boxes(scales, ratios)

# Get default boxes over feature map
default_boxes = tiling_default_boxes(center_xy, boxes)
default_boxes = xywh2xyxy(default_boxes)

# default boxes
default_boxes = default_boxes.reshape(-1, 4)

# ground truth coordinates(x1, y1, x2, y2), shape = (N_obj, 4)
gt_coords = gt_coords.reshape(-1, 4)

# 각 obj 별 iou 을 구합니다. shape = (N_default_boxes, N_obj)
ious = calculate_iou(xyxy2xywh(default_boxes), xyxy2xywh(gt_coords))

# iou 중 가장 overlay 비율이 큰 class을 선택합니다.
# shape = (N_default_boxes, )
max_overlay_cls = np.argmax(ious, axis=-1)

# 모든 obj 에 대해 iou 가 0.5 이하이면 background class, -1로 지정합니다.
background_mask = np.all(ious < 0.5, axis=-1)
max_overlay_cls[background_mask] = -1

# 기존의 정답 데이터에 [0, 0, 0, 0] 을 추가합니다.
gt_with_bg = np.concatenate([gt_coords, np.array([[0, 0, 0, 0]])], axis=0)

# 각 default boxes에 해당하는 ground truth 의 좌표값을 가져옵니다.
true_reg = gt_with_bg[max_overlay_cls]

# iou 가 0.5 이상 되는 값들의 index 을 가져옵니다.
model_true_bboxes = gt_with_bg[max_overlay_cls]
pos_mask = np.all(true_reg, axis=-1)
pos_ious_max = np.max(ious[np.any(ious > 0.5, axis=-1)], axis=-1)

# delta 값을 계산합니다.
pos_true_bboxes = xyxy2xywh(model_true_bboxes[pos_mask])
pos_default_boxes = xyxy2xywh(default_boxes[pos_mask])
pos_true_delta = calculate_delta(pos_default_boxes, pos_true_bboxes)
true_delta = np.zeros_like(default_boxes)
true_delta[pos_mask] = pos_true_delta

