from model import simple_detection_netowrk
from dataset import DetectionDataset
from utils import plot_images
from default_boxes import *
from utils import xywh2xyxy, draw_rectangles, images_with_rectangles, xyxy2xywh
import matplotlib.pyplot as plt
from iou import calculate_iou
from delta import calculate_delta, calculate_gt

# Load dataset
trainset = DetectionDataset(data_type='train')
gt_img, gt_info = trainset[0]
gt_coords = gt_info.iloc[:, 1:5].values
gt_coords = xywh2xyxy(gt_coords)
gt_labels = gt_info.iloc[:, -1].values

# Lenerate default boxes
scales = [20, 30, 40]
ratios = [(1, 1),
          (1.5, 0.5),
          (1.2, 0.8),
          (0.8, 1.2),
          (1.4, 1.4)]

# 적용할 default box 의 크기, boxes shape = (n_scale, n_ratio, 2=(h, w))
# ⚠️ 박스에 하나의 scale 만 적용된다고 가정합니다.
boxes_bucket = []
n_boxes = len(ratios)
for scale in scales:
    boxes = generate_default_boxes([scale], ratios)
    boxes_bucket.append(boxes)

# Generate model
inputs, (cls3_5, loc3_7), (cls4_5, loc4_7), (cls5_5, loc5_7) = simple_detection_netowrk(gt_img.shape, n_boxes,
                                                                                        n_classes=11)
pass