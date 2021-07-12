from tensorflow.keras.models import load_model
from delta import calculate_gt
from loss import detection_loss
import numpy as np
import pickle
from nms import non_maximum_suppression
from utils import images_with_rectangles, plot_images, xywh2xyxy, draw_rectangles

# load models
model = load_model('../models/model.h5', custom_objects={'detection_loss': detection_loss})

# load dataset
train_xs = np.load('../datasets/debug_true_images.npy')
train_ys = np.load('../datasets/debug_true_labels.npy')

# load default_boxes
f = open('../datasets/default_boxes_bucket.pkl', 'rb')
default_boxes_bucket = pickle.load(f)
default_boxes = np.concatenate(default_boxes_bucket, axis=0)

# predictions
pred_ = model.predict(x=train_xs)
pred_onehot = pred_[..., 4:]
pred_loc = pred_[..., :4]

# recorver default boxes
gt_hat = calculate_gt(default_boxes, pred_)

# get positive bool mask, shape (N_img, N_default_boxes)
pred_cls = np.argmax(pred_onehot, axis=-1)
pos_mask_bucket = (pred_cls != 10)

# 이미지 한장당 positive localization, classification 정보를 가져옵니다.
loc_per_img = []
cls_per_img = []
onehot_per_img = []
for mask, loc, cls, onehot in zip(pos_mask_bucket, gt_hat, pred_cls, pred_onehot):
    pos_loc = loc[mask]
    pos_cls = cls[mask]
    pos_mask = onehot[mask]
    loc_per_img.append(pos_loc)
    cls_per_img.append(pos_cls)
    onehot_per_img.append(pos_mask)


# Non Maximum Suppression per im
nms_bboxes = []
for onehot_, loc_, cls_ in zip(onehot_per_img, loc_per_img, cls_per_img):
    final_bboxes, _ = non_maximum_suppression(loc_, onehot_, 0.5)
    final_bboxes = xywh2xyxy(np.array(final_bboxes))
    nms_bboxes.append(final_bboxes)

rected_images = images_with_rectangles(train_xs * 255, nms_bboxes)
plot_images(rected_images)
