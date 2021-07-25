from tensorflow.keras.models import load_model
from delta import calculate_gt
from loss import detection_loss, ssd_loss
import numpy as np
import pickle
from nms import non_maximum_suppression
from utils import images_with_rectangles, plot_images, xywh2xyxy, draw_rectangles

# load models
model = load_model('../models/best_model.h5', custom_objects={'ssd_loss': ssd_loss})

# load dataset
train_xs = np.load('../datasets/debug_true_images.npy')
train_ys = np.load('../datasets/debug_true_labels.npy')

trues_delta = xywh2xyxy(train_ys[..., :4])
trues_cls = train_ys[..., -1]

# load default_boxes
f = open('../datasets/default_boxes_bucket.pkl', 'rb')
default_boxes_bucket = pickle.load(f)
default_boxes = np.concatenate(default_boxes_bucket, axis=0)

# predictions with batch images
preds = model.predict(x=train_xs)
preds_onehot = preds[..., 4:]  # shape=(N_img, N_anchor, n_classes)
preds_delta = preds[..., :4]  # shape=(N_img, N_anchor, 4)

# change relative coords to absolute coords for predictions
gts_hat = calculate_gt(default_boxes, preds_delta)  # shape=(N_img, N_anchor, 4)

# change relative coords to absolute coords for groundruths
gts = calculate_gt(default_boxes, trues_delta)  # shape=(N_img, N_anchor, 4)

# get foreground(not background) bool mask for prediction, shape (N_img, N_default_boxes)
preds_cls = np.argmax(preds_onehot, axis=-1)  # shape (N_img, N_default_boxes)
pos_preds_mask = (preds_cls != 10)  # shape (N_img, N_default_boxes)

# get foreground bool mask for true, shape (N_img, N_default_boxes)
pos_trues_mask = (trues_cls != 10)  # shape (N_img, N_default_boxes)

# 이미지 한장당 positive localization, classification 정보를 가져옵니다.
pos_preds_loc = []
pos_preds_cls = []
pos_preds_onehot = []
for pos_pred_mask, gt_hat, pred_cls, pred_onehot in zip(pos_preds_mask, gts_hat, preds_cls, preds_onehot):
    pos_loc = gt_hat[pos_pred_mask]
    pos_cls = pred_cls[pos_pred_mask]
    pos_mask = pred_onehot[pos_pred_mask]

    pos_preds_loc.append(pos_loc)
    pos_preds_cls.append(pos_cls)
    pos_preds_onehot.append(pos_mask)

# Non Maximum Suppression per image
nms_bboxes = []
for onehot_, loc_, cls_ in zip(pos_preds_onehot, pos_preds_loc, pos_preds_cls):
    final_bboxes, _, _ = non_maximum_suppression(loc_, onehot_, 0.5)
    final_bboxes = xywh2xyxy(np.array(final_bboxes))
    nms_bboxes.append(final_bboxes)

# 이미지 한장당 positive localization, classification 정보를 가져옵니다.
pos_trues_loc = []
pos_trues_cls = []
for pos_pred_mask, gt, true_cls in zip(pos_trues_mask, gts, trues_cls):
    pos_loc = gt[pos_pred_mask]
    pos_cls = true_cls[pos_pred_mask]
    pos_loc = xywh2xyxy(pos_loc)
    pos_trues_loc.append(pos_loc)
    pos_trues_cls.append(pos_cls)

# visualization prediction
rected_images = images_with_rectangles(train_xs * 255, pos_trues_loc, color=(0, 255, 0))
plot_images(rected_images)
rected_images = images_with_rectangles(train_xs * 255, nms_bboxes, color=(255, 255, 0))
plot_images(rected_images)
