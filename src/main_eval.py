from tensorflow.keras.models import load_model
from delta import calculate_gt
from loss import detection_loss
import numpy as np
import pickle

# load models
model = load_model('../models/model.h5', custom_objects={'detection_loss':detection_loss})

# load dataset
train_xs = np.load('../datasets/debug_true_images.npy')
train_ys = np.load('../datasets/debug_true_labels.npy')

# load default_boxes
f = open('../datasets/default_boxes_bucket.pkl', 'rb')
default_boxes_bucket = pickle.load(f)
default_boxes = np.concatenate(default_boxes_bucket, axis=0)

# predictions
pred_ = model.predict(x=train_xs)
pred_cls = pred_[..., 4:]
pred_loc = pred_[..., :4]

# recorver default boxes
gt_hat = calculate_gt(default_boxes, pred_)

# get positive bool mask
pred_cls = np.argmax(pred_cls, axis=-1)
pos_mask = (pred_cls != 10)

# get positive coordinate
pos_loc = pred_loc[pos_mask]
pass