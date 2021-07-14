from model import simple_detection_netowrk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from loss import detection_loss, ssd_loss

# load dataset
train_xs = np.load('../datasets/debug_true_images.npy')
train_ys = np.load('../datasets/debug_true_labels.npy')

input_shape = train_xs.shape[1:]
n_classes = 11
train_ys_cls = train_ys[..., 4:]
train_ys_cls = np.where(train_ys_cls == -1, 10, train_ys_cls)
train_ys_cls = to_categorical(train_ys_cls, num_classes=11)
train_ys = np.concatenate([train_ys[..., :4], train_ys_cls], axis=-1)

# Generate detection SSD model
n_boxes = 5
inputs, (cls3_5, loc3_7), (cls4_5, loc4_7), (cls5_5, loc5_7) = simple_detection_netowrk(input_shape, n_boxes, n_classes)
multi_head_cls = [cls3_5, cls4_5, cls5_5]
multi_head_loc = [loc3_7, loc4_7, loc5_7]
n_head = len(multi_head_loc)

# classification, localization head 을 합침
# cls: (N, h, w, n_classes * 5) -> (N, h * w * 5, n_classes)
# cls: (N, h * w  4 * 5) -> (N, h * w * 4, n_classes)

pred_merged_cls = tf.concat(
    [tf.reshape(head_cls, (-1, np.prod(head_cls.get_shape()[1:3]) * n_boxes, n_classes)) for head_cls in
     multi_head_cls], axis=1)

pred_merged_loc = tf.concat(
    [tf.reshape(head_loc, (-1, np.prod(head_loc.get_shape()[1:3]) * n_boxes, 4)) for head_loc in multi_head_loc],
    axis=1)
pred = tf.concat([pred_merged_loc, pred_merged_cls], axis=-1)
print('Model generated \nModel output shape : {}'.format(pred.get_shape()))

model = Model(inputs, pred)
model.compile('adam', loss=detection_loss)

pred_ = model.predict(train_xs)
model.fit(x=train_xs, y=train_ys)
