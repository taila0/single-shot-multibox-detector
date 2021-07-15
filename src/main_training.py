from model import simple_detection_netowrk
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from loss import detection_loss, ssd_loss

# load dataset
train_xs = np.load('../datasets/debug_true_images.npy')
train_ys = np.load('../datasets/debug_true_images.npy')

input_shape = train_xs.shape[1:]
n_classes = 11
train_ys_cls = train_ys[..., 4:]
train_ys_cls = np.where(train_ys_cls == -1, 10, train_ys_cls)
train_ys_cls = to_categorical(train_ys_cls, num_classes=11)
train_ys = np.concatenate([train_ys[..., :4], train_ys_cls], axis=-1)

# Generate detection SSD model
n_boxes = 5
inputs, predictions = simple_detection_netowrk(input_shape, n_boxes, n_classes)

model = Model(inputs, predictions)
model.compile('adam', loss=ssd_loss)

pred_ = model.predict(train_xs)
model.fit(x=train_xs, y=train_ys)
