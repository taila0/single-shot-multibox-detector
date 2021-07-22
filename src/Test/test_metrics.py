from unittest import TestCase
from tensorflow.keras.models import load_model
import numpy as np
from delta import calculate_gt
from utils import xywh2xyxy
import pickle


class TestMetrics(TestCase):
    def setUp(self):
        # load models
        self.model = load_model('../models/best_model.h5', custom_objects={'ssd_loss': ssd_loss})

        # load dataset
        self.train_xs = np.load('../datasets/debug_true_images.npy')
        self.train_ys = np.load('../datasets/debug_true_labels.npy')

        # split dataset to delta, classification
        self.trues_delta = xywh2xyxy(self.train_ys[..., :4])
        self.trues_cls = self.train_ys[..., -1]

        # predictions with batch images
        preds = self.model.predict(x=self.train_xs)
        self.preds_onehot = preds[..., 4:]  # shape=(N_img, N_anchor, n_classes)
        self.preds_delta = preds[..., :4]  # shape=(N_img, N_anchor, 4)

        # load default_boxes
        f = open('../datasets/default_boxes_bucket.pkl', 'rb')
        default_boxes_bucket = pickle.load(f)
        self.default_boxes = np.concatenate(default_boxes_bucket, axis=0)

        # change relative coords to absolute coords for groundruths
        self.gts_hat = calculate_gt(self.default_boxes, self.preds_delta)  # shape=(N_img, N_anchor, 4)
        self.gts = calculate_gt(self.default_boxes, self.trues_delta)  # shape=(N_img, N_anchor, 4)

        # get foreground(not background) bool mask for prediction, shape (N_img, N_default_boxes)
        self.preds_cls = np.argmax(self.preds_onehot, axis=-1)  # shape (N_img, N_default_boxes)
        self.pos_preds_mask = (self.preds_cls != 10)  # shape (N_img, N_default_boxes)

        # get foreground bool mask for true, shape (N_img, N_default_boxes)
        self.pos_trues_mask = (self.trues_cls != 10)  # shape (N_img, N_default_boxes)




