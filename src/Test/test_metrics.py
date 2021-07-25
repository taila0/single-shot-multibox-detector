from unittest import TestCase
from tensorflow.keras.models import load_model
import numpy as np
from delta import calculate_gt
from loss import ssd_loss
from metrics import matching
from nms import non_maximum_suppression
from utils import xywh2xyxy
import pickle


class TestMetrics(TestCase):
    def setUp(self):
        # load models
        self.model = load_model('../../models/best_model.h5', custom_objects={'ssd_loss': ssd_loss})

        # load dataset
        self.train_xs = np.load('../../datasets/debug_true_images.npy')
        self.train_ys = np.load('../../datasets/debug_true_labels.npy')
        f = open('../../datasets/debug_gt_labels_bucket.pkl', 'rb')
        self.gt_labels_bucket = pickle.load(f)
        f = open('../../datasets/debug_gt_coords_bucket.pkl', 'rb')
        self.gt_coords_bucket = pickle.load(f)

        # split dataset to delta, classification
        self.trues_delta = xywh2xyxy(self.train_ys[..., :4])
        self.trues_cls = self.train_ys[..., -1]

        # predictions with batch images
        self.preds = self.model.predict(x=self.train_xs)
        self.preds_onehot = self.preds[..., 4:]  # shape=(N_img, N_anchor, n_classes)
        self.preds_delta = self.preds[..., :4]  # shape=(N_img, N_anchor, 4)

        # load default_boxes
        f = open('../../datasets/default_boxes_bucket.pkl', 'rb')
        default_boxes_bucket = pickle.load(f)
        self.default_boxes = np.concatenate(default_boxes_bucket, axis=0)

        # change relative coords to absolute coords for predictions
        self.gts_hat = calculate_gt(self.default_boxes, self.preds_delta)  # shape=(N_img, N_anchor, 4)

        # change relative coords to absolute coords for groundruths
        self.gts = calculate_gt(self.default_boxes, self.trues_delta)  # shape=(N_img, N_anchor, 4)

        # get foreground(not background) bool mask for prediction, shape (N_img, N_default_boxes)
        self.preds_cls = np.argmax(self.preds_onehot, axis=-1)  # shape (N_img, N_default_boxes)
        self.pos_preds_mask = (self.preds_cls != 10)  # shape (N_img, N_default_boxes)

        # get foreground bool mask for true, shape (N_img, N_default_boxes)
        self.pos_trues_mask = (self.trues_cls != 10)  # shape (N_img, N_default_boxes)

        # 이미지 한장당 positive localization, classification 정보를 가져옵니다.
        pos_preds_loc = []
        pos_preds_cls = []
        pos_preds_onehot = []
        for pos_pred_mask, gt_hat, pred_cls, pred_onehot in zip(self.pos_preds_mask,
                                                                self.gts_hat,
                                                                self.preds_cls,
                                                                self.preds_onehot):
            pos_loc = gt_hat[pos_pred_mask]
            pos_cls = pred_cls[pos_pred_mask]
            pos_mask = pred_onehot[pos_pred_mask]

            pos_preds_loc.append(pos_loc)
            pos_preds_cls.append(pos_cls)
            pos_preds_onehot.append(pos_mask)

        # Non Maximum Suppression per image
        self.nms_bboxes = []
        self.nms_labels = []

        for onehot_, loc_, cls_ in zip(pos_preds_onehot, pos_preds_loc, pos_preds_cls):
            final_bboxes, _, final_labels = non_maximum_suppression(loc_, onehot_, 0.5)
            final_bboxes = np.array(final_bboxes)
            self.nms_bboxes.append(final_bboxes)
            self.nms_labels.append(final_labels)

    def test_mathcing(self):
        matching(self.nms_bboxes, self.gt_coords_bucket)
        pass
