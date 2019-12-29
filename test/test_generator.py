import numpy as np
from unittest import TestCase
from dataset import DetectionDataset
from prior import PriorBoxes
from iou import calculate_iou
import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical
from generator import matching_labels_coords, DetectionGenerator, label_generator


class TestGenerator(TestCase):

    def setUp(self):
        # sample data generator
        batch_size = 64
        self.dataset = DetectionDataset(data_type='train')
        self.imgs, self.labs_info = self.dataset[:batch_size]
        print('Number of Classes : {}'.format(self.dataset.num_classes))
        print('Image shape : {} || Label shape : {}'.format(self.imgs.shape, self.labs_info.shape))
        self.strides = [4, 8, 16]
        self.scales = [10, 25, 40]
        self.ratios = [(1, 1),
                  (1.5, 0.5),
                  (1.2, 0.8),
                  (0.8, 1.2),
                  (1.4, 1.4)]

        self.prior = PriorBoxes(self.strides, self.scales, self.ratios)
        self.prior_boxes = self.prior.generate((128, 128))  # prior boxes shape : (6720, 4)

        # 0번째 이미지를 쌤플 이미지로 사용함.
        self.group_labs = self.labs_info.groupby('image_index')
        for ind, labs in self.group_labs:
            self.labs = labs
            break
        self.gt_boxes = self.labs[['cx', 'cy', 'w', 'h']].values
        self.gt_labels = self.labs['label'].values
        self.iou = calculate_iou(self.prior_boxes, self.gt_boxes)
        print('Ground Truths Shape : {}'.format(self.gt_boxes.shape))
        print('IOU Shape : {}'.format(self.iou.shape))
        print(list(self.labs.groupby('image_index')))

    def test_generator(self):

        BACKGROUND_INDEX = self.dataset.num_classes
        match_indices = np.argwhere(self.iou > 0.5)
        pr_match_indices = match_indices[:, 0]# 어떤 prior boxes 에 matching 될 것 인지
        gt_match_indices = match_indices[:, 1]# 어떤 ground truth 에 matching 될 것 인지

        # labels
        prior_labels = np.ones(self.prior_boxes.shape[0]) * BACKGROUND_INDEX # shape : [N_prior]
        matching_labels = self.gt_labels[gt_match_indices]  # prior_label shape : [N_prior]
        prior_labels[pr_match_indices] = matching_labels
        prior_onehot_labels = to_categorical(prior_labels, self.dataset.num_classes+1)

        # coordinates
        matching_gt_coords = self.gt_boxes[gt_match_indices]
        matching_pr_coords = self.prior_boxes[pr_match_indices]

        # coordinates -> delta
        dx = (matching_gt_coords[:, 0] - matching_pr_coords[:, 0]) / matching_pr_coords[:, 2]
        dy = (matching_gt_coords[:, 1] - matching_pr_coords[:, 1]) / matching_pr_coords[:, 3]
        dw = np.log(matching_gt_coords[:, 2]/matching_pr_coords[:, 2])
        dh = np.log(matching_gt_coords[:, 3]/matching_pr_coords[:, 3])
        delta_loc = np.stack([dx, dy, dw, dh], axis=-1)

        # merge labels with delta
        delta_coords = np.zeros([self.prior_boxes.shape[0], 4])
        delta_coords[pr_match_indices] = delta_loc
        merged_y = np.concatenate([prior_onehot_labels, delta_coords], axis=-1)

        # Sangjae
        gen = DetectionGenerator(self.dataset, self.prior, 1, shuffle=False)
        train_imgs, train_labs = gen[0]

        # 우선적으로 이미지가 같은 이미지인지 확인
        np.testing.assert_almost_equal(train_imgs, self.imgs[0:1])

        # 라벨인 같은지 확인
        np.testing.assert_almost_equal(
            matching_labels_coords(self.prior_boxes, self.iou, 11, self.gt_boxes, self.gt_labels),
            merged_y
        )

    def test_label_generator(self):

        labels = label_generator(self.labs_info.groupby('image_index'), self.prior_boxes, 11)
        print(labels.shape)