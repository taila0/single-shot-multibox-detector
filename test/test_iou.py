from unittest import TestCase
from iou import calculate_iou
from dataset import DetectionDataset
from utils import plot_images
from default_boxes import *
from utils import xywh2xyxy, draw_rectangles, images_with_rectangles
import matplotlib.pyplot as plt
from iou import calculate_iou


class TestIOU(TestCase):
    def setUp(self):
        self.default_bboxes = np.load('./test_data/default_boxes.npy')
        self.gt_coords = np.load('./test_data/gt_coords.npy')
        self.gt_labels = np.load('./test_data/gt_labels.npy')
        self.sample_img = np.load('./test_data/sample_img.npy')

        self.positive_index = np.array([95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 108, 109,
        110, 111, 112, 113, 115, 117, 118, 119, 136, 139, 141, 144, 145,
        146, 147, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 170,
        172, 173, 174, 191, 199, 205, 207, 210, 211, 212, 213, 214, 219,
        250, 252, 253, 254, 294])
        self.default_pos_bboxes = self.default_bboxes[self.positive_index]

    def test_calculate_iou(self):
        self.ious = calculate_iou(self.default_pos_bboxes, self.gt_coords)
        pass
