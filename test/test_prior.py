from unittest import TestCase
from prior import PriorBoxes
from model import simple_detection_netowrk
import tensorflow.python.keras.backend as K
import tensorflow as tf
from dataset import DetectionDataset


class TestPrior(TestCase):
    def setUp(self):
        pass;

    def test_generator(self):
        self.strides = [4, 8, 16]
        self.scales = [10, 25, 40]
        self.ratios = [(1, 1),
                       (1.5, 0.5),
                       (1.2, 0.8),
                       (0.8, 1.2),
                       (1.4, 1.4)]

        self.prior = PriorBoxes(self.strides, self.scales, self.ratios)
        self.prior_boxes = self.prior.generate((128, 128))  # prior boxes shape : (6720, 4)
        print(self.prior_boxes.shape)



