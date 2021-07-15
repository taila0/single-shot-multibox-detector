from unittest import TestCase
import numpy as np


class TestDatasets(TestCase):
    def setUp(self):
        self.true_labs = np.load('../datasets/true_labels.npy')
        self.true_imgs = np.load('../datasets/true_images.npy')

    def test_check_dataset(self):
        """
        Description:
        데이터가 장성적으로 저장 되었는지 확인합니다.
        """
        print(self.true_labs.shape)

