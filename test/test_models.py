from unittest import TestCase
from model import simple_detection_netowrk
import tensorflow.python.keras.backend as K
import tensorflow as tf
from dataset import DetectionDataset


class TestModel(TestCase):
    def setUp(self):
        dataset = DetectionDataset(data_type='train')
        self.train_imgs, _ = dataset[:2000]

    def testModel(self):
        inputs, preds = simple_detection_netowrk((128, 128, 3), 5, 11)
        self.sess = K.get_session()
        self.sess.run(tf.global_variables_initializer())
        preds_ = self.sess.run(preds, {inputs: self.train_imgs[:10]})
        print(preds_.shape)
