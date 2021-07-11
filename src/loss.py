from tensorflow.keras.losses import MSE, CategoricalCrossentropy
import tensorflow as tf
import numpy as np


def detection_loss(y_true, y_pred):
    """
    Description:
        일번적인 detection model 의 loss 을 구합니다.
        구해야 할 loss는 아래와 같습니다.
        loss = Classification loss + Regression loss

        Classification loss:
            모든 데이터에 대해 classification loss 을 계산합니다.
        Regression loss:
            iou 가 50 이상인 positive 데이터 셋에 대해서만 loss 을 구합니다.

    :param y_true: (N data, N default_boxes, 4 + 1)
    :param y_pred: (N data, N default_boxes, 4 + n_classes)
    (※ background class 은 -1 로 지정되어 있음)
    """

    # classification error
    true_reg = y_true[..., :4]
    true_cls = y_true[..., 4:]
    pred_reg = y_pred[..., :4]
    pred_cls = y_pred[..., 4:]

    # Classification loss
    cls_loss = CategoricalCrossentropy()(true_cls, pred_cls)

    # Regression loss
    pos_index = (true_cls[..., -1] != 1)
    pos_true_reg = true_reg[pos_index]
    pos_pred_reg = pred_reg[pos_index]

    reg_loss = tf.reduce_mean(MSE(y_true=pos_true_reg, y_pred=pos_pred_reg))

    loss = cls_loss + reg_loss
    return loss
