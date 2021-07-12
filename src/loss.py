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

    # get positive, negative mask
    # shape = (N_img, N_default_boxes)
    pos_mask = (true_cls[..., -1] != 1)
    neg_mask = (true_cls[..., -1] == 1)

    # get positive negative index for tensor
    # shape = (N_pos, 2=(axis0, axis=1)) or (N_neg, 2=(axis0, axis=1))
    pos_index_tf = np.stack(np.where(pos_mask), axis=-1)
    neg_index_tf = np.stack(np.where(neg_mask), axis=-1)

    # Extract positive dataset
    pos_true_cls = true_cls[pos_mask]
    pos_pred_cls = tf.gather_nd(pred_cls, pos_index_tf)
    neg_true_cls = true_cls[neg_mask]
    neg_pred_cls = tf.gather_nd(pred_cls, neg_index_tf)

    # Negative 데이터을 positive 3배 비율로 추출합니다.
    n_pos = len(pos_index_tf)
    n_neg = len(neg_index_tf)
    neg_rand_index = np.arange(n_neg)
    np.random.shuffle(neg_rand_index)
    neg_true_cls = neg_true_cls[neg_rand_index][:n_pos * 3]
    neg_pred_cls = tf.gather(neg_pred_cls, neg_rand_index)[:n_pos * 3]

    trgt_pred_cls = tf.concat([neg_pred_cls, pos_pred_cls], axis=0)
    trgt_true_cls = np.concatenate([neg_true_cls, pos_true_cls], axis=0)

    # Classification loss
    cls_loss = CategoricalCrossentropy()(trgt_true_cls, trgt_pred_cls)

    # extract positive localization
    pos_true_reg = true_reg[pos_mask]
    pos_pred_reg = tf.gather_nd(pred_reg, pos_index_tf)

    # Regression loss
    reg_loss = tf.reduce_mean(MSE(y_true=pos_true_reg, y_pred=pos_pred_reg))

    loss = cls_loss + reg_loss
    return loss
