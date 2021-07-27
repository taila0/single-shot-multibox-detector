import numpy as np
from iou import calculate_iou
from utils import draw_rectangles, plot_images
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def match_gt(preds_loc, trues_loc, trues_cls, threshold=0.5, background_class=10):
    """
    Description:
    각 예측한 bounding box(ĝ)에 알맞는 ground truth 을 할당합니다.
    예측한 bounding box 와 ground truth 간 IOU 을 계산해 IOU 가 가장 높은 ground turth 을 정답으로 매칭 합니다.

    :param preds_loc: Ndarray, 2D array, shape: (N_pr_boxes, 4=(cx, cy, w, h) + n_classes)
    :param trues_loc: Ndarray, 2D array ,shape: (N_gt_boxes, 4=(cx, cy, w, h) + n_classes)

    :return:
    """
    # 각 이미지 별 ground truth 와 nms 가 적용된 prediction 간의 iou 을 계산
    gt_labels_bucket = []
    for pred_loc, true_loc, true_cls in zip(preds_loc, trues_loc, trues_cls):
        ious = calculate_iou(pred_loc, true_loc)

        # ious 중 가장 iou matching 이 많이된 class 을 할당합니다.
        gt_indices = np.argmax(ious, axis=-1)
        gt_labels = true_cls[gt_indices]

        # iou 중 threshold 보다 높은 ground truth 가 없으면 배경 클래스로 지정합니다.
        bg_mask = np.all(ious < threshold, axis=-1)
        gt_labels[bg_mask] = background_class

        # 이미지 별 계산된 anchor 별 ground turth class을 리스트에 추가합니다.
        gt_labels_bucket.append(gt_labels)

    return gt_labels_bucket


def mAP(onehots, trues, visualization=True):
    """
    Description:
        Mean Average Precision 을 구합니다.
        detection 평가지표로 각 class 별 precision 을 계산 후 평균을 계산해 반환합니다.

        Prediction 에서 Positive, Negative 결정은 아래와 같습니다.
        1. classification 에서 찾고자 하는 class 와 같은지 여부만 확인 합니다.
        example)
            찾고자 하는 class가 1 이라면
            prediction = [1, 1, 2, 2, 3, 3] => [T, T, N, N, N, N]

        True 와 False 의 기준 아래와 같습니다.
        아래 2개의 조건을 동시에 만족해야 True 가 됩니다.
            1. True, Pred 둘다 모두 Positive 인지 아닌지
            2. IOU가 0.5 이상 되는지 아닌지.

        AP(Average Precision) 은 IOU 을 기준으로 Precision-recall graph 의 면적을 의미합니다.
        mAP(mean Average Precision)은 모든 class 의 AP에 대한 평균 값을 의미합니다.

    Args:
        :param onehots: ndarray, shape=(N_sample, N_classes=(cx cy w h 0 ... N_classes-1))
        :param trues:  ndarray,  shape=(N_sample, N_classes=(cx cy w h 0 ... N_classes-1))
        :param visualization:  bool,

    :return: float
    """
    n_classes = onehots.shape[-1]
    # 배경 클래스를 제외한 모든 클래스의 precision-recall graph 을 그립니다.
    aps = []
    for class_index in range(n_classes-1):
        pos_trues = (trues == class_index)
        pos_probs = onehots[:, class_index]
        ap = average_precision_score(y_true=pos_trues, y_score=pos_probs)
        aps.append(ap)

        precision, recall, thresholds = precision_recall_curve(y_true=pos_trues, probas_pred=pos_probs)
        plt.plot(recall, precision)

    if visualization:
        plt.show()
    return np.mean(aps)


if __name__ == '__main__':
    sample_bg = np.zeros([100, 100, 3])
    target_bboxes = np.array([[0, 0, 25, 25], [75, 75, 100, 100], [75, 0, 100, 25], [0, 75, 25, 100]])
    rected_img = draw_rectangles(sample_bg, target_bboxes)
    plot_images([rected_img] * 2)

    a = [2, 1, 3, 8, 4, 10]
    sorted_index = np.argsort(a)[::-1]
    print(sorted_index)
