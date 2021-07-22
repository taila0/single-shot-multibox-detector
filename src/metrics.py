import numpy as np
import cv2
from iou import calculate_iou
from utils import draw_rectangles, plot_images
from sklearn.metrics import recall_score, precision_score


def matching(pred, true):
    """
    Description:
    각 prediction 에 ground truth 값을 1:1 매칭 합니다.
    만약 prediction에 해당 하는 ground truth 가 없을 때에는 FP 가 됩니다 .

    :param pred: Ndarray, 2D array, shape: (N_pr_boxes, 4=(cx cy w, h))
    :param true: Ndarray, 2D array ,shape: (N_gt_boxes, 4=(cx cy w, h))

    :return:
    """

    ious = calculate_iou(pred, true)


def mAP(pred, true):
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
        :param pred: ndarray, shape=(N_sample, 4+N_classes=(cx cy w h 0 ... N_classes-1))
        :param true:  ndarray,  shape=(N_sample, 4+N_classes=(cx cy w h 0 ... N_classes-1))

    :return: float
    """

    pred_reg = pred[..., :4]
    pred_onehot = pred[..., 4:]

    true_reg = true[..., :4]
    true_onehot = true[..., 4:]

    # 각 class 별 AP 계산
    n_classes = np.shape(true_onehot)[-1]
    average_precisions = {}





    return np.mean(average_precisions)


if __name__ == '__main__':
    sample_bg = np.zeros([100, 100, 3])
    target_bboxes = np.array([[0, 0, 25, 25], [75, 75, 100, 100], [75, 0, 100, 25], [0, 75, 25, 100]])
    rected_img = draw_rectangles(sample_bg, target_bboxes)
    plot_images([rected_img] * 2)

    a = [2, 1, 3, 8, 4, 10]
    sorted_index = np.argsort(a)[::-1]
    print(sorted_index)
