import numpy as np
import cv2
from utils import draw_rectangles, plot_images


def mAP(pred, true):
    """
    Description:
        Mean Average Precision 을 구합니다.
        detection 평가지표로 각 class 별 precision 을 계산 후 평균을 계산해 반환합니다.

        Positive, Negative 결정은 아래와 같습니다.
        1. classification 에서 찾고자 하는 class 와 같은지 여부만 확인 합니다.
        example)
            찾고자 하는 class 1
            prediction = [1, 1, 2, 2, 3, 3] => [T, T, N, N, N, N]

        True 와 False 의 기준 아래와 같습니다.
        아래 2개의 조건을 동시에 만족해야 True 가 됩니다.
        1. classification 정보가 맞는지 여부 확인
        2. IOU가 특정 기준점 이상이 되어야 함.

        AP(Average Precision) 은 IOU 을 기준으로 Precision-recall graph 의 면적을 의미합니다.
        mAP(mean Average Precision)은 모든 class 의 AP에 대한 평균 값을 의미합니다.

    Args:
        :param pred: ndarray, shape=(N_sample, 5=(cx cy w h class))
        :param true:  ndarray,  shape=(N_sample, 5=(cx cy w h class))

    :return: float
    """




if __name__ == '__main__':
    sample_bg = np.zeros([100, 100, 3])
    target_bboxes = np.array([[0, 0, 25, 25], [75, 75, 100, 100], [75, 0, 100, 25], [0, 75, 25, 100]])
    rected_img = draw_rectangles(sample_bg, target_bboxes)
    plot_images([rected_img] * 2)
