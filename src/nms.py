from iou import calculate_iou
import numpy as np


def non_maximum_suppression(bboxes, onehots, threshold):
    """

    Description:
        비-최대 억제(non-max suppresion)은 object detector가 예측한
        bounding box 중에서 정확한 bounding box를 선택하도록 하는 기법입니다.
        Non-max suppression 알고리즘 작동 단계
            1. 하나의 클래스에 대한 bounding boxes 목록에서 가장 높은 점수를 갖고 있는 bounding box를 선택하고 목록에서 제거합니다.
                그리고 final box에 추가합니다.
            2. 선택된 bounding box를 bounding boxes 목록에 있는 모든 bounding box와 IoU를 계산하여 비교합니다.
                IoU가 threshold보다 높으면 bounding boxes 목록에서 제거합니다.
            3. bounding boxes 목록에 남아있는 bounding box에서 가장 높은 점수를 갖고 있는 것을 선택하고 목록에서 제거합니다.
                그리고 final box에 추가합니다.
            4. 다시 선택된 bounding box를 목록에 있는 box들과 IoU를 비교합니다. threshold보다 높으면 목록에서 제거합니다.
            5. bounding boxes에 아무것도 남아 있지 않을 때 까지 반복합니다.
            6. 각각의 클래스에 대해 위 과정을 반복합니다.

    :param bboxes: list, shape=(N, 4=(cx, cy, w, h)) bounding boxes 값이 들어 있어야 함.
    :param onehots: ndarray, shape=(N, n_classes), 단일 클래스에 대한 prediction 값이 들어 있어야 함.
    :param threshold: float, 동일 박스라고 판단하는 기준점. 0과 1 사의의 값을 가짐.

    :return:
        final_bboxes : list, 겹쳐진 박스가 모두 제거된 최종 박스
        final_onehots : list, 겹쳐진 박스가 모두 제거된 최종 박스의 확률값
        final_labels : list, 겹쳐진 박스가 모두 제거된 최종 박스의 class
    """

    # class 중 확률값이 가장 큰 class 을 선택합니다.
    # shape (N, 4) -> (N, )
    labels = np.argmax(onehots, axis=-1)
    probs = np.max(onehots, axis=-1)

    # prediction 및 bounding boxes을 prediction 값이 높은 순서로 정렬(내림 차순)
    sorted_index = np.argsort(probs)[::-1]
    bboxes = bboxes[sorted_index]
    labels = labels[sorted_index]

    final_bboxes = []
    final_labels = []
    final_onehots = []

    # bounding boxes에 아무것도 없을때까지 수행
    while bboxes.tolist():
        # prediction 값이 가장 높은 bounding box 후보군을 선택하고 final bboxes 집어 넣음.
        trgt_bbox = bboxes[0]
        final_bboxes.append(trgt_bbox)
        trgt_labels = labels[0]
        final_labels.append(trgt_labels)
        trgt_onehots = onehots[0]
        final_onehots.append(trgt_onehots)

        # 후보 bounding box을 bboxes 에서 제거함
        bboxes = np.delete(bboxes, 0, axis=0)
        labels = np.delete(labels, 0, axis=0)
        onehots = np.delete(onehots, 0, axis=0)

        # 후보 bbox 와 bboxes 와의 iou 을 계산함.
        ious = calculate_iou(trgt_bbox[None], bboxes)

        # 후보 bbox 와 bboxes 와의 iou 을 계산해 특정 threshold 이상 겹치는 bbox 는 bboxes 후보에서 제거
        overlay_index = np.where(np.squeeze(ious > threshold))
        bboxes = np.delete(bboxes, overlay_index, axis=0)
        onehots = np.delete(onehots, overlay_index, axis=0)
        labels = np.delete(labels, overlay_index, axis=0)

    return final_bboxes, final_onehots, final_labels
