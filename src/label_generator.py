from dataset import DetectionDataset
from delta import calculate_delta
from default_boxes import *
from utils import xywh2xyxy, xyxy2xywh
from iou import calculate_iou


def label_generator(default_bboxes, gt_bboxes):
    """
    Desscription:
        이미지 한장에 대해 Detection 을 위한 라벨을 생성합니다.
        라벨 데이터는 값은 default bbox 에 대한 이미지 내 obj 의 상대적 위치 값(delta)로 이루어져 있습니다.
         - 아래 순서로 진행됩니다.
            1. default bboxes 생성
            2. default bboxes 와 ground truth 간의 iou 계산
            3. 학습할 default bbox 선택 (Matching policy)
            4. delta 계산
    Args:
        default_bboxes: ndarray, shape=(N_default_bbox, 4=(cx cy ,w, h))
        gt_bboxes: , shape=(N_gt, 4=(cx, cy, w, h))
    Returns:

    """
    ious = calculate_iou(default_bboxes, gt_bboxes)

    # iou 중 가장 overlay 비율이 큰 class을 선택합니다.
    # shape = (N_default_boxes, )
    max_overlay_cls = np.argmax(ious, axis=-1)

    # 모든 obj 에 대해 iou 가 0.5 이하이면 background class, -1로 지정합니다.
    background_mask = np.all(ious < 0.5, axis=-1)
    max_overlay_cls[background_mask] = -1

    # 기존의 정답 데이터에 [0, 0, 0, 0] 을 추가합니다.
    gt_with_bg = np.concatenate([gt_coords, np.array([[0, 0, 0, 0]])], axis=0)

    # 각 default boxes에 해당하는 ground truth 의 좌표값을 가져옵니다.
    true_reg = gt_with_bg[max_overlay_cls]

    # iou 가 0.5 이상 되는 값들의 index 을 가져옵니다.
    model_true_bboxes = gt_with_bg[max_overlay_cls]
    pos_mask = np.all(true_reg, axis=-1)

    # delta 값을 계산합니다.
    pos_true_bboxes = xyxy2xywh(model_true_bboxes[pos_mask])
    pos_default_boxes = xyxy2xywh(default_boxes[pos_mask])
    pos_true_delta = calculate_delta(pos_default_boxes, pos_true_bboxes)
    true_delta = np.zeros_like(default_boxes)
    true_delta[pos_mask] = pos_true_delta
    return true_delta


if __name__ == '__main__':
    # Get sample image and object coordinates
    trainset = DetectionDataset(data_type='train')
    gt_img, gt_info = trainset[0]
    gt_coords = gt_info.iloc[:, 1:5].values
    gt_coords = xywh2xyxy(gt_coords)
    gt_labels = gt_info.iloc[:, -1].values

    fmap = tf.constant(shape=(2, 8, 8, 2), value=1)
    h, w = fmap.get_shape()[1:3]
    n_layer = 11
    paddings = ['SAME'] * n_layer
    strides = [1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2]
    kernel_sizes = [3] * n_layer
    center_xy = original_rectangle_coords((h, w), kernel_sizes, strides, paddings)[:, :2]

    # get w, h
    scales = [30]
    ratios = [(1, 1),
              (1.5, 0.5),
              (1.2, 0.8),
              (0.8, 1.2),
              (1.4, 1.4)]

    # 적용할 default box 의 크기
    boxes = generate_default_boxes(scales, ratios)

    # Get default boxes(cx cy w h) over feature map,
    default_boxes = tiling_default_boxes(center_xy, boxes)

    # default boxes
    default_boxes = default_boxes.reshape(-1, 4)

    # ground truth coordinates(x1, y1, x2, y2), shape = (N_obj, 4)
    gt_coords = gt_coords.reshape(-1, 4)
    gt_coords = xyxy2xywh(gt_coords)

    # 이미지 한장에 대한 detection 라벨을 생성합니다.
    true_reg = label_generator(default_boxes, gt_coords)

