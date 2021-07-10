import tensorflow as tf
import numpy as np


def tiling_default_boxes(center_xy, sizes):
    """
    Description:
    original image 와 좌표 위치가 매칭된 feature map의 모든 cell에 모든 default box가 적용된 좌표값을 반환합니다.
    :param center_xy: shape=(N, 2), feature map의 각 cell과 original image에 매칭되는 좌표
    :param sizes: tuple, shape=(n_scales, n_ratios, 2), ratio에 scale 이 곱해진 결과값을 반환
        example)
            scales = (10, 100)
            ratios = ((1, 1) (0.5 ,1))
            return = [[[10. 10.], [5.  10.]],
                      [[100. 100.], [50.  100.]]]
    :return:
    """
    # ((w, h), (w, h) ... (w, h)) 로 구성되어 있음
    sizes = sizes.reshape(-1, 2)

    # broadcasting 을 위해 center_xy 을 중첩,
    # shape: (# center_xy, 2) -> (# sizes, # center xy, 2)
    stacked_xy = np.stack([center_xy] * len(sizes), axis=1)

    # broadcasting 을 위해 shape 을 변경함
    # shape (# center_xy, 2) -> (# sizes, # center xy, 2)
    stacked_wh = np.stack([sizes] * len(center_xy), axis=0)

    return np.concatenate([stacked_xy, stacked_wh], axis=-1)


def generate_default_boxes(scales, ratios):
    """
    Description:

    :param scales: tuple or list, (int, int, ... int ), shape=(n_shape, )
        example) (3, 6, 9)
        ratio 가 1 일때 default 박스의 size 크기
    :param ratios: tuple or list, ((H_ratio, W_ratio), (H_ratio, W_ratio) ... (H_ratio, W_ratio)) , shape=(n_ratio, 2)
        default boxes의 h, w 정보가 순차적으로 들어있는 자료구조.
        example) ((1, 0.5), (1, 1), ... (0.5, 1))
    :return: tuple, shape = (n_scales, n_ratios, 2) scale 별 ratio 가 적용된 h, w 을  반환
        example)
            scales = (10, 100)
            ratios = ((1, 1) (0.5 ,1))
            return = [[[10. 10.], [5.  10.]],
                      [[100. 100.], [50.  100.]]]

    """
    # shape (n_ratios, 2) -> (n_ratios, 1, 2)
    ratios = np.expand_dims(ratios, axis=1)

    # shape (n_scales) -> (n_scales, 1)
    scales = np.expand_dims(scales, axis=1)

    # shape (n_ratios, n_scales, 2) -> (n_scales, n_ratios, 2)
    scale_per_ratios = np.transpose(ratios * scales, axes=[1, 0, 2])

    return scale_per_ratios


def original_rectangle_coords(fmap_size, kernel_sizes, strides, paddings):
    """
    Description:
    주어진 Feature map의 center x, center y 좌표를 Original Image center x, center y에 맵핑합니다.
    아래 코드에서 사용된 공식은 "A guide to convolution arithmetic for deep learning" 에서 가져옴

    :param fmap_size: 1d array, 최종 출력된 fmap shape, (H, W) 로 구성
        example) (4, 4)
    :param kernel_sizes: tuple or list, 각 Layer 에 적용된 filter 크기,
        example) [3, 3]
    :param strides: tuple or list, 각 Layer 에 적용된 stride 크기
        example) [2, 1]
    :param paddings: tuple or list, List 의 Element 가 'SAME', 'VALID' 로 구성되어 있어야 함
        example) ['SAME', 'VALID']
    :return: feature map의 각 cell과 original image에 매칭되는 좌표를 반환,
        example)
           feature map
            +---+---+
            | a | b |
            +---+---+
            | c | d |
            +---+---+
                    a               b               c               d
            [[cx, cy, w, h], [cx, cy, w, h] [cx, cy, w, h], [cx, cy, w, h]]
    """

    rf = 1  # receptive field
    jump = 1  # 점과 점사이의 거리
    start_out = 0.5
    assert len(kernel_sizes) == len(strides) == len(paddings), 'kernel sizes, strides, paddings 의 크기가 같아야 합니다.'

    for stride, kernel_size, padding in zip(strides, kernel_sizes, paddings):
        # padding 의 크기를 계산합니다.
        if padding == 'SAME':
            padding = (kernel_size - 1) / 2
        else:
            padding = 0

        # 시작점을 계산합니다.
        start_out += ((kernel_size - 1) * 0.5 - padding) * jump

        # receptive field 을 계산합니다.
        rf += (kernel_size - 1) * jump

        # 점과 점사이의 거리를 계산합니다.
        jump *= stride

    xs, ys = np.meshgrid(range(fmap_size[0]), range(fmap_size[1]))
    xs = xs * jump + start_out
    ys = ys * jump + start_out
    ys = ys.ravel()
    xs = xs.ravel()
    n_samples = len(xs)

    # coords = ((cx, cy, w, h), (cx, cy, w, h) ... (cx, cy, w, h))
    coords = np.stack([ys, xs, [rf] * n_samples, [rf] * n_samples], axis=-1)
    return coords


if __name__ == '__main__':
    """
    feature map 의 각 cell의 center을 original image center에 맵핑함.
        example) [[cx cy  w  h], [[cx cy  w  h]] -> [[cx cy], [cx, cy]], shape (height * width, 2)
    get each cell center x, center y coordinates for original image
    """
    fmap = tf.constant(shape=(2, 32, 32, 2), value=1)
    n_layer = 7
    paddings = ['SAME'] * n_layer
    strides = [1, 1, 1, 1, 2, 1, 2]
    kernel_sizes = [3] * n_layer
    center_xy = original_rectangle_coords((32, 32), kernel_sizes, strides, paddings)[:, :2]

    # get w, h
    scales = [10, 25, 40]
    ratios = [(1, 1),
              (1.5, 0.5),
              (1.2, 0.8),
              (0.8, 1.2),
              (1.4, 1.4)]
    sizes = generate_default_boxes(scales, ratios)

    # Get default boxes over feature map
    default_boxes = tiling_default_boxes(center_xy, sizes)
