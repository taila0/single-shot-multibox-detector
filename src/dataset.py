import numpy as np
import cv2
import os
import pandas as pd
import wget
from utils import draw_rectangles, xywh2xyxy

DOWNLOAD_URL_FORMAT = "https://s3.ap-northeast-2.amazonaws.com/pai-datasets/all-about-mnist/{}/{}.csv"
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")


class DetectionDataset:
    def __init__(self, data_type="train",
                 digit=(2, 6), rescale_ratio=(.8, 2.5),
                 color_noise=(0.3, 0.7), bg_size=(128, 128),
                 bg_noise=(0, 0.1)):
        """
        generate data for Detection

        :param data_type: Select one, (train, test, validation)
        :param digit : the length of number (몇개의 숫자를 serialize할 것인지 결정)
          if digit is integer, the length of number is always same value.
          if digit is tuple(low_value, high_value),
          the length of number will be determined within the range

        :param bg_size : the shape of background image
        :param bg_noise : the background noise of image,
               bg_noise = (gaussian mean, gaussian stddev)
        """
        self.images, self.labels = load_dataset("mnist", data_type)
        if isinstance(digit, int):
            self.digit_range = (digit, digit + 1)
        else:
            self.digit_range = digit
        self.num_data = len(self.labels) // (self.digit_range[1] - 1)
        self.num_classes = 10
        self.index_list = np.arange(len(self.labels))

        self.rescale_ratio = rescale_ratio
        self.bg_size = bg_size
        self.bg_noise = bg_noise
        self.color_noise = color_noise
        self.config = {
            "data_type": data_type,
            "digit": digit,
            "rescale_ratio": rescale_ratio,
            "color_noise": color_noise,
            "bg_size": bg_size,
            "bg_noise": bg_noise
        }

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if isinstance(index, int):
            num_digit = np.random.randint(*self.digit_range)
            start_index = (self.digit_range[1] - 1) * index
            digits = self.index_list[start_index:start_index + num_digit]

            digit_images = self.images[digits]
            digit_labels = self.labels[digits].values
            image, digit_positions = self._scatter_random(digit_images)

            # (xmin, xmax, ymin, ymax) -> (center_x, center_y, width, height)
            center_x = (digit_positions[:, 0] + digit_positions[:, 1]) / 2
            center_y = (digit_positions[:, 2] + digit_positions[:, 3]) / 2
            width = (digit_positions[:, 1] - digit_positions[:, 0])
            height = (digit_positions[:, 3] - digit_positions[:, 2])
            digit_positions = np.stack([center_x, center_y, width, height], axis=-1)
            digit_info = np.concatenate([np.ones_like(digit_labels[:, None]) * index,
                                         digit_positions,
                                         digit_labels[:, None]],
                                        axis=1)
            digit_df = pd.DataFrame(digit_info,
                                    columns=['image_index', 'cx', 'cy', 'w', 'h', 'label'])
            digit_df.image_index = digit_df.image_index.astype(np.int)
            digit_df.label = digit_df.label.astype(np.int)
            return image, digit_df
        else:
            batch_images, batch_df = [], []
            indexes = np.arange(self.num_data)[index]
            for _index in indexes:
                num_digit = np.random.randint(*self.digit_range)
                start_index = (self.digit_range[1] - 1) * _index
                digits = self.index_list[start_index:start_index + num_digit]

                digit_images = self.images[digits]
                digit_labels = self.labels[digits].values

                image, digit_positions = self._scatter_random(digit_images)

                batch_images.append(image)
                # (xmin, xmax, ymin, ymax) -> (center_x, center_y, width, height)
                center_x = (digit_positions[:, 0] + digit_positions[:, 1]) / 2
                center_y = (digit_positions[:, 2] + digit_positions[:, 3]) / 2
                width = (digit_positions[:, 1] - digit_positions[:, 0])
                height = (digit_positions[:, 3] - digit_positions[:, 2])
                digit_positions = np.stack([center_x, center_y, width, height], axis=-1)

                digit_info = np.concatenate([np.ones_like(digit_labels[:, None]) * _index,
                                             digit_positions,
                                             digit_labels[:, None]],
                                            axis=1)
                digit_df = pd.DataFrame(digit_info,
                                        columns=['image_index', 'cx', 'cy', 'w', 'h', 'label'])
                digit_df.image_index = digit_df.image_index.astype(np.int)
                digit_df.label = digit_df.label.astype(np.int)
                batch_df.append(digit_df)

            return np.stack(batch_images), pd.concat(batch_df)

    def shuffle(self):
        indexes = np.arange(len(self.images))
        np.random.shuffle(indexes)

        self.images = self.images[indexes]
        self.labels = self.labels[indexes]
        self.labels.index = np.arange(len(self.labels))

    def _scatter_random(self, images):
        background = np.random.normal(*self.bg_noise,
                                      size=(*self.bg_size, 3))
        positions = []

        for image in images:
            image = self._rescale_random(image)
            background, position = self._place_random(image,
                                                      background,
                                                      positions)
            positions.append(position)

        return background, np.array(positions)

    def _rescale_random(self, image):
        value = np.random.uniform(*self.rescale_ratio)
        image = cv2.resize(image, None, fx=value, fy=value)
        return image

    def _place_random(self, image, background, prev_pos):
        height, width = self.bg_size
        height_fg, width_fg = image.shape

        x_min, x_max, y_min, y_max = crop_fit_position(image)

        counter = 10
        while True:
            y = np.random.randint(0, height - height_fg - 1)
            x = np.random.randint(0, width - width_fg - 1)

            position = np.array([(x_min + x), (x_max + x), (y_min + y), (y_max + y)])

            if not self._check_overlap(position, prev_pos) or counter < 0:
                # 이전의 object랑 겹치지 않으면 넘어감
                break
            counter -= 1
        image = image[..., None] * np.random.uniform(*self.color_noise, size=(1, 1, 3))
        background[y:y + height_fg, x:x + width_fg] += image

        background = np.clip(background, 0., 1.)
        return background, position

    def _check_overlap(self, curr_pos, prev_pos):
        if len(prev_pos) == 0:
            return False
        else:
            prev_pos = np.array(prev_pos)

        # 각 면적 구하기
        curr_area = (curr_pos[1] - curr_pos[0]) * (curr_pos[3] - curr_pos[2])
        prev_area = (prev_pos.T[1] - prev_pos.T[0]) * (prev_pos.T[3] - prev_pos.T[2])

        # Intersection 면적 구하기
        _, it_min_xs, _, it_min_ys = np.minimum(curr_pos, prev_pos).T
        it_max_xs, _, it_max_ys, _ = np.maximum(curr_pos, prev_pos).T

        it_width = ((it_min_xs - it_max_xs) > 0) * (it_min_xs - it_max_xs)
        it_height = ((it_min_ys - it_max_ys) > 0) * (it_min_ys - it_max_ys)

        intersection = (it_width * it_height)
        # 전체 면적 구하기
        union = (curr_area + prev_area) - intersection
        # IOU가 5%이상이 되는 겹침 현상 발생하면, 겹쳤다고 판정
        return np.max(intersection / union) >= 0.05

    def get_config(self):
        return self.config


def crop_fit_position(image):
    """
    get the coordinates to fit object in image

    :param image:
    :return:
    """
    positions = np.argwhere(
        image >= 0.1)  # set the threshold to 0.1 for reducing the noise

    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)

    return np.array([x_min, x_max, y_min, y_max])


def load_dataset(dataset, data_type):
    """
    Load the MNIST-Style dataset
    if you don't have dataset, download the file automatically

    :param dataset: Select one, (mnist, fashionmnist, handwritten)
    :param data_type: Select one, (train, test, validation)
    :return:
    """
    if dataset not in ["mnist", "fashionmnist", "handwritten"]:
        raise ValueError(
            "allowed dataset: mnist, fashionmnist, handwritten")
    if data_type not in ["train", "test", "validation"]:
        raise ValueError(
            "allowed data_type: train, test, validation")

    file_path = os.path.join(
        DATASET_DIR, "{}/{}.csv".format(dataset, data_type))

    if not os.path.exists(file_path):
        os.makedirs(os.path.split(file_path)[0], exist_ok=True)
        url = DOWNLOAD_URL_FORMAT.format(dataset, data_type)
        wget.download(url, out=file_path)

    df = pd.read_csv(file_path)

    images = df.values[:, 1:].reshape(-1, 28, 28)
    images = images / 255  # normalization, 0~1
    labels = df.label  # label information
    return images, labels


if __name__ == '__main__':
    mnist_dataset = DetectionDataset()
    # label : image no, cx, cy, w, h, class
    image, labels = mnist_dataset[0]

    xywh = np.array(labels)[:, 1:5]
    xyxy = xywh2xyxy(xywh)

    rected_images = draw_rectangles((image*255).astype('uint'), xyxy)
    import matplotlib.pyplot as plt
    plt.imshow(rected_images)
    plt.show()
