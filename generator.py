import numpy as np
from tensorflow.python.keras.utils import to_categorical
from iou import calculate_iou
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.utils import to_categorical


def label_generator(train_labels_bucket, prior_boxes, n_classes):
    train_labels = []
    for index, train_label_df in train_labels_bucket:
        gt_boxes = train_label_df[['cx', 'cy', 'w', 'h']].values  # shape => [N_prior , 4]
        gt_labels = train_label_df['label'].values  # shape => [N_prior, N_classes]

        # Calculate IOU
        iou = calculate_iou(prior_boxes, gt_boxes)

        # Generate Detection Labels
        matched_labels = matching_labels_coords(prior_boxes, iou, n_classes, gt_boxes, gt_labels)
        train_labels.append(matched_labels)
    return np.asarray(train_labels)


def matching_labels_coords(prior_boxes, iou, n_classes, gt_boxes, gt_labels, threshold=0.5):
    """

    :param iou:
    :param n_classes:
    :param gt_boxes: shape => [N_prior , 4]
    :param gt_labels: shape =>  [N_prior , N_classes(with background)]
        count 을 1 부터 시작한다. mnist 경우 N_classes 는 11 이 된다.
    :return:

    TODO: Matching Policy 을 분리 시켜야 한다.
    """
    # background index 는 가장 마지막 index 로 한다.
    BACKGROUND_INDEX = n_classes - 1

    # threshold 이상이면 positive 로 본다
    match_indices = np.argwhere(iou > threshold)
    pr_match_indices = match_indices[:, 0]
    gt_match_indices = match_indices[:, 1]

    # matching labels
    prior_labels = np.ones(iou.shape[0]) * BACKGROUND_INDEX  # shape => [N_prior]
    matching_labels = gt_labels[gt_match_indices] # shape => [N_prior]
    prior_labels[pr_match_indices] = matching_labels
    prior_onehot_labels = to_categorical(prior_labels, n_classes)

    # matching coordinates
    matching_gt_coords = gt_boxes[gt_match_indices]
    matching_pr_coords = prior_boxes[pr_match_indices]

    # coordinates -> delta
    dx = (matching_gt_coords[:, 0] - matching_pr_coords[:, 0]) / matching_pr_coords[:, 2]
    dy = (matching_gt_coords[:, 1] - matching_pr_coords[:, 1]) / matching_pr_coords[:, 3]
    dw = np.log(matching_gt_coords[:, 2] / matching_pr_coords[:, 2])
    dh = np.log(matching_gt_coords[:, 3] / matching_pr_coords[:, 3])
    delta_loc = np.stack([dx, dy, dw, dh], axis=-1)

    delta_coords = np.zeros([prior_boxes.shape[0], 4])
    delta_coords[pr_match_indices] = delta_loc
    merge_y = np.concatenate([prior_onehot_labels, delta_coords], axis=-1)
    return merge_y


from prior import PriorBoxes
from dataset import DetectionDataset


class DetectionGenerator(Sequence):
    'Generates Localization dataset for Keras'
    def __init__(self, dataset:DetectionDataset, prior:PriorBoxes,
                 batch_size=32, best_match_policy=False, shuffle=True):
        'Initialization'
        # Dictionary로 받았을 때에만 Multiprocessing이 동작가능함.
        # Keras fit_generator에서 Multiprocessing으로 동작시키기 위함
        if isinstance(dataset, dict):
            self.dataset = DetectionDataset(**dataset)
        elif isinstance(dataset, DetectionDataset):
            self.dataset = dataset
        else:
            raise ValueError('dataset은 dict혹은 DetectionDataset Class로 이루어져 있어야 합니다.')

        if isinstance(prior, dict):
            self.prior = PriorBoxes(**prior)
        elif isinstance(prior, PriorBoxes):
            self.prior = prior
        else:
            raise ValueError('PriorBoxes은 dict 혹은 PriorBoxes Class로 이루어져 있어야 합니다.')

        self.batch_size = batch_size
        self.best_match_policy = best_match_policy
        self.shuffle = shuffle
        self.num_classes = self.dataset.num_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        images, ground_truths = self.dataset[self.batch_size * index:
                                             self.batch_size * (index + 1)]
        pr_boxes = self.prior.generate(images.shape[1:])
        y_trues = label_generator(ground_truths.groupby('image_index'), pr_boxes, self.num_classes + 1)
        return images, y_trues

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.dataset.shuffle()
