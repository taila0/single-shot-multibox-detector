from model import simple_detection_netowrk
from tensorflow.python.keras.models import Model
import tensorflow.python.keras.backend as K
from prior import PriorBoxes
from dataset import DetectionDataset
from generator import label_generator
from loss import SSDLoss

if __name__ == '__main__':
    # n_classes
    n_classes = 11# with background
    n_anchors = 5
    image_shape = (128, 128)

    # Generate Dataset
    dataset = DetectionDataset(data_type='train')
    train_imgs, train_labs_info = dataset[:100]
    train_labs_bucket = train_labs_info.groupby('image_index')

    # Generate Detection Network
    inputs, pred = simple_detection_netowrk((128, 128, 3), n_anchors, n_classes)

    # Generate prior boxes
    strides = [4, 8, 16]
    scales = [10, 25, 40]
    ratios = [(1, 1),
              (1.5, 0.5),
              (1.2, 0.8),
              (0.8, 1.2),
              (1.4, 1.4)]
    prior = PriorBoxes(strides, scales, ratios)
    prior_boxes = prior.generate(image_shape)

    # Generate labels
    train_labs = label_generator(train_labs_bucket, prior_boxes, n_classes)

    # Define Loss
    ssd_loss = SSDLoss(1.0, 3.)

    # Training
    model = Model(inputs, pred)
    model.compile('adam', loss=ssd_loss)
    results = model.fit(train_imgs / 255.,
                        train_labs,
                        validation_split=0.1,
                        batch_size=5,
                        epochs=20)
