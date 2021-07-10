from model import simple_detection_netowrk
from tensorflow.python.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from prior import PriorBoxes
from src.dataset import DetectionDataset
from generator import DetectionGenerator
from loss import SSDLoss

if __name__ == '__main__':
    # n_classes
    n_classes = 11 # with background
    n_anchors = 5
    image_shape = (128, 128)

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

    # Generate Dataset
    trainset = DetectionDataset(data_type='train')
    validset = DetectionDataset(data_type='validation')
    traingen = DetectionGenerator(trainset.config,
                                  prior.config,
                                  batch_size=64)
    validgen = DetectionGenerator(validset.config,
                                  prior.config,
                                  batch_size=64)
    # Define Loss
    ssd_loss = SSDLoss(1.0, 3.)

    # Training
    model = Model(inputs, pred)
    model.compile(Adam(1e-3),
                  loss=SSDLoss(1.0, 3.))

    rlrop = ReduceLROnPlateau(factor=0.1,
                              min_lr=1e-6,
                              patience=5,
                              cooldown=3)
    callbacks = []
    callbacks.append(rlrop)
    model.fit_generator(traingen,
                        epochs=50,
                        validation_data=validgen,
                        use_multiprocessing=True,
                        workers=6,
                        callbacks=callbacks)