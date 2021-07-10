from tensorflow.python.keras.layers import Input, Conv2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Concatenate, Reshape
from tensorflow.python.keras.layers import Softmax


def simple_detection_netowrk(input_shape, n_anchors, n_classes):
    inputs = Input(shape=input_shape)

    num_features = 16
    conv1_1 = Conv2D(num_features, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
    norm1_1 = BatchNormalization(name='norm1_1')(conv1_1)

    conv1_2 = Conv2D(num_features, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_2')(norm1_1)
    norm1_2 = BatchNormalization(name='norm1_2')(conv1_2)

    conv1_3 = Conv2D(num_features, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_3')(norm1_2)
    norm1_3 = BatchNormalization(name='norm1_3')(conv1_3)

    # block 2
    conv2_1 = Conv2D(num_features * 2, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_1')(norm1_3)
    norm2_1 = BatchNormalization(name='norm2_1')(conv2_1)

    conv2_2 = Conv2D(num_features * 2, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu',
                     name='conv2_2')(norm2_1)
    norm2_2 = BatchNormalization(name='norm2_2')(conv2_2)

    # block 3
    conv3_1 = Conv2D(num_features * 4, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_1')(norm2_2)
    norm3_1 = BatchNormalization(name='norm3_1')(conv3_1)

    conv3_2 = Conv2D(num_features * 4, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu',
                     name='conv3_2')(norm3_1)
    norm3_2 = BatchNormalization(name='norm3_2')(conv3_2)

    # block 4
    conv4_1 = Conv2D(num_features * 8, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_1')(norm3_2)
    norm4_1 = BatchNormalization(name='norm4_1')(conv4_1)

    conv4_2 = Conv2D(num_features * 8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu',
                     name='conv4_2')(norm4_1)
    norm4_2 = BatchNormalization(name='norm4_2')(conv4_2)

    # block 5
    conv5_1 = Conv2D(num_features * 8, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_1')(norm4_2)
    norm5_1 = BatchNormalization(name='norm5_1')(conv5_1)

    conv5_2 = Conv2D(num_features * 8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu',
                     name='conv5_2')(norm5_1)
    norm5_2 = BatchNormalization(name='norm5_2')(conv5_2)

    clss3_3 = Conv2D(n_anchors * n_classes, (3, 3), padding='same', activation=None, name='clas3_3')(norm3_2)
    rshp3_4 = Reshape((-1, n_classes), name='rshp3_4')(clss3_3)
    soft3_5 = Softmax(name='soft3_5')(rshp3_4)
    locz3_6 = Conv2D(n_anchors * 4, (3, 3), padding='same', activation=None, name='locz3_6')(norm3_2)
    rshp3_7 = Reshape((-1, 4), name='rshp3_7')(locz3_6)
    cnct3_8 = Concatenate(axis=-1, name='cnct3_8')([soft3_5, rshp3_7])

    # multi head 4
    clss4_3 = Conv2D(n_anchors * n_classes, (3, 3), padding='same', activation=None, name='clas4_3')(norm4_2)
    rshp4_4 = Reshape((-1, n_classes), name='rshp4_4')(clss4_3)
    soft4_5 = Softmax(name='soft4_5')(rshp4_4)
    locz4_6 = Conv2D(n_anchors * 4, (3, 3), padding='same', activation=None, name='locz4_6')(norm4_2)
    rshp4_7 = Reshape((-1, 4), name='rshp4_7')(locz4_6)
    cnct4_8 = Concatenate(axis=-1, name='cnct4_8')([soft4_5, rshp4_7])

    # multi head 5
    clss5_3 = Conv2D(n_anchors * n_classes, (3, 3), padding='same', activation=None, name='clas5_3')(norm5_2)
    rshp5_4 = Reshape((-1, n_classes), name='rshp5_4')(clss5_3)
    soft5_5 = Softmax(name='soft5_5')(rshp5_4)
    locz5_6 = Conv2D(n_anchors * 4, (3, 3), padding='same', activation=None, name='locz5_6')(norm5_2)
    rshp5_7 = Reshape((-1, 4), name='rshp5_7')(locz5_6)
    cnct5_8 = Concatenate(axis=-1, name='cnct5_8')([soft5_5, rshp5_7])

    cnct6_1 = Concatenate(axis=1, name='concat6_1')([cnct3_8, cnct4_8, cnct5_8])
    return inputs, cnct6_1
