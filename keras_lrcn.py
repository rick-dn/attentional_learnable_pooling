from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, Lambda, InputLayer
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.core import *
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras_applications.mobilenet import MobileNet
from keras.layers.convolutional import *
from keras.layers import GaussianNoise
from keras.layers import SeparableConv1D
from keras.layers.recurrent import *
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
import tensorflow as tf
from keras import backend as K

# from keras_models import tcn_resnet_org, tcn_resnet_2
# from RoiPoolingConv import RoiPoolingConv


def cnn_lstm_lrcn(n_classes, batch_size, all_dim, pool_size, roi_size, kernel_regularizer):
    """Build a CNN into RNN.
    Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
    Heavily influenced by VGG-16:
        https://arxiv.org/abs/1409.1556
    Also known as an LRCN:
        https://arxiv.org/pdf/1411.4389.pdf
    """

    rgb_dim = all_dim
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), kernel_initializer="he_normal",
                                     kernel_regularizer=kernel_regularizer,
                                     activation='relu', padding='same'), input_shape=rgb_dim))
    model.add(TimeDistributed(Conv2D(32, (3, 3),
                                     kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3, 3), kernel_initializer="he_normal",
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), kernel_initializer="he_normal",
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3, 3), kernel_initializer="he_normal",
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (3, 3), kernel_initializer="he_normal",
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(256, (3, 3), kernel_initializer="he_normal",
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3, 3), kernel_initializer="he_normal",
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(512, (3, 3), kernel_initializer="he_normal",
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(512, (3, 3), kernel_initializer="he_normal",
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(n_classes, kernel_initializer="he_normal", activation='softmax'))

    return model


def main():

    # time_distributed()
    # exit()

    frames = 32
    # model = td_cnn_lstm(n_classes=16, batch_size=16,
    #                          all_dim=[(frames, 60), (frames, 20, 4), (frames, 224, 224, 3)],
    #                          pool_size=55, roi_size=7, kernel_regularizer=None)

    model = cnn_lstm_lrcn(n_classes=16, batch_size=16,
                          all_dim=(frames, 224, 224, 3),
                          pool_size=55, roi_size=7, kernel_regularizer=None)

    # x = Input(batch_shape=(16, 229, 229, 3))
    # model = MobileNetV2(weights=None, input_tensor=x, include_top=True)
    # model = model.layers[18].output
    # model = Flatten()(model)
    # model = Dense(16, activation='softmax')(model)
    # model = Model(input=x, outputs=model)

    print(model.summary())


if __name__ == '__main__':
    main()