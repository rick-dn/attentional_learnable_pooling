from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, Lambda, InputLayer, Flatten
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.convolutional_recurrent import ConvLSTM2D
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


def conv_lstm_3d(n_classes, batch_size, all_dim,
                 kernel_regularizer=None, activation='relu',
                 roi_size=None, pool_size=None,):

    # input
    print('input shape : ', all_dim)
    # dim, left_dim, rgb_dim = all_dim
    rgb_dim = all_dim

    # input
    # J = Input(batch_shape=(batch_size, *dim))
    # L = Input(batch_shape=(batch_size, *left_dim))
    # print('L: ', L)
    # R = Input(batch_shape=(batch_size, *rgb_dim))

    seq = Sequential()
    seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                       input_shape=rgb_dim,
                       kernel_initializers='he_normal',
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), strides=2,
                       kernel_initializers='he_normal',
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), strides=2,
                       kernel_initializers='he_normal',
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), strides=2,
                       kernel_initializers='he_normal',
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   kernel_initializers='he_normal',
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))

    seq.add(Flatten())

    seq.add(Dense(n_classes, kernel_initializers='he_normal', activation='softmax'))

    return seq


def conv_lstm(n_classes, batch_size, all_dim,
              kernel_regularizer=None, activation='relu',
              roi_size=None, pool_size=None,):

    # input
    print('input shape : ', all_dim)
    # dim, left_dim, rgb_dim = all_dim
    rgb_dim = all_dim

    # input
    # J = Input(batch_shape=(batch_size, *dim))
    # L = Input(batch_shape=(batch_size, *left_dim))
    # print('L: ', L)
    # R = Input(batch_shape=(batch_size, *rgb_dim))

    model = Sequential()
    model.add(ConvLSTM2D(30,  kernel_size=(3, 3),  return_sequences=True,
                         kernel_initializer='he_normal',
                         input_shape=rgb_dim))
    # returns a sequence of vectors of dimension 32
    model.add(ConvLSTM2D(30, kernel_size=(3, 3),
                         kernel_initializer='he_normal',
                         return_sequences=True))
    # returns a sequence of vectors of dimension 32
    model.add(ConvLSTM2D(16, kernel_initializer='he_normal',
                         kernel_size=(3, 3)))
    # return a single vector of dimension 32
    # model.add(Dense(32, activation=activation))
    model.add(Flatten())
    model.add(Dense(n_classes, kernel_initializer='he_normal',
                    activation='softmax'))

    return model


def main():

    # time_distributed()
    # exit()

    frames = 32
    # model = conv_lstm(n_classes=16, batch_size=16,
    #                   all_dim=[(frames, 60), (frames, 20, 4), (frames, 224, 224, 3)],
    #                   pool_size=55, roi_size=7, kernel_regularizer=None)

    model = conv_lstm_3d(n_classes=16, batch_size=16,
                         all_dim=(16, 224, 224, 3),
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
