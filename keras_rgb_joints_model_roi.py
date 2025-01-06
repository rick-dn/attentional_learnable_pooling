from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, Lambda, InputLayer
from keras.layers import LSTM
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.core import *
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras_applications.mobilenet import MobileNet
from keras.layers.convolutional import *
from keras.layers import GaussianNoise
from keras.layers import SeparableConv1D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
import tensorflow as tf
from keras import backend as K

# from keras_models import tcn_resnet_org, tcn_resnet_2
from RoiPoolingConv import RoiPoolingConv


def time_distributed():
    # model = InceptionResNetV2(weights=None, input_tensor=x, include_top=False)
    # # print(model.summary())
    # model = model.layers[14].output

    RGB = Input(batch_shape=(32, 16, 224, 224, 3))
    LEFT = Input(batch_shape=(32, 16, 20, 4))

    rgb = Input(shape=(224, 224, 3))
    # j = Input(shape=(60))
    left = Input(shape=(20, 4))

    model = MobileNetV2(weights='imagenet', input_tensor=rgb, include_top=False)
    print(model.summary())
    exit()
    model = model.layers[18].output

    # model = ResNet50(weights='imagenet', input_tensor=rgb, include_top=False)
    #base_out = model.layers[-2].output
    # '''Get the output from conv 3rd block '''
    #base_out = model.get_layer('add_7').output
    # model = model.layers[77].output

    # model = RoiPoolingConv(pool_size=7, num_rois=20)([model, left])

    # model = Model(input=[rgb, left], output=model)
    model = Model(input=rgb, output=model)

    # model = RoiPoolingConv(pool_size=7, num_rois=20)

    # print(model.summary())
    # exit()

    # model = TimeDistributed(model)([RGB, LEFT])
    model = TimeDistributed(model)(RGB)
    # model = Model(input=[RGB, LEFT], output=model)
    model = Model(input=RGB, output=model)

    # print(model.summary())
    # exit()

    return model


def cnn_model(x):

    # model = InceptionResNetV2(weights=None, input_tensor=x, include_top=False)
    # # print(model.summary())
    # model = model.layers[14].output

    # model = MobileNetV2(weights=None, input_tensor=x, include_top=True)
    # model = model.layers[18].output
    
    model = ResNet50(weights='imagenet', input_tensor=x, include_top=False)
    #base_out = model.layers[-2].output
    # '''Get the output from conv 3rd block '''
    #base_out = model.get_layer('add_7').output
    base_out = model.layers[77].output

    return model


def td_cnn_lstm(n_classes, batch_size, all_dim, pool_size, roi_size, kernel_regularizer):

    print('input shape : ', *all_dim)
    print('batch size: ', batch_size)
    dim, left_dim, rgb_dim = all_dim

    # input
    J = Input(batch_shape=(batch_size, *dim))
    L = Input(batch_shape=(batch_size, *left_dim))
    R = Input(batch_shape=(batch_size, *rgb_dim))

    # time distributed
    x = Lambda(lambda i: K.reshape(i, (batch_size * rgb_dim[0], *rgb_dim[1:])))(R)
    rois = Lambda(lambda i: K.reshape(i, (batch_size * left_dim[0], *left_dim[1:])))(L)
    print('x shape after td lambda: ', x.shape)
    print('rois shape after td lambda: ', rois.shape)

    # normal cnnmodel
    # model = InceptionResNetV2(weights=None, input_tensor=x, include_top=False)
    # y = model.layers[10].output

    model = MobileNetV2(weights=None, input_tensor=x, include_top=False)
    print(model.summary())
    y = model.layers[10].output
    print('y after cnn: ', y)

    # roi pool
    roi_pool = RoiPoolingConv(pool_size=7, num_rois=20)([y, rois])
    print('roi pool output shape: ', roi_pool.shape)

    # spatially distributed
    # y = Lambda(lambda i: K.reshape(i, (batch_size, rgb_dim[0], y.shape[1] * y.shape[2] * y.shape[3])))(y)
    roi_pool = Lambda(lambda i: K.reshape(i, (batch_size, rgb_dim[0],
                                              roi_pool.shape[1] *
                                              roi_pool.shape[2] *
                                              roi_pool.shape[3] *
                                              roi_pool.shape[4])))(roi_pool)
    print('roi_pool.shape after sd lambda: ', roi_pool.shape)

    y = LSTM(128, kernel_regularizer=kernel_regularizer, return_sequences=True, kernel_initializer='he_normal')(roi_pool)
    y = LSTM(128, return_sequences=False, kernel_regularizer=kernel_regularizer, kernel_initializer='he_normal')(y)
    y = Dense(n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, kernel_initializer='he_normal')(y)

    # final model object
    model = Model(input=[J, L, R], output=y)
    # model = Model(input=R, output=y)

    return model


def main():

    # time_distributed()
    # exit()

    frames = 32
    model = td_cnn_lstm(n_classes=16, batch_size=16,
                        all_dim=[(frames, 60), (frames, 20, 4), (frames, 224, 224, 3)],
                         pool_size=55, roi_size=7, kernel_regularizer=None)

    # model = td_cnn_lstm(n_classes=16, batch_size=16,
    #                    all_dim=(frames, 224, 224, 3),
    #                    pool_size=55, roi_size=7, kernel_regularizer=None)

    # x = Input(batch_shape=(16, 229, 229, 3))
    # model = MobileNetV2(weights=None, input_tensor=x, include_top=True)
    # model = model.layers[18].output
    # model = Flatten()(model)
    # model = Dense(16, activation='softmax')(model)
    # model = Model(input=x, outputs=model)

    print(model.summary())


if __name__ == '__main__':
    main()

