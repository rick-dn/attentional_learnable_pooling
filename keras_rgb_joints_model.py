from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, Lambda, InputLayer
from keras.layers.merge import add, concatenate, Concatenate, Add
from keras.models import Model
from keras.layers.core import *
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.resnet50 import ResNet50
# from keras_applications.mobilenet import MobileNet
from keras.layers.convolutional import *
from keras.layers import MaxPooling1D, Bidirectional
from keras.layers import GaussianNoise
from keras.layers import SeparableConv1D
from keras.layers.recurrent import *
from keras_layer_normalization import LayerNormalization
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
import tensorflow as tf
from keras import backend as K
from keras_self_attention import ScaledDotProductAttention, SeqSelfAttention
from keras_multi_head import MultiHeadAttention
# from keras_models import tcn_resnet_org, tcn_resnet_2
from encoder_decoder import EncoderDecoder
# from RoiPoolingConv import RoiPoolingConv

import os


def time_distributed(model_weights):
    # model = InceptionResNetV2(weights=None, input_tensor=x, include_top=False)
    # # print(model.summary())
    # model = model.layers[14].output

    RGB = Input(batch_shape=(32, 16, 224, 224, 3))
    LEFT = Input(batch_shape=(32, 16, 20, 4))

    rgb = Input(shape=(224, 224, 3))
    # j = Input(shape=(60))
    left = Input(shape=(20, 4))

    model = InceptionResNetV2(weights='imagenet', input_tensor=rgb, include_top=False, pooling='avg')
    # model = InceptionResNetV2(weights='imagenet', input_tensor=rgb, include_top=False)
    # # print(model.summary())
    # model = model.layers[14].output

    # model = MobileNetV2(weights='imagenet', input_tensor=rgb, include_top=False, pooling='avg')
    # model = MobileNetV2(weights='imagenet', input_tensor=rgb, include_top=False)
    # model = model.layers[18].output

    # model = ResNet50(weights='imagenet', input_tensor=rgb, include_top=False)
    # base_out = model.layers[-2].output
    # model = model.layers[77].output

    # model = RoiPoolingConv(pool_size=7, num_rois=20)([model, left])

    # model = Model(input=[rgb, left], output=model)
    model = Model(input=rgb, output=model.output)

    # model = RoiPoolingConv(pool_size=7, num_rois=20)

    # print(model.summary())
    # exit()

    # model = TimeDistributed(model)([RGB, LEFT])
    model = TimeDistributed(model)(RGB)
    # model = Model(input=[RGB, LEFT], output=model)
    model = Model(input=RGB, output=model)

    if os.path.isfile(model_weights):
        model.load_weights(model_weights, by_name=True)

    # print(model.summary())
    # exit()

    return model


def cnn_model(x):

    # model = InceptionResNetV2(weights=None, input_tensor=x, include_top=False)
    # # print(model.summary())
    # model = model.layers[14].output

    model = MobileNetV2(weights=None, input_tensor=x, include_top=True)
    model = model.layers[18].output

    return model


def td_cnn_lstm(n_classes, batch_size, all_dim, model_weights, kernel_regularizer):

    # normal cnnmodel
    # x = Input(shape=(229, 229, 3))
    # model = InceptionResNetV2(weights=None, input_tensor=x, include_top=True)
    # model = model.layers[10]

    # model = Sequential()
    # model.add(InputLayer(batch_input_shape=(16, 30, 229, 229, 3)))
    # model.add(TimeDistributed(InceptionResNetV2(weights=None, include_top=True).layers[50]))

    print('input shape : ', *all_dim)
    # dim, left_dim, rgb_dim = all_dim
    rgb_dim = all_dim

    # input
    # J = Input(batch_shape=(batch_size, *dim))
    # L = Input(batch_shape=(batch_size, *left_dim))
    R = Input(batch_shape=(batch_size, *rgb_dim))

    # time distributed
    # y = Lambda(lambda i: K.reshape(i, (batch_size * rgb_dim[0], *rgb_dim[1:])))(R)
    # rois = Lambda(lambda i: K.reshape(i, (batch_size * left_dim[0], *left_dim[1:])))(L)

    cnn_output = time_distributed(model_weights)(R)

    # cnn_output = Lambda(lambda x: K.reshape(x, (batch_size, rgb_dim[0],
    #                                                   cnn_output.shape[2])))(cnn_output)

    # roi_pool = RoiPoolingConv(pool_size=7, num_rois=20)([y, rois])
    # print(cnn_output.summary())

    print('y.shape after cnn: ', cnn_output.shape)

    # y = TimeDistributed(Flatten())(y)
    # y = cnn_model(y)

    # print('y.shape after time distributed flatten: ', y.shape)

    # roi pooling
    # print('rois: ', rois)
    # roi_pool = RoiPoolingConv(pool_size=7, num_rois=20)([y, rois])
    # print('roi_pool: ', roi_pool)

    # spatially distributed
    # y = Lambda(lambda i: K.reshape(i, (batch_size, rgb_dim[0], y.shape[1] * y.shape[2] * y.shape[3])))(y)
    # roi_pool = Lambda(lambda i: K.reshape(i, (batch_size, rgb_dim[0],
    #                                           roi_pool.shape[1] *
    #                                           roi_pool.shape[2] *
    #                                           roi_pool.shape[3] *
    #                                           roi_pool.shape[4])))(roi_pool)

    # print('roi_pool.shape after lambda: ', roi_pool.shape)

    # roi_pool = TimeDistributed(MaxPooling2D(pool_size=2))(roi_pool)
    #
    # print(print('roi_pool.shape after max pool: ', roi_pool.shape))
    #
    # roi_pool = Lambda(lambda i: K.reshape(i, (batch_size, rgb_dim[0],
    #                                           roi_pool.shape[0] *
    #                                           roi_pool.shape[1] *
    #                                           roi_pool.shape[2])))(roi_pool)




    #  no roi
    # y = tcn_resnet_org(n_classes=n_classes,
    #                    dim=(int(y.shape[1]), int(y.shape[2])),
    #                    gap=1,
    #                    dropout=0.5,
    #                    kernel_regularizer=kernel_regularizer,
    #                    activation='relu')(y)

    #  with roi

    # y = tcn_resnet_org(n_classes=n_classes,
    #                    dim=(int(roi_pool.shape[1]), int(roi_pool.shape[2])),
    #                    gap=1,
    #                    dropout=0.5,
    #                    kernel_regularizer=kernel_regularizer,
    #                    activation='relu')(roi_pool)

    # y = LSTM(64, kernel_regularizer=kernel_regularizer)(roi_pool)
    # y = SeqSelfAttention()(cnn_output)

    y = MultiHeadAttention(head_num=4)(cnn_output)
    y = Add()([y, cnn_output])
    print('y after concat: ', y)
    y = LayerNormalization()(y)
    # y = Bidirectional(LSTM(32, kernel_regularizer=kernel_regularizer, return_sequences=True, kernel_initializer='he_normal'))(y)
    # print('y lstm 1: ', y)
    # y = Bidirectional(LSTM(64, kernel_regularizer=kernel_regularizer, return_sequences=True, kernel_initializer='he_normal'))(y)
    # y = MaxPooling1D(strides=128)(y)
    # y = EncoderDecoder(output_dim=64)(cnn_output)
    # y = ScaledDotProductAttention()(cnn_output)
    y = Flatten()(y)
    y = Dense(n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, kernel_initializer='he_normal')(y)

    # model = TimeDistributed(model)(x)

    # model = TimeDistributed(Flatten())(model)
    # model = Dense(16, activation='softmax')(model)

    # final model object
    # model = Model(input=[J, L, R], output=y)
    model = Model(input=R, output=y)

    return model


def main():

    # time_distributed()
    # exit()

    frames = 32
    # model = td_cnn_lstm(n_classes=16, batch_size=16,
    #                          all_dim=[(frames, 60), (frames, 20, 4), (frames, 224, 224, 3)],
    #                          pool_size=55, roi_size=7, kernel_regularizer=None)

    model = td_cnn_lstm(n_classes=16, batch_size=16,
                        all_dim=(frames, 224, 224, 3),
                        model_weights=None, kernel_regularizer=None)

    # x = Input(batch_shape=(16, 229, 229, 3))
    # model = MobileNetV2(weights=None, input_tensor=x, include_top=True)
    # model = model.layers[18].output
    # model = Flatten()(model)
    # model = Dense(16, activation='softmax')(model)
    # model = Model(input=x, outputs=model)

    print(model.summary())


if __name__ == '__main__':
    main()

