import numpy as np
from matplotlib import pyplot as plt
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils.multi_gpu_utils import multi_gpu_model

from keras.regularizers import l2, l1
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import keras.backend as K

from project_config import Config

from data_generator import DataGenerator
from keras_rgb_joints_model import td_cnn_lstm
# from keras_lrcn import cnn_lstm_lrcn
# from keras_models import tcn_resnet
# from ed_tcn import ed_tcn
# from org_ed_tcn import ED_TCN
# from keras_rgb_joints_model import td_cnn_lstm
# from keras_conv2d_lstm import conv_lstm, conv_lstm_3d

K.clear_session()
# gpu parameters
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#gpu_config = tf.ConfigProto()
#gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.70
#set_session(tf.Session(config=gpu_config))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True  
sess = tf.Session(config=config)
set_session(sess)

# project parameters
config = Config()
data_processor = config.data_processor
# Parameters
# config = project_config.NTURGBDataset
# config = project_config.MSRDailyActivity


# OPTIMIZER PARAMS
# loss = 'categorical_crossentropy'
loss = config.params['loss']
lr = config.params['learning_rate']
momentum = config.params['momentum']
decay = config.params['decay']
activation = config.params['activation']
# optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=True)
optimizer = Adam(lr=lr, decay=decay)
# optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.99)
#optimizer = RMSprop(lr=lr, decay=decay, epsilon=1e-08)
# reg = l1(config.params['reg'])
reg = l2(config.params['reg'])


#  test train split
# partition, labels = split_test_train(config.test_train_split_source)

# data generators
training_generator = DataGenerator(data_processor, istrain=True, data_aug=True, config=config)
validation_generator = DataGenerator(data_processor, istrain=False, data_aug=True, config=config)
# test generator
# training_generator.__getitem__(2)
# validation_generator.__getitem__(2)
# exit()

# model = tcn_resnet_org(n_classes=config.params['n_classes'],
#                        dim=config.params['dim'],
#                        gap=True,
#                        dropout=dropout,
#                        kernel_regularizer=reg,
#                        activation=activation)

# n_nodes = [64, 96]
# model = ED_TCN(n_nodes, conv_len=25, n_classes=16, n_feat=60, max_len=190, causal='True')

# model = my_model_res_tcn(batch_size=config.params['batch_size'],
#                          n_classes=config.params['n_classes'],
#                          dim=config.params['dim'])

# model = ed_tcn(config.params['n_classes'], config.params['batch_size'], config.params['dim'])

model = td_cnn_lstm(n_classes=config.params['n_classes'],
                    batch_size=config.params['batch_size'],
                    all_dim=config.params['rgb_dim'],
                    model_weights=config.params['loading_checkpoint'],
                    kernel_regularizer=reg)

# model = cnn_lstm_lrcn(n_classes=config.params['n_classes'],
#                       batch_size=config.params['batch_size'],
#                       all_dim=config.params['rgb_dim'],
#                       pool_size=config.params['pool_size'],
#                       roi_size=config.params['roi_size'],
#                       kernel_regularizer=reg)

# model = conv_lstm(n_classes=config.params['n_classes'],
#                   batch_size=config.params['batch_size'],
#                   all_dim=config.params['rgb_dim'],
#                   kernel_regularizer=reg,
#                   activation=activation
#                   )

# model = conv_lstm(n_classes=config.params['n_classes'],
#                   batch_size=config.params['batch_size'],
#                   all_dim=config.params['rgb_dim'],
#                   kernel_regularizer=reg,
#                   activation=activation)

# model = multi_gpu_model(model, gpus=2)
print(model.summary())
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

try:

    checkpoint = ModelCheckpoint(config.params['saving_checkpoint'],
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True, mode='max')

    callbacks_list = [checkpoint]

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=config.params['steps_per_epoch'],
                        epochs=config.params['epochs'],
                        callbacks=callbacks_list,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        workers=6)

finally:

    print('highest acc train: {:.2f}'.format(np.max(model.history.history.get('acc'))))
    print('highest acc val: {:.2f}'.format(np.max(model.history.history.get('val_acc'))))
    print('highest acc val epoch: {}'.format(np.argmax(model.history.history.get('val_acc')) + 1))

    # serialize model to json
    model_json = model.to_json()
    with open(config.params['jason_file'], "w") as json_file:
        json_file.write(model_json)

    plt.scatter(range(len(model.history.history.get('val_loss'))),
                model.history.history.get('val_loss'), c='r')
    plt.scatter(range(len(model.history.history.get('loss'))),
                model.history.history.get('loss'), c='g')
    plt.show()
