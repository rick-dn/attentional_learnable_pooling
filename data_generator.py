import numpy as np
import keras
# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
#import skvideo.io
import cv2 as cv

from data_processor_msr import DataProcessor
# from msr_daily_activity_roi import my_pooling


class DataGenerator(keras.utils.Sequence):

    """Generates data for Keras"""

    def __init__(self, data_processor, istrain=True, data_aug=False, batch_size=32, dim=None, rgb_dim=None, left_dim=None,
                 n_classes=None, save_skel_dir=None, save_rgb_dir=None, pool_size=None, roi_size=None,
                 shuffle=True, config=None):
        """Initialization"""

        self.data_processor = data_processor
        if istrain:
            self.list_IDs = data_processor.partition['train']
        else:
            self.list_IDs = data_processor.partition['validation']
        self.labels = data_processor.labels

        self.joints_dim = config.params['joints_dim']
        self.left_dim = config.params['roi_dim']
        self.rgb_dim = config.params['rgb_dim']
        self.batch_size = config.params['batch_size']
        self.n_classes = config.params['n_classes']
        self.shuffle = shuffle
        self.on_epoch_end()
        self.save_skel_dir = save_skel_dir
        self.rgb_data_dir = config.params['rgb_data']
        self.data_aug = data_aug
        self.pool_size = config.params['pool_size']
        self.roi_size = config.params['roi_size']

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        # Initialization

        joint_batch = np.empty((self.batch_size, *self.joints_dim))
        rgb_batch = np.empty((self.batch_size, *self.rgb_dim))
        roi_batch = np.empty((self.batch_size, *self.left_dim))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store class
            y[i] = self.labels[ID]
            # print('ID: ', ID)
            # print('label: ', self.labels[ID])
            # print('rgb file name: ', self.rgb_data_dir + "/" + ID + '_rgb.npy')

            # Store sample
            # skel_data = np.load(self.save_skel_dir + ID + '_skeleton.npy').item()
            rgb_file = np.load(self.rgb_data_dir + "/" + ID + '_rgb.npy')
            # print(rgb file shape data gen: ',rgb_file.shape)

            if self.data_aug:
                rgb_file = self.data_processor.data_aug_rgb(rgb_file)

            # my_pooling(skel_data=skel_data['data'],
            #            rgb_data=rgb_data,
            #            frame_info=skel_data['frame_info'])

            # if self.data_aug:
            #     skel_data, rgb_data, frame_info = data_aug_skel_only(skel_data=skel_data['data'],
            #                                                          rgb_data=rgb_data,
            #                                                          frame_info=skel_data['frame_info'])
            # else:
            #     skel_data, frame_info = skel_data['data'], skel_data['frame_info']
            #
            # roi_data[i, ] = get_roi_coord(skel_data, rgb_data, frame_info, self.roi_size, self.pool_size)
            #
            # X[i, ] = normalize_and_reshape(skel_data, frame_info)
            rgb_file = self.data_processor.normalize_rgb(rgb_file)
            # rgb_file = preprocess_input(rgb_data)
            rgb_batch[i, ] = rgb_file

        # return rgb_data, keras.utils.to_categorical(y, num_classes=self.n_classes)
        # return X, y
        # return [X, roi_data, R], y
        return rgb_batch, y
