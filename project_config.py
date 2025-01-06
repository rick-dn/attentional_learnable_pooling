import os
import numpy as np
import data_processor_msr
import data_processor_ntu


class Config:

    params = {'batch_size': 4,
              'shuffle': True,
              'epochs': 200,
              'steps_per_epoch': 40,
              'learning_rate': 0.00001,
              'momentum': 0.9,
              'decay': 1e-6,  # 1e-4 1e-6
              'loss': 'sparse_categorical_crossentropy', # 'categorical_crossentropy',
              'reg': 1e-5,
              'activation': 'relu',
              'dataset': 'msr'}  # 'msr'

    def __init__(self):

        if self.params['dataset'] == 'msr':
            self.msr_daily_activity()
            self.data_processor = data_processor_msr.DataProcessor(self.params)
        elif self.params['dataset'] == 'ntu':
            self.ntu_rgbd_dataset()
            self.data_processor = data_processor_ntu.DataProcessor(self.params)
        else:
            raise ValueError('dataset should be either msr or ntu')

    def msr_daily_activity(self):

        dataset_dir = '/data/Rick/datasets/msr_daily_activity'
        working_dir = '/data/Rick/localdrive/myprojects/dataset_prep/msr_daily_activity_all'

        self.params['skeleton_raw_path'] = dataset_dir + '/*.txt'
        self.params['rgb_raw_path'] = dataset_dir + '/*.avi'
        self.params['depth_raw_path'] = dataset_dir + '/*.bin'

        self.params['skeleton_data'] = working_dir + '/msr_skeleton'
        self.params['rgb_data'] = working_dir + '/msr_rgb'
        self.params['depth_data'] = working_dir + '/depth_rgb'

        loading_checkpoint = working_dir + '/inception_resnet_self_attn_30_09'
        saving_checkpoint = working_dir + '/inception_resnet_multi_attn_4_layer_norm_res_11_11'

        if not os.path.exists(loading_checkpoint):
            os.mkdir(loading_checkpoint)
        if not os.path.exists(saving_checkpoint):
            os.mkdir(saving_checkpoint)

        self.params['time_size'] = 30
        self.params['no_of_joints'] = 20
        self.params['coords_per_joint'] = 3
        self.params['rgb_frame_size'] = 224
        self.params['joints_dim'] = (self.params['time_size'],
                                     self.params['no_of_joints'] * self.params['coords_per_joint'])
        self.params['rgb_dim'] = (self.params['time_size'],
                                  self.params['rgb_frame_size'],
                                  self.params['rgb_frame_size'], 3)
        self.params['roi_dim'] = (self.params['time_size'], self.params['no_of_joints'], 4)
        self.params['n_classes'] = 16

        self.params['roi_size'] = 7
        self.params['pool_size'] = 58

        self.params['jason_file'] = saving_checkpoint + '/model.json'
        self.params['loading_checkpoint'] = loading_checkpoint + '/model_weights.hdf5'
        self.params['saving_checkpoint'] = saving_checkpoint + '/model_weights.hdf5'


    def ntu_rgbd_dataset(self):

        dataset_dir = '/media/sf_F_DRIVE/localdrive/datasets/ntu'
        working_dir = '/media/sf_F_DRIVE/localdrive/myprojects/dataset_prep/ntu_rgbd'

        self.params['skeleton_raw_path'] = dataset_dir + '/nturgbd_skeletons/nturgbd_skeletons/*.skeleton'
        # TODO
        self.params['rgb_raw_path'] = dataset_dir + '/nturgbd_rgb/**/nturgb+d_rgb/*_rgb.avi'
        # self.params['depth_raw_path'] = dataset_dir + '**/*.bin'

        self.params['skeleton_data'] = working_dir + '/ntu_skeleton_dir_170'
        # TODO
        self.params['rgb_data'] = working_dir + '/ntu_rgb_dir'
        # self.params['depth_data'] = working_dir + 'depth_dir/'

        self.params['missing_skeleton'] = dataset_dir + \
            'NTURGB-D-matlab/NTURGB-D-master/Matlab/samples_with_missing_skeletons.txt'
        self.params['full_missing_skeleton'] = working_dir + 'full_missing_skeleton.txt'

        self.params['time_size'] = 85


def main():

    config = Config()
    pre_process = config.data_processor

    # process everything together
    if not os.path.exists(config.params['skeleton_data']):
        os.makedirs(config.params['skeleton_data'])
    if not os.path.exists(config.params['rgb_data']):
        os.makedirs(config.params['rgb_data'])
    rgb_data = pre_process.process_all_data()

    # ID = pre_process.partition['train'][np.random.randint(0, 156)]
    # rgb_file = np.load(config.params['rgb_data'] + "/" + ID + '_rgb.npy')
    # pre_process.data_aug_rgb(rgb_file)


if __name__ == '__main__':
    main()
