import numpy as np
import cv2 as cv
import glob
from tqdm import tqdm
import re
from skimage.transform import rescale, rotate
from skimage.util import random_noise
from skimage import exposure
from scipy import ndimage
import random
from matplotlib import pyplot as plt
import time
import os
from sys import exit


class DataProcessor:
    def __init__(self, config):

        self.params = config
        self.partition, self.labels = self.split_test_train()

    def split_test_train(self):

        partition = {}
        labels = {}
        train_ids = []
        val_ids = []

        train_counter = 0
        val_counter = 0

        # getting ID from text file
        for index, filename in enumerate(glob.iglob(self.params['rgb_data'] + '/*.npy', recursive=True)):

            # print('filename: ', filename)
            ID = re.search("a[0-1][0-9]_s[0-1][0-9]_e0[1-2]", filename).group()
            subject = int(re.search("_s(.+?)_", ID).group(1))
            activity = int(re.search("a(.+?)_", ID).group(1)) - 1

            # print('subject activity: ', subject, activity)

            if subject <= 5:
                train_ids.append(ID)
                labels[ID] = activity
                train_counter += 1

            else:
                val_ids.append(ID)
                labels[ID] = activity
                val_counter += 1

        partition['train'] = train_ids
        partition['validation'] = val_ids

        print('train counter, val counter: ', train_counter, val_counter)

        return partition, labels

    @staticmethod
    def crop_image(image, joints_gt, bbox):

        x_min, y_min, x_max, y_max = bbox

        image = image[y_min:y_max, x_min:x_max]
        joints_gt -= np.array([x_min, y_min])

        return image, joints_gt

    @staticmethod
    def crop_to_bbox(rgb_data, pskel_data, bbox):

        x_min, y_min, x_max, y_max = bbox

        rgb_data = rgb_data[:, y_min:y_max, x_min:x_max]
        pskel_data[:] -= np.array([x_min, y_min, 0])

        return rgb_data, pskel_data

    @staticmethod
    def calc_joints_bbox(frame_info, pskel_data):

        all_bbox = []

        for joints_gt in pskel_data:

            x_min = np.min(joints_gt[:, 0][joints_gt[:, 0] != 0])
            y_min = np.min(joints_gt[:, 1][joints_gt[:, 1] != 0])
            x_max = np.max(joints_gt[:, 0][joints_gt[:, 0] != 0])
            y_max = np.max(joints_gt[:, 1][joints_gt[:, 1] != 0])

            x_min -= 50
            y_min -= 50
            x_max += 50
            y_max += 50

            frame_bbox = [x_min.clip(min=0).astype(int),
                          y_min.clip(min=0).astype(int),
                          x_max.clip(max=frame_info[1]).astype(int),
                          y_max.clip(max=frame_info[2]).astype(int)]
            all_bbox.append(frame_bbox)
        all_bbox = np.asarray(all_bbox)

        return [np.min(all_bbox[:, 0]), np.min(all_bbox[:, 1]), np.max(all_bbox[:, 2]), np.max(all_bbox[:, 3])]

    def data_aug(self, skel_data, rgb_data, frame_info):
        """Rotate the image.

        Rotate the image such that the rotated image is enclosed inside the tightest
        rectangle. The area not occupied by the pixels of the original image is colored
        black.

        Parameters
        ----------

        rgb_data : numpy.ndarray or None
            numpy vid

        skel_data: numpy array of vid joints

        frame_info: no of frames, no of joints, coords per joint

        Returns
        -------

        numpy.ndarray
            Rotated Image

        """
        # grab the dimensions of the image and then determine the
        # centre

        angle = np.random.randint(-30, 30)
        scale = np.random.uniform(.75, 1.25)

        # De-normalize
        # skel_data = coord_de_normalize(skel_data.reshape(frame_info[0], 20, 3), frame_info)

        # just for checking
        # print('data aug input shapess: ', rgb_data.shape, skel_data.shape)

        # for (frame, joint) in zip(rgb_data[:1], skel_data[:1]):
        #
        #     # draw joints matplotlib
        #     plt.frame_infoscatter(joint[:, 0], joint[:, 1])
        #     # Display the resulting frame
        #     plt.imshow(frame)
        #     plt.show()
        #     exit()
        # exit()

        (h, w) = frame_info[1:3]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv.getRotationMatrix2D((cX, cY), angle, scale)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        rgb_data_aug = []
        for index, image in enumerate(rgb_data):

            # perform the actual rotation and return the image
            image = cv.warpAffine(image, M, (nW, nH))

            # resize to fit nn
            image = cv.resize(image, (299, 299))

            rgb_data_aug.append(image)
        rgb_data_aug = np.asarray(rgb_data_aug)

        skel_data_aug = np.zeros((frame_info[0], 20, 3))
        for index, joints_gt in enumerate(skel_data):

            # joints transform
            ones = np.ones(shape=(len(joints_gt), 1))
            points_ones = np.hstack([joints_gt[:, 0:2], ones])
            joints_gt[:, 0:2] = M.dot(points_ones.T).T

            skel_data_aug[index] = joints_gt

        skel_data_aug = np.asarray(skel_data_aug)

        # for (frame, joint) in zip(rgb_data_aug[:1], skel_data_aug[:1]):
        #
        #     # draw joints matplotlib
        #     plt.scatter(joint[:, 0], joint[:, 1])
        #     # Display the resulting frame
        #     plt.imshow(frame)
        #     plt.show()
        #     exit()
        # exit()

        # resize to fit nn
        Rx, Ry = 299/nW, 299/nH

        skel_data_aug[:, :, 0] = skel_data_aug[:, :, 0] * Rx
        skel_data_aug[:, :, 1] = skel_data_aug[:, :, 1] * Ry

        frame_info = [frame_info[0], 299, 299, 3]

        # print('frame info: ', frame_info)
        # print('rgb data shape: ', rgb_data_aug.shape)
        # print('skel data shape: ', skel_data_aug.shape)

        return skel_data_aug, rgb_data_aug, frame_info

    @staticmethod
    def get_roi_coord(joints_gt, rgb_data, frame_info, roi_size, filter_size):

        # print('roi size, filter size: ', roi_size, filter_size)
        # print('frame info: ', frame_info)

        # plt.scatter(joints_gt[0, :, 0], joints_gt[0, :, 1])
        # plt.imshow(rgb_data[0])
        # plt.show()

        fx, fy = filter_size / frame_info[1], filter_size / frame_info[2]

        joints_gt = joints_gt[:, :, :2]

        joints_gt[:] *= np.array([fx, fy])

        shift = roi_size // 2
        left = joints_gt[:] - np.array([shift, shift])
        left[left < 0] = 0  # out of bound check lower bound

        right = left + np.array([roi_size, roi_size])
        left[right > filter_size] = filter_size - roi_size

        pool_dim = filter_size * np.ones((*left.shape[:2], 2))
        left = np.c_[left, pool_dim]

        # for testing
        # image = cv.resize(rgb_data[0], (filter_size, filter_size))
        # plt.scatter(joints_gt[0, :, 0], joints_gt[0, :, 1])
        # plt.scatter(left[0, :, 0], left[0, :, 1])
        # plt.imshow(image)
        # plt.show()

        # hstack filter size
        # print('left: ', left)

        return left

    def data_aug_rgb(self, rgb_data):

        # generate randomness
        angle = np.random.randint(-30, 30)
        scale = np.random.uniform(.75, 1.25)
        seed = np.random.randint(0, 10)

        # for rotation and scale
        (h, w) = rgb_data.shape[1:3]
        (cX, cY) = (w // 2, h // 2)
        # get rotation matrix
        M = cv.getRotationMatrix2D((cX, cY), angle, scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        rgb_data_aug =[]
        for frame in rgb_data:

            # introduce noise
            # frame = random_noise(frame, seed=seed)
            # perform the actual rotation and return the image
            frame = cv.warpAffine(frame, M, (nW, nH))
            # resize to fit nn
            frame = cv.resize(frame, (224, 224))
            rgb_data_aug.append(frame)

            # testing
            # print('frame shape: ', frame.shape)
            # cv.imshow('frame', cv.cvtColor(frame, cv.COLOR_RGB2BGR))
            # cv.imshow('frame', frame)
            # cv.waitKey()

        return np.asarray(rgb_data_aug, dtype=np.float32)

    def normalize_and_reshape(self, skel_data, frame_info):
        # checking
        # print('data shape: ', np.asarray(rgb_data_aug).shape, np.asarray(skel_data_aug).shape)

        # normalize joints and vectorize
        skel_data, spine = self.coord_normalize_torso(skel_data, frame_info)
        # skel_data_aug = coord_normalize(skel_data_aug, frame_info)
        skel_data = skel_data.reshape(frame_info[0], -1)

        # just for testing
        # if rgb_data.any():
        #     skel_data_aug = skel_data_aug.reshape(frame_info[0], 20, 3)
        #     skel_data_aug = coord_de_normalize_torso(skel_data_aug, spine, frame_info)
        #     # skel_data_aug = coord_de_normalize(skel_data_aug, frame_info)
        #     for frame, joint in zip(rgb_data_aug[:1], skel_data_aug[:1]):
        #
        #         # draw joints matplotlib
        #         plt.scatter(joint[:, 0], joint[:, 1], c='r')
        #         # plt.scatter(joint[18, 0], joint[18, 1], c='b')
        #         # plt.scatter(joint[1, 0], joint[1, 1], c='orange')
        #         # Display the resulting frame
        #         plt.imshow(frame)
        #         plt.show()
        # exit()

        return skel_data

    def sample_frames(self, frame_data):

        orig_size = len(frame_data)
        # target_size = self.params['time_size']
        target_shape = list(frame_data.shape[1:])
        target_shape.insert(0, self.params['time_size'])

        # print('target shape', target_shape)

        # if orig_size < target_size:
        #
        #     index = np.asarray(np.floor(np.linspace(0, orig_size, target_size+ 1)), dtype=int)[:-1]
        #     frame_data = frame_data[index]

        if orig_size < self.params['time_size']:

            new_frame = np.zeros(target_shape)
            new_frame[:frame_data.shape[0]] = frame_data
            frame_data = new_frame

        else:

            index = np.asarray(np.floor(np.linspace(0, orig_size-1, self.params['time_size'])), dtype=int)
            frame_data = frame_data[index]

        return np.asarray(frame_data)

    @staticmethod
    def get_skel_data(skeleton_file):

        # read skeleton file
        skeleton_file = open(skeleton_file, "r")

        # read number of frames
        no_of_frames = int(skeleton_file.readline().split(" ")[0])
        # print('no of frames skeleton: ', no_of_frames)

        # accumulate skeleton data
        skeleton_data = []

        for frame in range(no_of_frames):

            # read number of joints in a frame. Either 1 or 2 person
            no_of_joints = int(skeleton_file.readline())
            # print('in frame: {}, the no of joints: {}'.format(frame, no_of_joints))
            assert no_of_joints == 40 or no_of_joints == 80

            joints_per_frame = []

            # anyway ignore second skeleton. always go for 40 (20 skeleton)
            for joints in range(20):

                # read real world joints and ignore
                skeleton_file.readline()

                #  read  normalized joints world
                joint = skeleton_file.readline().split(" ")

                joints_per_frame.append([float(joint[0]), float(joint[1]), float(joint[2])])

                # read normalized joints and ignore
                # skeleton_file.readline()

            assert len(joints_per_frame) == 20
            skeleton_data.append(np.asarray(joints_per_frame, dtype=np.float32))

            # skip second skeleton
            if no_of_joints == 80:
                for skip in range(20):
                    skeleton_file.readline()
                    skeleton_file.readline()

        assert len(skeleton_data) == no_of_frames

        return np.asarray(skeleton_data), no_of_frames

    @staticmethod
    def get_rgb_data(rgb_file):

        # read video file
        # print('rgb_file: ', rgb_file)
        cap = cv.VideoCapture(rgb_file)

        frame_info = [int(cap.get(cv.CAP_PROP_FRAME_COUNT)),
                      int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
                      int(cap.get(cv.CAP_PROP_CHANNEL))]
        # print('number of frames, frame width, height: ', frame_info)

        frame_counter = 0
        rgb_data = []

        # Read until video is completed
        while cap.isOpened():

            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret:

                # Display the resulting frame
                # cv.imshow('Frame', frame)

                frame_counter += 1

                rgb_data.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

                # Press Q on keyboard to  exit
                # if cv.waitKey(25) & 0xFF == ord('q'):
                #     break

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        # cv.destroyAllWindows()

        # assert
        assert frame_info[0] == frame_counter

        return np.asarray(rgb_data), frame_info

    @staticmethod
    def get_depth_data(depth_file):

        # TODO: Depth processing

        depth_data = []

        return depth_data

    def process_all_data(self):

        # rand_data = np.random.randint(0, 320)
        for index, filename in tqdm(enumerate(glob.iglob(self.params['skeleton_raw_path'],
                                                         recursive=True))):

            # for testing
            # if index != rand_data:
            #     continue# if orig_size < target_size:

            rgb_file = filename.replace('skeleton.txt', 'rgb.avi')
            rgb_data, frame_info = self.get_rgb_data(rgb_file)
            pskeleton_data, no_of_skel_frames = self.get_skel_data(filename)

            # coord denormalize
            pskeleton_data = self.coord_de_normalize(pskeleton_data, frame_info)

            # sync frames by discarding last frame
            frame_info[0] = np.min([no_of_skel_frames, frame_info[0]])

            # # for testing
            # for frame, joint in zip(rgb_data, pskeleton_data):
            #
            #     # draw joints matplotlib
            #     plt.scatter(joint[0, 0], joint[0, 1])
            #
            #     # Display the resulting frame
            #     plt.imshow(frame)
            #     plt.show()
            #     exit()
            #
            # joint = 10
            # plt.plot(pskeleton_data[:, joint, 0], pskeleton_data[:, joint, 1])
            # plt.show()
            # # print('frame shape: ', frame.shape)
            # exit()

            # get bounding box
            bbox = self.calc_joints_bbox(frame_info, pskeleton_data)

            # crop to fit bounding box and discard last frame if frame numbers of rgb and skel
            # data is different
            rgb_data, pskeleton_data = self.crop_to_bbox(rgb_data[:frame_info[0]],
                                                         pskeleton_data[:frame_info[0]], bbox)

            # for testing
            # for frame, joint in zip(rgb_data, pskeleton_data):
            #
            #     # draw joints matplotlib
            #     plt.scatter(joint[:, 0], joint[:, 1])
            #     # Display the resulting frame
            #     plt.imshow(frame)
            #     plt.show()
            #     print('frame shape: ', frame.shape)
            #     exit()
            #
            # exit()

            # coord normalize
            # pskeleton_data = coord_normalize(pskeleton_data, frame_info)
            # pskeleton_data[:, :, 2] /= 50000

            # sample frames
            pskeleton_data = self.sample_frames(pskeleton_data)
            rgb_data = self.sample_frames(rgb_data)
            frame_info = rgb_data.shape

            rgb_data_reshaped = []
            for frame in rgb_data:
                rgb_data_reshaped.append(cv.resize(frame, (self.params['rgb_frame_size'], self.params['rgb_frame_size'])))
            rgb_data = rgb_data_reshaped
            del rgb_data_reshaped

            # for testing
            # for frame, joint in zip(rgb_data, pskeleton_data):
            #
            #     # draw joints matplotlib
            #     plt.scatter(joint[:, 0], joint[:, 1], c='r')
            #     # Display the resulting frame
            #     plt.imshow(frame)
            #     plt.show()
            #     print('frame shape: ', frame.shape)
            #     # exit()
            #
            # exit()

            # add meta data
            pskeleton_data = {'data': pskeleton_data, 'frame_info': frame_info}

            np.save(self.params['rgb_data'] + '/' + os.path.basename(rgb_file).split(".")[0], rgb_data)
            # np.save(self.params['skeleton_data'] + '/' + os.path.basename(filename).split(".")[0], pskeleton_data)

        print('total skeleton files processed: ', index + 1)

    @staticmethod
    def coord_de_normalize(joints_gt, frame_info):

        w, h = frame_info[1:3]

        # center_x, center_y = w // 2, h // 2

        joints_gt[:, :, 0] *= w
        joints_gt[:, :, 1] *= h

        # joints_gt += np.array([center_x, center_y])

        return joints_gt

    @staticmethod
    def coord_de_normalize_torso(joints_gt, spine, frame_info):

        w, h = frame_info[1:3]

        joints_gt[:, :, 0] *= w
        joints_gt[:, :, 1] *= h
        joints_gt[:, :, 2] *= 50000

        joints_gt += spine

        return joints_gt

    @staticmethod
    def coord_normalize(joints_gt, frame_info):

        w, h = frame_info[1:3]

        # center_x, center_y = w // 2, h // 2

        joints_gt[:, :, 0] /= w
        joints_gt[:, :, 1] /= h

        # joints_gt += np.array([center_x, center_y])

        return joints_gt

    @staticmethod
    def coord_normalize_torso(joints_gt, frame_info):

        w, h = frame_info[1:3]

        spine = joints_gt[:, 1, :]
        spine = np.expand_dims(spine, axis=1).repeat(20, axis=1)

        joints_gt -= spine

        joints_gt[:, :, 0] /= w
        joints_gt[:, :, 1] /= h
        joints_gt[:, :, 2] /= 50000

        return joints_gt, spine

    def normalize_rgb(self, rgb_data):
        """Preprocesses a Numpy array encoding a batch of images.

        # Arguments
            x: Input array, 3D or 4D.
            data_format: Data format of the image array.
            mode: One of "caffe", "tf" or "torch".
                - caffe: will convert the images from RGB to BGR,
                    then will zero-center each color channel with
                    respect to the ImageNet dataset,
                    without scaling.
                - tf: will scale pixels between -1 and 1,
                    sample-wise.
                - torch: will scale pixels between 0 and 1 and then
                    will normalize each channel with respect to the
                    ImageNet dataset.

        # Returns
            Preprocessed Numpy array.
        """

        # rgb_data = self.sample_frames(rgb_data)
        # print(rgb_data.shape)

        # rgb_data_reshaped = []
        # for frame in rgb_data:
        #     rgb_data_reshaped.append(cv.resize(frame, (self.params['rgb_frame_size'], self.params['rgb_frame_size'])))

        # for testing
        # for frame in rgb_data_reshaped:
        #     # draw joints matplotlib
        #     # plt.scatter(joint[:, 0], joint[:, 1], c='r')
        #     # Display the resulting frame
        #     plt.imshow(frame)
        #     plt.show()
        #     print('frame shape: ', frame.shape)
        #     # exit()

        # print(rgb_data[0, :, :, 0].shape)
        # exit()

        # rgb_data_reshaped = np.asarray(rgb_data_reshaped, dtype=np.float32) / 127.5
        # rgb_data_reshaped -= 1

        # rgb_data = np.asarray(rgb_data_reshaped, dtype=np.float32)
        rgb_data_reshaped = []

        # mean = np.mean(rgb_data, axis=(0, 1, 2))
        # for i in range(0, 3):
        #     rgb_data[:, :, :, i] = rgb_data[:, :, :, i] - mean[i]
        # print(rgb_data[0, :, :, 0])
        rgb_data = rgb_data / 127.5
        rgb_data = rgb_data - 1
        # print(rgb_data[0, :, :, 0])
        # exit()

        return rgb_data
        # return np.asarray(rgb_data, dtype=np.float32)
