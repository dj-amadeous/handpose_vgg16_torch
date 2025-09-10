import torch
import torchvision
from torch.utils.data import Dataset
import glob
import numpy as np
from scipy.io import loadmat
import cv2




class HandposeDataset(Dataset):
    def __init__(self, root_path, joint_list, dataset='nyu', mode='train',
                 batch_size=32, image_size=(224,224), shuffle=True):
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.joint_list = joint_list
        if self.dataset == 'nyu':
            train_dataset_path = root_path + '/train'
            test_dataset_path = root_path + '/test'
            labels_train = loadmat(train_dataset_path + '/joint_data.mat')
            labels_test = loadmat(test_dataset_path + '/joint_data.mat')
            if mode == 'train':
                self.image_paths = glob.glob(train_dataset_path + '/depth_1_*.png')
                # get the array with the requested joint list for the nyu train dataset
                # it is convoluted because of the way that nyu labels are stored in mat files
                self.labels = np.array(labels_train['joint_uvd'][0][:, joint_list, :])
            elif mode == 'test':
                self.image_paths = glob.glob(test_dataset_path + '/depth_1_*.png')
                self.labels = np.array(labels_test['joint_uvd'][0][:, joint_list, :])
            else:
                print("error in mode of dataset")
                exit()
            self.n_samples = len(self.image_paths)
            # order the image paths if they're nyu so that the array indices match the labels
            self.image_paths.sort(key=lambda st: int(st.split('_')[-1].split('.')[0]))
        else:
            print("Dataset not yet implemented")
            exit()

    def keypoint_crop(self, kp):
        """
        THIS FUNCTION IS NOT USED, POTENTIAL FUTURE USAGE
        get coord of bounding cube parallel to axes in real-world space

        :Parameter
        ----------
        kp: numpy array of the keypoints of the image

        """
        x_min, y_min, z_min = np.min(kp, axis=0)
        x_max, y_max, z_max = np.max(kp, axis=0)
        vertices = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ])
        return vertices

    def expand_to_cube(self, vertices):
        """
        THIS FUNCTION IS NOT USED, POTENTIAL FUTURE USAGE
        the vertices are as defined in the bounding_points function,
         alternating min max on Z, min min max max on Y, and min min min min max max ... on X

        :param vertices: numpy array of the vertices of the bounding box associated with image
        :return: new bounding box that is in the shape of a cube, but still contains the same image.
        """
        x_length = np.abs(vertices[4][0] - vertices[0][0])
        y_length = np.abs(vertices[2][1] - vertices[0][1])
        z_length = np.abs(vertices[1][2] - vertices[0][2])

        max_length = max(x_length, y_length, z_length)

        center = np.mean(vertices, axis=0)

        half_length = max_length / 2

        cube_vertices = np.array([
            [center[0] - half_length, center[1] - half_length, center[2] - half_length],
            [center[0] - half_length, center[1] - half_length, center[2] + half_length],
            [center[0] - half_length, center[1] + half_length, center[2] - half_length],
            [center[0] - half_length, center[1] + half_length, center[2] + half_length],
            [center[0] + half_length, center[1] - half_length, center[2] - half_length],
            [center[0] + half_length, center[1] - half_length, center[2] + half_length],
            [center[0] + half_length, center[1] + half_length, center[2] - half_length],
            [center[0] + half_length, center[1] + half_length, center[2] + half_length],
        ])

        return cube_vertices

    def uvd_to_xyz(self, img):
        '''
        THIS FUNCTION IS NOT USED, POTENTIAL FUTURE USAGE
        this function takes specific points and converts them to xyz. It CANNOT convert a depth map to UVD
        to convert a depth map to uvd, you must first turn the depth map into a voxel grid.
        if you had a voxel grid then you would end up with a 4D output (which this function does not support)
        of x,y,z and [1,0] depending on if there is an object in the space or not.
        :param img:
        :return:
        '''
        xRes = 640
        yRes = 480
        xzFactor = 1.08836710
        yzFactor = 0.817612648
        normalized_x = np.array((img[:, :, 0] / xRes) - 0.5, dtype=np.float32)
        normalized_y = np.array(0.5 - (img[:, :, 1] / yRes), dtype=np.float32)
        xyz = np.zeros(img.shape, dtype=np.float32)
        xyz[:, :, 2] = img[:, :2]
        xyz[:, :, 0] = normalized_x * xyz[:, :, 2] * xzFactor
        xyz[:, :, 1] = normalized_y * xyz[:, :, 2] * yzFactor

    def __getitem__(self, item):
        img = cv2.imread(self.image_paths[item])
        if self.dataset == 'nyu':
            img = np.asarray(img)
            # get depth image into proper 16 bit format
            lower_channel = np.array(img[:,:,0], np.uint16)
            upper_channel = np.array(img[:,:,1], np.uint16)
            depth_img = lower_channel + np.left_shift(upper_channel, 8)
            # crop ground truth, and bilinear interpolate to 225x225, +/- 10 for margin of error
            min_u = np.floor(np.min(self.labels[item, :,0])).astype(np.int32) - 10
            max_u = np.ceil(np.max(self.labels[item, :,0])).astype(np.int32) + 10
            min_v = np.floor(np.min(self.labels[item, :,1])).astype(np.int32) - 10
            max_v = np.ceil(np.max(self.labels[item, :,1])).astype(np.int32) + 10
            cropped = np.array(depth_img[min_u:max_u + 1, min_v:max_v + 1])
            vgg_input = cv2.resize(cropped, self.image_size, interpolation=cv2.INTER_LINEAR)
            max_depth = np.max(vgg_input)
            # normalize to floats between 0-1
            vgg_input = np.array(vgg_input/max_depth, np.float32)
            vgg_input = np.stack([vgg_input]*3, axis=0)
            output = self.labels[item].flatten()

            return torch.from_numpy(vgg_input).float(), torch.from_numpy(output).float()


    def __len__(self):
        return self.n_samples