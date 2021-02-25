# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
from nuscenes.nuscenes import NuScenes
import PIL.Image as pil
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
import skimage.transform
from pyquaternion import Quaternion
import numpy as np
import os
import os.path as osp
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from torch.nn.functional import max_pool2d

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class NuscDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_root
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_root,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 version='v1.0-mini',
                 sensor='CAM_FRONT',
                 is_train=False,
                 img_ext='.jpg'):
        super(NuscDataset, self).__init__()

        self.data_path = data_root
        self.data_path = '/share/nuscenes'
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.nusc = NuScenes(version=version, dataroot=self.data_path, verbose=True)
        self.sensor = sensor
        self.data_root = '/share/nuscenes'
        self.full_res_shape = (1600, 640)

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                # for nuscenes dataset we crop the image to resolution 1600x640
                for i in range(self.num_scales):

                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.nusc.sample)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        sample = self.nusc.sample[index]
        if sample['prev'] == '':
            sample = self.nusc.get('sample', token=sample['next'])
        elif sample['next'] == '':
            sample = self.nusc.get('sample', token=sample['prev'])
        elif sample['prev'] == '' and sample['next'] == '':
            raise FileNotFoundError('Can not find three consecutive samples')

        for i in self.frame_idxs:
            if i == "s":
                raise NotImplementedError('nuscenes dataset does not support stereo depth')
            else:
                inputs[("color", i, -1)] = self.get_color(sample, i, do_flip)
            #
            # for inp in inputs:
            #     if 'color' in inp:
            #         inputs[inp].save('/home/user/xwh/monodepth2-master/dataset_check/{}.png'.format(str(index)+'_'+str(i)))
        prev_sample = self.nusc.get('sample', token=sample['prev'])
        next_sample = self.nusc.get('sample', token=sample['next'])
        inputs[("pose_gt", -1, 0)] = self.get_poses(prev_sample, sample).astype(float)
        inputs[("pose_gt", 0, 1)] = self.get_poses(sample, next_sample).astype(float)
        inputs[("pose_gt", 0, -1)] = self.get_poses(sample, prev_sample).astype(float)
        #
        # inputs[("pose_gt", -1, 0)][[1,2],:] = inputs[("pose_gt", -1, 0)][[2,1],:]
        # inputs[("pose_gt", 0, 1)][[1,2],:] = inputs[("pose_gt", 0, 1)][[2,1],:]
        # inputs[("pose_gt", 0, -1)][[1,2],:] = inputs[("pose_gt", 0, -1)][[2,1],:]

        inputs[("pose_gt", -1, 0)][0,3]  = inputs[("pose_gt", -1, 0)][0,3]/1600
        inputs[("pose_gt", -1, 0)][1,3] = inputs[("pose_gt", -1, 0)][1, 3] /900
        inputs[("pose_gt", -1, 0)][2,3] = inputs[("pose_gt", -1, 0)][2, 3] /80

        inputs[("pose_gt", 0, 1)][0,3]  = inputs[("pose_gt", 0, 1)][0,3]/1600
        inputs[("pose_gt", 0, 1)][1,3] = inputs[("pose_gt", 0, 1)][1, 3] /900
        inputs[("pose_gt", 0, 1)][2,3] = inputs[("pose_gt", 0, 1)][2, 3] /80

        inputs[("pose_gt", 0, -1)][0,3]  = inputs[("pose_gt", 0, -1)][0,3]/1600
        inputs[("pose_gt", 0, -1)][1,3] = inputs[("pose_gt", 0, -1)][1, 3] /900
        inputs[("pose_gt", 0, -1)][2,3] = inputs[("pose_gt", 0, -1)][2, 3] /80

        # adjusting intrinsics to match each scale in the pyramid
        SENSOR_DATA = self.nusc.get('sample_data', sample['data'][self.sensor])
        sensor_calib_data = self.nusc.get('calibrated_sensor', SENSOR_DATA['calibrated_sensor_token'])
        K = np.zeros((4,4), dtype=np.float32)
        K[:,3] = [0,0,0,1]
        K[:3,:3] = np.array(sensor_calib_data['camera_intrinsic'])
        K[0,:] /= 1600
        K[1, :] /= 900
        self.K = K

        for scale in range(self.num_scales):
            K = self.K.copy()
            #  nusc camera instrinsic has not been normalized by orinigal image size
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            # print('-'*20)
            # print('Nusc ("K", {}) : '.format(scale), K)
            # print('-' * 20)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            LIDAR_RECORD = sample['data']['LIDAR_TOP']
            CAMERA_RECORD = sample['data'][self.sensor]
            points, depth, img = self.map_pointcloud_to_image(self.nusc,LIDAR_RECORD, CAMERA_RECORD, render_intensity=False)
            depth_gt = self.get_depth(self.nusc, sample, do_flip)
            # depth_gt[points[:, 1].astype(np.int), points[:, 0].astype(np.int)] = depth
            # TODO FINISH THE NUSCENES DEPTH MAP GENERATION
            # depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, sample, i, do_flip):
        if i == 0:
            color_path = self.nusc.get('sample_data', token=sample['data'][self.sensor])['filename']
            full_color_path = os.path.join(self.data_root, color_path)
            color = self.loader(full_color_path)
            # color = color.crop((0,2,1600,898))
            color = color.crop((0,240,1600,880))

        elif i == -1:
            prev_sample = self.nusc.get('sample', token=sample['prev'])
            color_path = self.nusc.get('sample_data', token=prev_sample['data'][self.sensor])['filename']
            full_color_path = os.path.join(self.data_root, color_path)
            color = self.loader(full_color_path)
            # color = color.crop((0,2,1600,898))
            color = color.crop((0,240,1600,880))

        if i == 1:
            next_sample = self.nusc.get('sample', token=sample['next'])
            color_path = self.nusc.get('sample_data', token=next_sample['data'][self.sensor])['filename']
            full_color_path = os.path.join(self.data_root, color_path)
            color = self.loader(full_color_path)
            # color = color.crop((0,2,1600,898))
            color = color.crop((0,240,1600,880))


        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def check_depth(self):
        return True

    def get_poses(self, current_sample, next_sample):

        current_sample_data = self.nusc.get('sample_data', current_sample['data'][self.sensor])
        next_sample_data = self.nusc.get('sample_data', next_sample['data'][self.sensor])

        EGO_current_token = current_sample_data['ego_pose_token']
        EGO_current = self.nusc.get('ego_pose',token=EGO_current_token)

        EGO_next_token = next_sample_data['ego_pose_token']
        EGO_next = self.nusc.get('ego_pose',token=EGO_next_token)

        # H_pose_current = np.zeros((4, 4))
        # H_pose_next = np.zeros((4, 4))
        # H_pose_current[3, 3] = 1
        # H_pose_next[3, 3] = 1
        # pose1_rot = Quaternion(EGO_current['rotation']).rotation_matrix
        # pose1_tran = EGO_current['translation']
        # H_pose_current[0:3, 0:3] = pose1_rot
        # H_pose_current[0:3, 3] = pose1_tran
        #
        # pose2_rot = Quaternion(EGO_next['rotation']).rotation_matrix
        # pose2_tran = EGO_next['translation']
        # H_pose_next[0:3, 0:3] = pose2_rot
        # H_pose_next[0:3, 3] = pose2_tran

        H_current_to_next = self.get_relative_pose(EGO_current, EGO_next)

        return H_current_to_next


    def get_depth(self, nusc, sample, do_flip):
        pointsensor_token = sample['data']['LIDAR_TOP']
        camsensor_token = sample['data'][self.sensor]
        pts, depth, img = self.map_pointcloud_to_image(nusc, pointsensor_token, camsensor_token)
        depth_gt = np.zeros((img.size[0], img.size[1]))
        pts_int = np.array(pts, dtype=int)
        depth_gt[pts_int[0,:], pts_int[1,:]] = depth

        # apply maxpool on nusc depth gt to make the lidar point denser
        # depth_gt = max_pool2d(torch.from_numpy(depth_gt).unsqueeze(dim=0), kernel_size=3, stride=1, padding=1)
        # depth_gt = depth_gt.squeeze(0).numpy()

        # we crop nuscenes depth_gt to match the orinigal color image shape
        # depth_gt = depth_gt[:,2:898]
        depth_gt = depth_gt[:,240:880]

        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        # depth_gt = skimage.transform.resize(
        #     depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
        return np.transpose(depth_gt, (1,0))
        # return depth_gt

    def get_relative_pose(self, pose1, pose2):
        """
        calculate relative from pose1 to pose2 in the global frame
        :param from_pose:
        :param to_pose:
        :return:
        """
        H_pose1 = np.zeros((4, 4))
        H_pose2 = np.zeros((4, 4))
        H_pose1[3, 3] = 1
        H_pose2[3, 3] = 1

        pose1_rot = Quaternion(pose1['rotation']).rotation_matrix
        pose1_tran = pose1['translation']
        H_pose1[0:3, 0:3] = pose1_rot
        H_pose1[0:3, 3] = pose1_tran

        pose2_rot = Quaternion(pose2['rotation']).rotation_matrix
        pose2_tran = pose2['translation']
        H_pose2[0:3, 0:3] = pose2_rot
        H_pose2[0:3, 3] = pose2_tran

        H_pose1_inv = np.linalg.inv(H_pose1)
        relative_pose_matrix = np.dot(H_pose1_inv, H_pose2)
        return relative_pose_matrix




    def map_pointcloud_to_image(self,
                                nusc,
                                pointsensor_token: str,
                                camera_token: str,
                                min_dist: float = 1.0,
                                render_intensity: bool = False,
                                show_lidarseg: bool = False):
        """
        Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
        plane.
        :param pointsensor_token: Lidar/radar sample_data token.
        :param camera_token: Camera sample_data token.
        :param min_dist: Distance from the camera below which points are discarded.
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :param show_lidarseg: Whether to render lidar intensity instead of point depth.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """

        cam = nusc.get('sample_data', camera_token)
        pointsensor = nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            # Ensure that lidar pointcloud is from a keyframe.
            assert pointsensor['is_key_frame'], \
                'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'
            pc = LidarPointCloud.from_file(pcl_path)
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(nusc.dataroot, cam['filename']))

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        if render_intensity:
            assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                              'not %s!' % pointsensor['sensor_modality']
            # Retrieve the color from the intensities.
            # Performs arbitary scaling to achieve more visually pleasing results.
            intensities = pc.points[3, :]
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            intensities = intensities ** 0.1
            intensities = np.maximum(0, intensities - 0.5)
            coloring = intensities

        else:
            # Retrieve the color from the depth.
            coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask]

        return points, coloring, im


if __name__ == '__main__':
    dataset = NuscDataset()