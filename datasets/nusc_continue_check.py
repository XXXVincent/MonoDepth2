from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from pyquaternion import Quaternion
import numpy as np
import os
import shutil
import os.path as osp
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

dst_folder = '/home/user/xwh/nusc_continue_check'
nusc = NuScenes(version='v1.0-mini', dataroot='/share/nuscenes', verbose=True)
nusc_scenes = nusc.list_scenes()

sample = nusc.sample[388]
ind = 0
while sample['next'] != '':
    color_path = nusc.get('sample_data', token=sample['data']['CAM_FRONT'])['filename']
    full_color_path = os.path.join('/share/nuscenes', color_path)
    shutil.copyfile(full_color_path, os.path.join(dst_folder, str(ind).zfill(6)+'.jpg'))
    sample = nusc.get('sample', token=sample['next'])
    ind += 1
