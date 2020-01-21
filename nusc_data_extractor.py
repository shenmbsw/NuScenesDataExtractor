import os.path as osp

import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points


class NuScenesDataExtractor:
    """ Helper class to extract raw data from the database """

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        self.radar_sensor_channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT',
                                      'RADAR_BACK_RIGHT']
        self.camera_sensor_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
                                       'CAM_BACK_RIGHT', 'CAM_BACK']
        self.scene = None
        self.scene_name = None
        self.sample = None
        self.sample_token = None
        print(" [*] NuScenesDataExtractor initiated.")

    def set_scene(self, scene):
        """ Selecting a scene """
        self.scene_name = scene
        scene_token = self.nusc.field2token('scene', 'name', self.scene_name)[0]
        self.scene = self.nusc.get('scene', scene_token)
        self.sample_token = self.scene['first_sample_token']
        self.sample = self.nusc.get('sample', self.sample_token)

    def advance(self):
        """
        Advances one time step in the scene.
        :param n: number of timesteps to advance
        """
        if not hasattr(self, 'sample'):
            raise ValueError(" [!] Can't advance, if no scene was selected before.")
        if self.sample['next'] == '':
            return False
        self.sample = self.nusc.get('sample', self.sample['next'])
        self.sample_token = self.sample['token']
        return True

    def get_timestamp(self):
        """
        Get the timestamp for current sample
        :return timestamp: an int respect timestamp in microsecond
        """
        return self.sample['timestamp']

    def get_camera_image(self, channel='CAM_FRONT'):
        """
        Extracting camera image for current timestep in current scene for specified camera channel.
        :param channel: Camera channel selection. Front camera as default
        :return camera image as PIL image
        """

        # Check for correct camera channel selection
        assert channel in self.camera_sensor_channels, " [!] Camera channel \"{}\" not found.".format(channel)

        # Select sensor data record
        sample_data_token = self.sample['data'][channel]
        sd_record = self.nusc.get('sample_data', sample_data_token)
        filename = osp.join(self.nusc.dataroot, sd_record['filename'])
        image = Image.open(filename)

        print(" [*] Camera image extracted.")
        return image

    def get_each_sweep_lidar_pointcloud(self, nsweeps=5):
        """
        Extracting lidar pointcloud for current timestep in current scene
        :return
            Point cloud [[Position(x,y,z, a) X n_points] * timestamps]
            Time stamps a list of int for timestamps in microsecond
        """

        # Select sensor data record
        channel = 'LIDAR_TOP'
        sample_data_token = self.sample['data'][channel]
        sd_record = self.nusc.get('sample_data', sample_data_token)
        # Get aggregated point cloud in lidar frame.
        chan = sd_record['channel']
        pc, times = LidarPointCloud.from_file_multisweep(self.nusc, self.sample, chan, channel, nsweeps)
        times_interval = np.unique(times)[::-1]
        unique_times = self.get_timestamp() - times_interval.astype(np.int)
        points = []
        for t in times_interval:
            points.append(pc.points[:, times[0] == t])
        return points, unique_times

    def get_lidar_pointcloud(self, nsweeps=5):
        """
        Extracting lidar pointcloud for current timestep in current scene
        :return Point cloud [Position(x,y,z,i,t) X n_points]
        """

        # Select sensor data record
        channel = 'LIDAR_TOP'
        sample_data_token = self.sample['data'][channel]
        sd_record = self.nusc.get('sample_data', sample_data_token)
        # Get aggregated point cloud in lidar frame.
        chan = sd_record['channel']
        pc, times = LidarPointCloud.from_file_multisweep(self.nusc, self.sample, chan, channel, nsweeps)
        points = np.concatenate([pc.points, times], axis=0)
        return points

    def get_all_radar_pointclouds(self):
        """
        Extracting radar detection pointclouds with velocities for current timestep in current scene of all radar sensors.
        :return (Point cloud [n_radars, Position(x,y,z), n_points], Point cloud [n_radars, Velocity(x,y,z), n_points])
        """

        points = []
        points_vel = []

        for channel in self.radar_sensor_channels:
            p, v = self.get_radar_pointcloud(channel=channel)
            points = [points, p]
            points_vel = [points_vel, v]

        print(" [*] All radar point clouds extracted.")
        return points, points_vel

    def get_radar_pointcloud(self, channel='RADAR_FRONT'):
        """
        Extracting radar detection pointcloud with velocities for current timestep in current scene for specified radar channel.
        :param channel: Radar channel selection. Front radar as default
        :return (Point cloud [Position(x,y,z) X n_points], Point cloud [Velocity(x,y,z) X n_points])
        """

        # Check for correct radar channel selection
        assert channel in self.radar_sensor_channels, " [!] Radar channel \"{}\" not found.".format(channel)

        # Select sensor data record
        sample_data_token = self.sample['data'][channel]
        sd_record = self.nusc.get('sample_data', sample_data_token)
        lidar_token = self.sample['data']['LIDAR_TOP']

        # The point cloud is transformed to the lidar frame for visualization purposes.
        ref_chan = 'LIDAR_TOP'
        pc, times = RadarPointCloud.from_file_multisweep(self.nusc, self.sample, channel, ref_chan, nsweeps=1)

        # Transform radar velocities (x is front, y is left), as these are not transformed when loading the point
        # cloud.
        radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        lidar_sd_record = self.nusc.get('sample_data', lidar_token)
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        velocities = pc.points[8:10, :]  # Compensated velocity
        velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
        velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
        velocities = np.dot(Quaternion(lidar_cs_record['rotation']).rotation_matrix.T, velocities)
        velocities[2, :] = np.zeros(pc.points.shape[1])

        # Show point cloud.
        points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
        points_vel = view_points(pc.points[:3, :] + velocities, np.eye(4), normalize=False)

        print(" [*] Radar point cloud extracted from channel \"" + channel + "\". Shape: " + str(points.shape))
        return points, points_vel

