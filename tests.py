"""Tests for generate_motion_models.py"""

import gtsam
import numpy as np

from generate_motion_models import *


class TestGenerateMotionModels:
    @classmethod
    def setup_class(cls):
        filename = "a1_walking_straight.csv"
        header_row = np.loadtxt(filename, dtype=str, max_rows=1)
        cls.header = str(header_row).split(",")
        cls.data = np.loadtxt(filename, skiprows=1, delimiter=",")
        cls.foot = "FR"

    def test_get_foot_rotation(self):
        R = get_foot_rotation(self.foot, self.data[0], self.header)
        true_rotation = gtsam.Rot3.Quaternion(
            0.898496150970459, 7.798700971761718e-05, -0.438981294631958, -3.8177138776518404e-05)
        assert(np.allclose(R.matrix(), true_rotation.matrix()))

    def test_get_foot_translation(self):
        true_translation = gtsam.Point3(
            0.16839776933193207, -0.13419416546821594, -0.24395912885665894)
        t = get_foot_translation(self.foot, self.data[0], self.header)

        assert(np.allclose(true_translation, t))

    def test_get_foot_pose(self):
        true_rotation = gtsam.Rot3.Quaternion(
            0.898496150970459, 7.798700971761718e-05, -0.438981294631958, -3.8177138776518404e-05)
        true_translation = gtsam.Point3(
            0.16839776933193207, -0.13419416546821594, -0.24395912885665894)
        true_pose = gtsam.Pose3(true_rotation, true_translation)

        pose = get_foot_pose(self.foot, self.data[0], self.header)

        assert(np.allclose(true_pose.matrix(), pose.matrix()))

    def test_get_between_pose(self):
        between_vector = get_between_pose(self.foot, self.data[0],
                                          self.data[1], self.header)

        start_pose = get_foot_pose(self.foot, self.data[0], self.header)
        end_pose = get_foot_pose(self.foot, self.data[1], self.header)

        true_between_vector = start_pose.logmap(end_pose)

        assert(np.allclose(true_between_vector, between_vector))
