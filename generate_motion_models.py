"""Script to sample data and generate motion models."""

import argparse
from copy import deepcopy

import gtsam
import numpy as np

np.set_printoptions(linewidth=180, suppress=True, precision=8)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="The CSV data file")
    return parser.parse_args()


def subsample(data, every=5):
    """Subsample to the desired frequency. Use every `every`th sample."""
    return data[::every]


def get_foot_translation(foot: str, data: np.ndarray, header: list) -> gtsam.Point3:
    """Get the translation for the `foot` in the base frame."""
    tx = data[header.index(foot + "_tx")]
    ty = data[header.index(foot + "_ty")]
    tz = data[header.index(foot + "_tz")]
    return gtsam.Point3(tx, ty, tz)


def get_foot_rotation(foot: str, data: np.ndarray, header: list) -> gtsam.Rot3:
    """Get the rotation for the `foot` in the base frame."""
    rw = data[header.index(foot + "_rw")]
    rx = data[header.index(foot + "_rx")]
    ry = data[header.index(foot + "_ry")]
    rz = data[header.index(foot + "_rz")]
    return gtsam.Rot3.Quaternion(rw, rx, ry, rz)


def get_foot_pose(foot, data, header) -> gtsam.Pose3:
    """Get the pose for the `foot` in the base frame."""
    R = get_foot_rotation(foot, data, header)
    t = get_foot_translation(foot, data, header)
    return gtsam.Pose3(R, t)


def get_between_pose(foot, start, end, header):
    start_foot_pose = get_foot_pose(foot, start, header)
    # start_base_pose = get_base_pose(start, header)
    # start_foot_pose = body_to_world_frame(start_base_pose, start_foot_pose)
    end_foot_pose = get_foot_pose(foot, end, header)
    # end_base_pose = get_base_pose(end, header)
    # end_foot_pose = body_to_world_frame(end_base_pose, end_foot_pose)
    # end frame in the start frame
    # return start_foot_pose.logmap(end_foot_pose)
    between_vector = np.empty(6)
    between_vector[0:3] = start_foot_pose.rotation().logmap(
        end_foot_pose.rotation())
    between_vector[3:6] = end_foot_pose.translation() - \
        start_foot_pose.translation()
    return between_vector
    # return end_foot_pose.translation() - start_foot_pose.translation()


def get_base_pose(data, header):
    """Get the base pose in the world frame."""
    tx = data[header.index("tx")]
    ty = data[header.index("ty")]
    tz = data[header.index("tz")]
    rw = data[header.index("rw")]
    rx = data[header.index("rx")]
    ry = data[header.index("ry")]
    rz = data[header.index("rz")]
    R = gtsam.Rot3.Quaternion(rw, rx, ry, rz)
    t = gtsam.Point3(tx, ty, tz)
    pose = gtsam.Pose3(R, t)
    return pose


def body_to_world_frame(wTb, bTl):
    wTl = wTb.compose(bTl)
    return wTl


def sample_trajectories(data, header, feet, contacts):
    foot_trajectories = {
        contact_seq: np.empty((0, 6)) for contact_seq in contacts
    }

    contact_counts = {foot: np.zeros(len(contacts)) for foot in feet}
    feet_trajectories = {key: deepcopy(foot_trajectories) for key in feet}

    base_positions = np.empty((0, 6))

    for idx, (start, end) in enumerate(zip(data[:-1], data[1:])):
        # print(idx)
        start_base_pose = get_base_pose(start, header)
        end_base_pose = get_base_pose(end, header)
        base_between_pose = start_base_pose.logmap(end_base_pose)
        base_positions = np.vstack((base_positions, base_between_pose))

        for foot in feet:
            start_contact = int(start[header.index(foot + "_contact")])
            end_contact = int(end[header.index(foot + "_contact")])
            contact_sequence = "{0}{1}".format(start_contact, end_contact)
            between_pose_vector = get_between_pose(foot, start, end, header)
            feet_trajectories[foot][contact_sequence] = np.vstack(
                (feet_trajectories[foot][contact_sequence], between_pose_vector))
            contact_counts[foot][contacts.index(contact_sequence)] += 1

    for foot, counts in contact_counts.items():
        counts = counts.reshape((2, 2))
        counts = counts / counts.sum(axis=1)[:, None]
        print(np.linalg.eig(counts))
        print(foot, "\n", counts)
        print("\n\n")

    return base_positions, feet_trajectories


def main():
    """Main runner."""
    args = parse_arguments()
    # load the CSV data
    header_row = np.loadtxt(args.filename, dtype=str, max_rows=1)
    header = str(header_row).split(",")

    data = np.loadtxt(args.filename, skiprows=1, delimiter=",")

    # 500 is the original frequency, 100 is the desired frequency
    # so we skip every 500//100 = 5 samples
    # data = subsample(data, 500//100)
    data = subsample(data, 10)  # 50 Hz

    feet = ("FR", "FL", "RR", "RL")
    contacts = ("00", "01", "10", "11")

    base_trajectories, feet_trajectories = sample_trajectories(
        data, header, feet=feet, contacts=contacts)

    print("base mean\n",
          gtsam.Pose3.Expmap(np.mean(base_trajectories, axis=0)))
    print("base stddev\n",
          gtsam.Pose3.Expmap(np.std(base_trajectories, axis=0)))
    print("\n\n")
    for foot in feet:
        for contact in contacts:
            contact_samples = feet_trajectories[foot][contact]
            # print(contact_samples)
            print(foot, contact)
            mu = np.mean(contact_samples, axis=0)
            tau = np.std(contact_samples, axis=0)
            cov = np.cov(contact_samples.T)
            print("mean\n", gtsam.Rot3.Expmap(mu[0:3]), mu[3:6])
            print("stddev\n", gtsam.Rot3.Expmap(tau[0:3]), tau[3:6])
            # print("covariance matrix\n", cov)
            print("\n\n")


if __name__ == "__main__":
    main()
