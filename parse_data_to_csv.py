"""Script to parse output from simulator into an easy to read CSV file."""

import argparse

import numpy as np


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="npz file with all the data")
    parser.add_argument("output", default="a1_data.csv",
                        help="CSV filename to which to store the parse data")
    return parser.parse_args()


def convert(datafile, output_file):
    """Parse the .npz datafile and generate a CSV file from it."""
    data = np.load(datafile)
    base_position = data["base_position"]
    base_rotation = data["base_rotation"]
    base_vels = data["base_vels"]
    imu_rates = data["imu_rates"]
    joint_angles = data["joint_angles"]

    # true_joint_angles = data["true_joint_angles"]
    foot_positions = data["foot_positions"]
    foot_orientations = data["foot_orientations"]
    foot_contacts = data["foot_contacts"]

    # these are specific to the A1 robot
    joint_names = ['FR_hip_joint', 'FR_upper_joint', 'FR_lower_joint',
                   'FL_hip_joint', 'FL_upper_joint', 'FL_lower_joint',
                   'RR_hip_joint', 'RR_upper_joint', 'RR_lower_joint',
                   'RL_hip_joint', 'RL_upper_joint', 'RL_lower_joint'
                   ]

    feet = ["FR", "FL", "RR", "RL"]

    # Move the w value of the rotation quaternion to the first position.
    old_idxes, new_idxes = [0, 1, 2, 3], [3, 0, 1, 2]
    base_rotation[:, old_idxes] = base_rotation[:, new_idxes]
    foot_orientations[:, :, old_idxes] = foot_orientations[:, :, new_idxes]

    with open(output_file, "w") as file:
        # write the header
        file.write("tx,ty,tz,rw,rx,ry,rz,vx,vy,vz,wx,wy,wz,ax,ay,az,")
        for foot in feet:
            for d in ["x", "y", "z"]:
                file.write(foot + "_t" + d + ",")
        for foot in feet:
            for d in ["w", "x", "y", "z"]:
                file.write(foot + "_r" + d + ",")
        for foot in feet:
            file.write(foot + "contact,")
        file.write(",".join(joint_names))

        file.write("\n")

        dt = 0.002  # we get 500 readings per second
        for idx in range(len(base_position)):
            # Compute the acceleration via numerical derivative
            if idx == 0:
                acc = np.zeros(3)
            else:
                acc = (base_vels[idx] - base_vels[idx-1])/dt

            # join the data as a single row
            datum = np.hstack((base_position[idx], base_rotation[idx],
                               base_vels[idx], imu_rates[idx], acc,
                               foot_positions[idx].reshape(12),
                               foot_orientations[idx].reshape(16),
                               foot_contacts[idx],
                               joint_angles[idx])).tolist()
            # Write the data
            row = ",".join([str(x) for x in datum])
            file.write(row + "\n")


def main():
    args = parse_args()
    convert(args.data, args.output)


if __name__ == "__main__":
    main()