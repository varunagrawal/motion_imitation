"""Script to parse output from simulator into an easy to read CSV file."""

import argparse

import numpy as np


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="npz file with all the data")
    parser.add_argument("output",
                        default="a1_data.csv",
                        help="CSV filename to which to store the parse data")
    parser.add_argument("dt", type=float, help="Timestep for each measurement")
    return parser.parse_args()


def convert(datafile, output_file, dt):
    """Parse the .npz datafile and generate a CSV file from it."""
    data = np.load(datafile)
    base_position = data["base_position"]
    base_rotation = data["base_rotation"]
    base_vels = data["base_vels"]
    omega_b = data["omega_b"]
    joint_angles = data["joint_angles"]
    timesteps = data["timesteps"]

    # true_joint_angles = data["true_joint_angles"]
    foot_positions = data["foot_positions_base"]
    foot_orientations = data["foot_orientations_base"]
    foot_contacts = data["foot_contacts"]

    # these are specific to the A1 robot
    joint_names = [
        'FR_hip_joint', 'FR_upper_joint', 'FR_lower_joint', 'FL_hip_joint',
        'FL_upper_joint', 'FL_lower_joint', 'RR_hip_joint', 'RR_upper_joint',
        'RR_lower_joint', 'RL_hip_joint', 'RL_upper_joint', 'RL_lower_joint'
    ]

    feet = ["FR_lower", "FL_lower", "RR_lower", "RL_lower"]

    # Move the w value of the rotation quaternion to the first position.
    old_idxes, new_idxes = [0, 1, 2, 3], [3, 0, 1, 2]
    base_rotation[:, old_idxes] = base_rotation[:, new_idxes]
    foot_orientations[:, :, old_idxes] = foot_orientations[:, :, new_idxes]

    with open(output_file, "w") as file:
        # write the header
        file.write("time,tx,ty,tz,rw,rx,ry,rz,vx,vy,vz,wx,wy,wz,ax,ay,az,")
        for foot in feet:
            for d in ["x", "y", "z"]:
                file.write(foot + "_t" + d + ",")
        for foot in feet:
            for d in ["w", "x", "y", "z"]:
                file.write(foot + "_r" + d + ",")
        for foot in feet:
            file.write(foot + "_contact,")
        file.write(",".join(joint_names))

        file.write("\n")

        # Compute the acceleration via numerical derivative in the world/navigation frame
        for idx, _ in enumerate(base_position):
            if idx == 0:
                acc_n = np.zeros(3)
            else:
                acc_n = (base_vels[idx] - base_vels[idx - 1]) / dt

            # join the data as a single row
            datum = np.hstack(
                (timesteps[idx], base_position[idx], base_rotation[idx],
                 base_vels[idx], omega_b[idx], acc_n,
                 foot_positions[idx].reshape(12),
                 foot_orientations[idx].reshape(16), foot_contacts[idx],
                 joint_angles[idx])).tolist()
            # Write the data
            row = ",".join([str(x) for x in datum])
            file.write(row + "\n")


def main():
    """Main function."""
    args = parse_args()
    convert(args.data, args.output, args.dt)


if __name__ == "__main__":
    main()
