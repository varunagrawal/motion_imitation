"""
Generate data for the A1 robot.
An IMU sensor is provided on the robot.
"""
import inspect
import os
import time
from datetime import datetime

import numpy as np
import pybullet  # pytype:disable=import-error
import pybullet_data
import scipy.interpolate
from absl import app, flags, logging
from motion_imitation.envs.sensors import robot_sensors, sensor
from motion_imitation.robots import a1, robot_config
from motion_imitation.robots.gamepad import gamepad_reader
from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import (locomotion_controller, openloop_gait_generator,
                            raibert_swing_leg_controller)
from mpc_controller import \
    torque_stance_leg_controller_quadprog as torque_stance_leg_controller
from pybullet_utils import bullet_client

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

#from mpc_controller import torque_stance_leg_controller
#import mpc_osqp

flags.DEFINE_string("logdir", "logs", "where to log trajectories.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 5., "maximum time to run the robot.")
flags.DEFINE_string("video_name", "a1_walking",
                    "name of video file (without extension)")
flags.DEFINE_string("log_file", "sample_log", "name of the log file")
flags.DEFINE_string(
    "trajectory", "straight",
    "Trajectory option, choose from {standing, straight, diagonal, square}")
FLAGS = flags.FLAGS

_NUM_SIMULATION_ITERATION_STEPS = 300
_MAX_TIME_SECONDS = 30.

_STANCE_DURATION_SECONDS = [
    0.3
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).

# Standing
# _DUTY_FACTOR = [1.] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0., 0., 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
# )

# Tripod
# _DUTY_FACTOR = [.8] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
# )

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


def standing(vx=0, vy=0, wz=0):
    """The robot is standing in place"""
    time_points = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    speed_points = (
        # Walk forward
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        # Pause
        (0, 0, 0, 0),
        # Walk forward
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        # Pause again
        (0, 0, 0, 0),
        # Walk again
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        # and relax
        (0, 0, 0, 0),
        (0, 0, 0, 0))
    return time_points, speed_points


def straight_line(vx=1.0, vy=0.2, wz=1.6):
    """Generate a simple straight line trajectory"""
    time_points = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    speed_points = (
        # Walk forward
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        # Pause
        (0, 0, 0, 0),
        # Walk forward
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        # Pause again
        (0, 0, 0, 0),
        # Walk again
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        # and relax
        (0, 0, 0, 0),
        (0, 0, 0, 0))
    return time_points, speed_points


def diagonal_line(vx=1.0, vy=1.0, wz=1.6):
    """Generate a simple straight line trajectory"""
    time_points = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    speed_points = (
        # Walk forward
        (vx, vy, 0, 0),
        (vx, vy, 0, 0),
        (vx, vy, 0, 0),
        (vx, vy, 0, 0),
        (vx, vy, 0, 0),
        (vx, vy, 0, 0),
        # Pause
        (0, 0, 0, 0),
        # Walk forward
        (vx, vy, 0, 0),
        (vx, vy, 0, 0),
        (vx, vy, 0, 0),
        # Pause again
        (0, 0, 0, 0),
        # Walk again
        (vx, vy, 0, 0),
        (vx, vy, 0, 0),
        (vx, vy, 0, 0),
        # and relax
        (0, 0, 0, 0),
        (0, 0, 0, 0))
    return time_points, speed_points


def square(vx=1.0, vy=0, wz=1.6):
    """
    Generate square trajectory, starting and ending at origin.
    """

    time_points = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    speed_points = (
        # Get set
        (0, 0, 0, 0),
        # Walk forward and then turn left
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (0, 0, 0, wz),
        (0, 0, 0, wz),
        # Walk forward and then turn left
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (0, 0, 0, wz),
        (0, 0, 0, wz),
        # Walk forward and then turn left
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),
        (0, 0, 0, wz),
        (0, 0, 0, wz),
        (vx, 0, 0, 0),
        (vx, 0, 0, 0),  # Walk to start point
        (0, 0, 0, 0))  # and relax.

    return time_points, speed_points


def trajectory_function(t, option="straight"):
    """Creates a speed profile based on time t to generate the trajectory."""

    # time_points = (0, 5, 10, 15, 20, 25, 30)
    # speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz),
    #                 (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

    # standing, straight, diagonal, square
    if option == "standing":
        time_points, speed_points = standing()
    elif option == "straight":
        time_points, speed_points = straight_line()
    elif option == "square":
        time_points, speed_points = square()
    elif option == "diagonal":
        time_points, speed_points = diagonal_line(
        )  # causes robot to flip over

    speed = scipy.interpolate.interp1d(time_points,
                                       speed_points,
                                       kind="previous",
                                       fill_value="extrapolate",
                                       axis=0)(t)

    return speed[0:3], speed[3], False


def _setup_controller(robot):
    """Demonstrates how to create a locomotion controller."""
    desired_speed = (0, 0)
    desired_twisting_speed = 0

    gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
        robot,
        stance_duration=_STANCE_DURATION_SECONDS,
        duty_factor=_DUTY_FACTOR,
        initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
        initial_leg_state=_INIT_LEG_STATE)
    window_size = 20
    state_estimator = com_velocity_estimator.COMVelocityEstimator(
        robot, window_size=window_size)
    sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_height=robot.MPC_BODY_HEIGHT,
        foot_clearance=0.01)

    st_controller = torque_stance_leg_controller.TorqueStanceLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_body_height=robot.MPC_BODY_HEIGHT
        # ,qp_solver = mpc_osqp.QPOASES #or mpc_osqp.OSQP
    )

    controller = locomotion_controller.LocomotionController(
        robot=robot,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        swing_leg_controller=sw_controller,
        stance_leg_controller=st_controller,
        clock=robot.GetTimeSinceReset)
    return controller


def _update_controller_params(controller, lin_speed, ang_speed):
    controller.swing_leg_controller.desired_speed = lin_speed
    controller.swing_leg_controller.desired_twisting_speed = ang_speed
    controller.stance_leg_controller.desired_speed = lin_speed
    controller.stance_leg_controller.desired_twisting_speed = ang_speed


def main(argv):
    """Runs the locomotion controller example."""
    del argv  # unused

    # Construct simulator
    if FLAGS.show_gui:
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    p.setPhysicsEngineParameter(numSolverIterations=30)

    TIMESTEP = 0.001
    p.setTimeStep(TIMESTEP)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, FLAGS.video_name + ".mp4")

    sensors = [
        robot_sensors.IMUSensor(),
        robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS,
                                       noisy_reading=True),
        robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS,
                                       noisy_reading=False,
                                       name="TrueMotorAngles"),
    ]
    # Construct robot class:
    robot = a1.A1(p,
                  sensors=sensors,
                  motor_control_mode=robot_config.MotorControlMode.HYBRID,
                  enable_action_interpolation=False,
                  reset_time=2,
                  time_step=TIMESTEP,
                  action_repeat=5)

    controller = _setup_controller(robot)

    controller.reset()

    command_function = trajectory_function

    if FLAGS.logdir:
        if FLAGS.log_file:
            log_file = FLAGS.log_file
        else:
            log_file = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        logdir = os.path.join(FLAGS.logdir, log_file)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    start_time = robot.GetTimeSinceReset()
    current_time = start_time

    timesteps = []
    base_position, base_rotation, base_vels, actions = [], [], [], []
    omega_b, joint_angles, true_joint_angles = [], [], []
    foot_positions_base, foot_orientations_base = [], []
    foot_positions_world, foot_orientations_world = [], []
    foot_contacts = []

    for motor_name, motor_id in robot._joint_name_to_id.items():
        if motor_id in robot._motor_id_list:
            print(motor_id, motor_name)

    while current_time - start_time < FLAGS.max_time_secs:
        # time.sleep(0.0008) #on some fast computer, works better with sleep on real A1?
        start_time_robot = current_time
        start_time_wall = time.time()
        # Updates the controller behavior parameters.
        lin_speed, ang_speed, e_stop = command_function(
            current_time, FLAGS.trajectory)

        if e_stop:
            logging.info("E-stop kicked, exiting...")
            break

        _update_controller_params(controller, lin_speed, ang_speed)
        controller.update()

        hybrid_action, _ = controller.get_action()

        timesteps.append(current_time)
        base_position.append(np.array(robot.GetBasePosition()))
        # each orientation is a quaternion
        base_rotation.append(np.array(robot.GetBaseOrientation()))
        base_vels.append(np.array(robot.GetBaseVelocity()))
        omega_b.append(np.array(robot.GetTrueBaseRollPitchYawRate()))
        joint_angles.append(np.array(robot.GetMotorAngles()))

        # Foot positions are with respect to the base frame
        foot_position, foot_orientation = \
            robot.GetFootPositionsAndOrientationsInBaseFrame()
        foot_positions_base.append(foot_position)
        foot_orientations_base.append(foot_orientation)

        foot_position, foot_orientation = \
            robot.GetFootPositionsAndOrientationsInWorldFrame()
        foot_positions_world.append(foot_position)
        foot_orientations_world.append(foot_orientation)

        foot_contacts.append(np.array(robot.GetFootContacts()))

        # true_joint_angles.append(
        #     np.array(robot.GetAllSensors()[2].get_observation()))

        actions.append(hybrid_action)
        robot.Step(hybrid_action)
        current_time = robot.GetTimeSinceReset()

        expected_duration = current_time - start_time_robot
        actual_duration = time.time() - start_time_wall
        if actual_duration < expected_duration:
            time.sleep(expected_duration - actual_duration)
        print("actual_duration =", actual_duration)

    if FLAGS.logdir:
        np.savez(os.path.join(logdir, 'action.npz'),
                 action=actions,
                 timesteps=timesteps,
                 base_position=base_position,
                 base_rotation=base_rotation,
                 base_vels=base_vels,
                 omega_b=omega_b,
                 joint_angles=joint_angles,
                 true_joint_angles=true_joint_angles,
                 foot_positions_base=foot_positions_base,
                 foot_orientations_base=foot_orientations_base,
                 foot_positions_world=foot_positions_world,
                 foot_orientations_world=foot_orientations_world,
                 foot_contacts=foot_contacts)
        logging.info("========= logged {0} to: {1}".format(
            len(timesteps), logdir))


if __name__ == "__main__":
    app.run(main)
