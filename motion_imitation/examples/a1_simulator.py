"""
Simulate the A1 robot.
To move the robot, you need to use a gamepad controller.
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

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


#from mpc_controller import torque_stance_leg_controller
#import mpc_osqp


flags.DEFINE_string("logdir", "logs", "where to log trajectories.")
flags.DEFINE_bool("use_gamepad", True,
                  "whether to use gamepad to provide control input.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 5., "maximum time to run the robot.")
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


def _generate_example_linear_angular_speed(t):
    """Creates an example speed profile based on time for demo purpose."""
    vx = 0.6
    vy = 0.2
    wz = 0.8

    time_points = (0, 5, 10, 15, 20, 25, 30)
    speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz),
                    (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

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
    p.setTimeStep(0.001)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    sensors = [
        robot_sensors.IMUSensor(),
        robot_sensors.MotorAngleSensor(
            num_motors=a1.NUM_MOTORS, noisy_reading=True),
        robot_sensors.MotorAngleSensor(
            num_motors=a1.NUM_MOTORS, noisy_reading=False, name="TrueMotorAngles"),

    ]
    # Construct robot class:
    robot = a1.A1(p,
                  sensors=sensors,
                  motor_control_mode=robot_config.MotorControlMode.HYBRID,
                  enable_action_interpolation=False,
                  reset_time=2,
                  time_step=0.002,
                  action_repeat=1)

    controller = _setup_controller(robot)

    controller.reset()
    if FLAGS.use_gamepad:
        print("Using gamepad")
        gamepad = gamepad_reader.Gamepad()
        command_function = gamepad.get_command
    else:
        command_function = _generate_example_linear_angular_speed

    if FLAGS.logdir:
        logdir = os.path.join(FLAGS.logdir,
                              datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(logdir)

    start_time = robot.GetTimeSinceReset()
    current_time = start_time
    base_position, base_rotation, base_vels, actions = [], [], [], []
    imu_rates, joint_angles, true_joint_angles, foot_positions = [], [], [], []

    for motor_name, motor_id in robot._joint_name_to_id.items():
        if motor_id in robot._motor_id_list:
            print(motor_id, motor_name)

    while current_time - start_time < FLAGS.max_time_secs:
        # time.sleep(0.0008) #on some fast computer, works better with sleep on real A1?
        start_time_robot = current_time
        start_time_wall = time.time()
        # Updates the controller behavior parameters.
        lin_speed, ang_speed, e_stop = command_function(current_time)
        # print(lin_speed)
        if e_stop:
            logging.info("E-stop kicked, exiting...")
            break

        _update_controller_params(controller, lin_speed, ang_speed)
        controller.update()

        hybrid_action, _ = controller.get_action()
        base_position.append(np.array(robot.GetBasePosition()).copy())
        # each orientation is a quaternion
        base_rotation.append(np.array(robot.GetBaseOrientation()).copy())
        base_vels.append(np.array(robot.GetBaseVelocity()).copy())
        imu_rates.append(np.array(robot.GetBaseRollPitchYawRate()).copy())
        joint_angles.append(np.array(robot.GetMotorAngles()).copy())
        # Foot positions are with respect to the base frame
        foot_positions.append(np.array(robot.GetFootPositionsInBaseFrame()).copy())

        true_joint_angles.append(
            np.array(robot.GetAllSensors()[2].get_observation()).copy())

        actions.append(hybrid_action)
        robot.Step(hybrid_action)
        current_time = robot.GetTimeSinceReset()

        expected_duration = current_time - start_time_robot
        actual_duration = time.time() - start_time_wall
        if actual_duration < expected_duration:
            time.sleep(expected_duration - actual_duration)
        print("actual_duration=", actual_duration)
        print("base position=", robot.GetBasePosition())

    if FLAGS.use_gamepad:
        gamepad.stop()

    if FLAGS.logdir:
        np.savez(os.path.join(logdir, 'action.npz'),
                 action=actions,
                 base_position=base_position,
                 base_rotation=base_rotation,
                 base_vels=base_vels,
                 imu_rates=imu_rates,
                 joint_angles=joint_angles,
                 true_joint_angles=true_joint_angles,
                 foot_positions=foot_positions)
        logging.info("logged to: {}".format(logdir))


if __name__ == "__main__":
    app.run(main)
