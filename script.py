from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import gym
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
import matplotlib.pyplot as plt

class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.gripper_touch_forces = False
action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, obs_config=obs_config, headless=False,
    robot_configuration='sawyer')
env.launch()

env.task = env.get_task(ReachTarget)

action_size=env.action_size
print(f"Action size:{action_size}")

description, obs = env.task.reset()
print(description)
print(obs)

cam_placeholder = Dummy('cam_cinematic_placeholder')
env._gym_cam = VisionSensor.create([640, 360])
env._gym_cam.set_pose(cam_placeholder.get_pose())