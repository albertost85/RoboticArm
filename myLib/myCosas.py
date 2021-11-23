import numpy as np
import gym
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback


class myEnv(gym.Wrapper):
    _max_steps_episode: int
    _step_counter: int
    _distance: float
    _delta_distance: float

    def __init__(self, env=None, max_steps_episode=500, mode="mode0"):
        super(myEnv, self).__init__(env)
        self._step_counter = 0
        self._max_steps_episode = max_steps_episode
        self._mode = mode
        self._distance = -1
        print("Wrapping the environment in a homemade step to do cosas model " + mode + "\n")

    def reset(self):
        obs = self.env.reset()
        # ¡¡¡ OJO !!! Ahora el objetivo estará siempre en la misma posición. Para facilitar (aún más) las cosas al robot.
        #self.task._task.target.set_position([0.35000000, 0.22000000, 0.97000000])
        obs, _, _, _ = self.env.step(np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]))
        try:
            self._distance = distance_cal(obs)
            print("Reset environment, initial distance is {0}\n".format(self._distance))
        except:
            print("Something went wrong on reset")
        return obs

    def step(self, action):
        self._step_counter += 1
        obs, reward, done, info = self.env.step(action)
        try:
            if done:
                print("Episodio resuelto en {0} steps\n".format(self._step_counter))
                self._step_counter = 0
            # mode0 returns -1 if goal is not achieved in max_steps_episode, and 1 if is achieved before.
            if self._mode == "mode0":
                if self._max_steps_episode > 0:
                    if self._step_counter >= self._max_steps_episode:
                        done = True
                        reward = -1
                        self._step_counter = 0
                        print("A cascarla, ha tardado mucho\n")
            # mode1 might or might not have an episode step limit, but returns -1 in each step while goal is not
            # accomplished
            if self._mode == "mode1":
                if not done:
                    reward = -1
                if self._max_steps_episode > 0:
                    if self._step_counter >= self._max_steps_episode:
                        done = True
                        self._step_counter = 0
                        print("A cascarla, ha tardado mucho\n")

            # mode2 might or might not have an episode step limit, but a reward proportional to the delta distance to
            # target. In every step, this reward is decreased. When finished, reward is 1. if doesn't reach the
            # target in the max_step, returns a negative reward tbd.
            if self._mode == "mode2":
                #new_distance = distance_cal(obs)
                #self._delta_distance = new_distance - self._distance
                #self._distance = new_distance
                if not done:
                    # reward = np.tanh(1/(distance+1e-5))
                    #reward = np.exp(-1 * distance_cal(obs))
                    reward = -distance_cal(obs) -1 # Penalización por cada step de más que tarde en resolver el episodio
                if self._max_steps_episode > 0:
                    if self._step_counter >= self._max_steps_episode:
                        done = True
                        self._step_counter = 0
                        print("A cascarla, ha tardado mucho\n")
            if self._mode == "mode3": # Devuelve un reward arbitrario e inmenso = 500 si resuelve el episodio
                if not done:
                    reward = -distance_cal(obs) -1 # Penalización por cada step de más que tarde en resolver el episodio
                else:
                    reward = 500
                if self._max_steps_episode > 0:
                    if self._step_counter >= self._max_steps_episode:
                        done = True
                        self._step_counter = 0
                        print("A cascarla, ha tardado mucho\n")
        except:
            print("Error on step")
        return obs, reward, done, info

class disabledRobot(gym.Wrapper):
    _disability: int
    def __init__(self, env=None, disability=0):
        super(disabledRobot, self).__init__(env)
        if disability >127:
            disability = 0
            print(f"Disability over {disability}, enter a 7 bits binary number")
        if disability > 0:
            print(f"Robot disabled {disability} joints")
        self._disability = disability
    def step(self, action):
        for i, act in enumerate(action):
            if 2**i & self._disability > 0:
                action[i] = 0.0
        #print(f"DEBUG: robot new actions {action}")
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


def averaverqueyolovea(env, model, max_steps=1000):
    episode_rewards = [0.0]
    steps = 0

    obs = env.reset()
    done = False
    while (not done) and (steps < max_steps):
        steps = steps + 1
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards[-1] += reward
        env.render()
    return done, episode_rewards


def evaluate_1(env, model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]

    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        # Stats
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


def execfile(filepath, my_globals=None, my_locals=None):
    if my_globals is None:
        my_globals = {}
    my_globals.update({
        "__file__": filepath,
        "__name__": "__main__",
    })
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), my_globals, my_locals)

def distance_sqrt_cal(obs):
    '''calculate distance between end effector and target'''
    ee_pose = np.array(obs[22:25])
    target_pose = np.array(obs[-3:])

    distance_sqrt = (target_pose[0]-ee_pose[0])**2 + (target_pose[1]-ee_pose[1])**2 + (target_pose[2]-ee_pose[2])**2

    # reward = np.tanh(1/(distance+1e-5))
    # reward = np.exp(-1*distance)

    return distance_sqrt

def distance_cal(obs):
    '''calculate distance between end effector and target'''
    ee_pose = np.array(obs[22:25])
    target_pose = np.array(obs[-3:])

    distance = np.sqrt((target_pose[0]-ee_pose[0])**2 +
                       (target_pose[1]-ee_pose[1])**2 + (target_pose[2]-ee_pose[2])**2)

    # reward = np.tanh(1/(distance+1e-5))
    # reward = np.exp(-1*distance)

    return distance
