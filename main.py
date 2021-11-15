import os
import argparse
import gym
import torch
import rlbench.gym
from myLib.myCosas import execfile
from myLib.myCosas import myEnv
from stable_baselines3 import A2C, PPO, SAC, DDPG, TD3
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy, CnnPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

writer = SummaryWriter()

# Directories
# Create log dir
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--example", required=False, help="Run one of the examples:\n\t'panda_grab'\n\t'panda_gym'"
                                                        "\n\t'baselines_lunar'")
ap.add_argument("-t", "--target", required=False, help="Environment targets\n\t'reach_target-state-v0'")
ap.add_argument("-i", "--interface", required=False, help="Enviroment interface\n\t'human'\n\t'rgb'\n\t'headless'")
ap.add_argument("-m", "--model", required=False, help="Stable baselines3 model\n\t'A2C'\n\t'DQN'\n\t'DDPG'"
                                                      "\n\t'HER'\n\t'PPO'\n\t'SAC'\n\t'TD3'")
ap.add_argument("-n", "--name", required=False, help="Enter name of the model, if not existing, will be created")
ap.add_argument("-w", "--workout", required=False, help="Number of iterations to train the model")
ap.add_argument("-v", "--evaluate", required=False, help="Evaluate the policy unless is zero or none")
ap.add_argument("-rm", "--rewardmodel", required=False, help="choose one of the reward modes")
args = vars(ap.parse_args())
args.setdefault("target", "reach_target")
args.setdefault("interface", "rgb")
args.setdefault("model", "A2C")
args.setdefault("rewardmodel", "mode0")

reset_timesteps = False

if args.get('example') is not None:
    if args.get('example') == 'panda_grab':
        execfile("examples/pyrep_panda-reach-target.py")
    elif args.get('example') == 'panda_gym':
        execfile("examples/rlbench_gym.py")
    elif args.get('example') == 'baselines_lunar':
        execfile("examples/testlunar.py")
    else:
        print("Argument {} is not a valid example".format(args.get('example')))
else:
    if args.get('target') == 'reach_target':
        env_name = 'reach_target-state-v0'
    else:
        env_name = 'reach_target-state-v0'
        print("Argument {} is not a valid environment target, setting 'reach_target-state-v0'"
              .format(args.get('target')))

    if args.get('interface') == 'human':
        render_mode = 'human'
        observation_mode = 'state'
        policy_mode = 'MlpPolicy'
    elif args.get('interface') == 'rgb':
        render_mode = 'rgb_array'
        observation_mode = 'vision'
        policy_mode = 'CnnPolicy'
    elif args.get('interface') == 'headless':
        render_mode = None
        observation_mode = 'state'
        policy_mode = 'MlpPolicy'
    else:
        render_mode = 'rgb_array'
        observation_mode = 'vision'
        policy_mode = 'CnnPolicy'
        print("Argument {} is not a valid environment mode, setting to 'array'"
              .format(args.get('interface')))
    if args.get('rewardmodel') is None:
        reward_model = "mode0"
    elif args.get('rewardmodel') == 'mode0':
        reward_model = "mode1"
    elif args.get('rewardmodel') == 'mode1':
        reward_model = "mode1"
    elif args.get('rewardmodel') == 'mode2':
        reward_model = "mode2"
    elif args.get('rewardmodel') == 'mode3':
        reward_model = "mode3"
    else:
        reward_model = "mode0"
    if args.get('model') is None:
        model_name = 'A2C'
    else:
        model_name = args.get('model')

    # All ready to create environment
    print(env_name)
    print(render_mode)
    print(observation_mode)
    env = gym.make(env_name, render_mode=render_mode, observation_mode=observation_mode)
    # Wrapping custom environment to limit episode steps. Wrapper is in external
    # library. Also returns -1 when max limit has been reached.
    env = myEnv(env, mode=reward_model)

    name_model = env_name + '_' +  '_' + model_name + '_' + reward_model
    if model_name == 'A2C':
        try:
            # the saved model does not contain the replay buffer

            model = A2C.load(name_model, env)

            # load it into the loaded_model #NO en A2C, SI en SAC
            # model.load_replay_buffer(name_model + '_replay_buffer')
            # tb_log_name = model.num_timesteps + '_run'
            # now the loaded replay is not empty anymore
            print(f"The loaded_model has {model.num_timesteps} transitions in its buffer")

            # Load the policy independently from the model
            try:
                if policy_mode == 'CnnPolicy':
                    policy = CnnPolicy.load(name_model + '_policy')  # ojo. si es CNN es distinto
                    print("loaded cnn policy from " + name_model + '_policy\n')
                else:
                    policy = MlpPolicy.load(name_model + '_policy')  # ojo. si es CNN es distinto
                    print("loaded mlp policy from " + name_model + '_policy\n')
                model.policy = policy
            except:
                print("no nada")
            # Evaluate the loaded policy
            # mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=1, deterministic=True)

            # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

            # Retrieve the environment
            # env = model.get_env()

        except FileNotFoundError:
            print(f"File model not found: {name_model} creating new one")
            model = A2C(policy_mode, env, n_steps = 5, verbose=1, tensorboard_log="./tb_log_name/") 
            reset_timesteps = True
    elif model_name == 'PPO':
        try:
            model = PPO.load(name_model, env)
            print(f"The loaded_model has {model.num_timesteps} transitions in its buffer")
            # Load the policy independently from the model
            try:
                if policy_mode == 'CnnPolicy':
                    policy = CnnPolicy.load(name_model + '_policy')  # ojo. si es CNN es distinto
                    print("loaded cnn policy from " + name_model + '_policy\n')
                else:
                    policy = MlpPolicy.load(name_model + '_policy')
                    print("loaded mlp policy from " + name_model + '_policy\n')
                model.policy = policy
            except:
                print("no nada")
        except FileNotFoundError:
            print(f"File model not found: {name_model} creating new one")
            model = PPO(policy_mode, env, verbose=1, tensorboard_log="./tb_log_name/")
            reset_timesteps = True
    elif model_name == 'SAC':
        try:
            model = SAC.load(name_model, env)
            print(f"The loaded_model has {model.num_timesteps} transitions in its buffer")

            # Load the policy independently from the model
            try:
                if policy_mode == 'CnnPolicy':
                    policy = CnnPolicy.load(name_model + '_policy')  # ojo. si es CNN es distinto
                    print("loaded cnn policy from " + name_model + '_policy\n')
                else:
                    policy = MlpPolicy.load(name_model + '_policy')  # ojo. si es CNN es distinto
                    print("loaded mlp policy from " + name_model + '_policy\n')
                model.policy = policy
            except:
                print("no nada")
        except FileNotFoundError:
            print(f"File model not found: {name_model} creating new one")
            policy_kwargs = dict(net_arch=dict(pi=[64, 64, 64], qf=[256, 256]))
            #policy_kwargs = dict(net_arch=[128, 128, 128])
            #policy_kwargs = None
            model = SAC(policy_mode, env, verbose=1, tensorboard_log="./tb_log_name/", policy_kwargs=policy_kwargs)   #, buffer_size = 1000000, learning_starts=20000), batch_size=1000, train_freq = (1, "episode"))
            reset_timesteps = True
    elif model_name == 'TD3':
        try:
            model = TD3.load(name_model, env)
            print(f"The loaded_model has {model.num_timesteps} transitions in its buffer")

            # Load the policy independently from the model
            try:
                if policy_mode == 'CnnPolicy':
                    policy = CnnPolicy.load(name_model + '_policy')  # ojo. si es CNN es distinto
                    print("loaded cnn policy from " + name_model + '_policy\n')
                else:
                    policy = MlpPolicy.load(name_model + '_policy')  # ojo. si es CNN es distinto
                    print("loaded mlp policy from " + name_model + '_policy\n')
                model.policy = policy
            except:
                print("no nada")
        except FileNotFoundError:
            print(f"File model not found: {name_model} creating new one")
            model = TD3(policy_mode, env, verbose=1, tensorboard_log="./tb_log_name/")
            reset_timesteps = True
    elif model_name == 'DDPG':
        try:
            model = DDPG.load(name_model, env)

            print(f"The loaded_model has {model.num_timesteps} transitions in its buffer")

            # Load the policy independently from the model
            try:
                if policy_mode == 'CnnPolicy':
                    policy = CnnPolicy.load(name_model + '_policy')  # ojo. si es CNN es distinto
                    print("loaded cnn policy from " + name_model + '_policy\n')
                else:
                    policy = MlpPolicy.load(name_model + '_policy')
                    print("loaded mlp policy from " + name_model + '_policy\n')
                model.policy = policy
            except:
                print("no nada")
        except FileNotFoundError:
            print(f"File model not found: {name_model} creating new one")
            model = DDPG(policy_mode, env, verbose=1, tensorboard_log="./tb_log_name/")
            reset_timesteps = True
    else:
        print("Argument {} is not a valid model, setting to 'A2C'"
              .format(args.get('model')))
        try:
            model = A2C.load(name_model, env)
        except FileNotFoundError:
            model = A2C(policy_mode, env, verbose=1)

    if args.get('workout') is not None:
        # Evaluate the model every n steps
        # and save the evaluation to the "logs/" folder
        # model.set_env(env)
        #my_eval_env = gym.make(env_name, render_mode=render_mode, observation_mode=observation_mode)
        #my_eval_env = myEnv(env, mode=reward_model)
        final_steps = int(args.get('workout')) + model.num_timesteps
        # Si se interrumpe el entrenamiento por runtimeerror de coppelia, probar si funciona.
        while model.num_timesteps < final_steps:
            if reset_timesteps:
                print("No need to reset environment\n")
            else:
                model.env.reset()
            try:
                model.learn(int(args.get('workout')), eval_freq=10000, n_eval_episodes=5, eval_env = env, eval_log_path="./eval_logs_" + name_model + '_' + reward_model + '/', tb_log_name=model_name + '_' + reward_model, reset_num_timesteps=reset_timesteps)
            except RuntimeError as e:
                print("\n---------------------------\nRuntime error: {0}\n---------------------------\n".format(e))
                print("Saving model with {0} timesteps\n".format(model.num_timesteps))
                reset_timesteps = False
            # save the model
            model.save(name_model)


        # now save the replay buffer too No con A2C
        # model.save_replay_buffer(name_model + '_replay_buffer')

        # Save the policy independently from the model
        # Note: if you don't save the complete model with `model.save()`
        # you cannot continue training afterward
        policy = model.policy
        policy.save(name_model + '_policy')

    if args.get('evaluate') is not None:
        # Evaluate the policy evaluate_policy function not existing
        num_evals = int(args.get('evaluate'))
        if num_evals > 0:
            mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=num_evals, deterministic=True)
            print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    """
    # Pruebas
    import matplotlib.pyplot as plt
    env.render(mode=render_mode)
    obs = env.reset()
    plt.imshow(obs['wrist_rgb'])
    plt.show()
    img = obs['front_rgb']  # state, left_shoulder_rgb, right_shoulder_rgb
    plt.imshow(img)
    plt.show()"""

    env.close()
