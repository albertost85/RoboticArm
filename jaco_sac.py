import os
import argparse
import gym
import rlbench.gym
from myLib.myCosas import myEnv
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy, CnnPolicy


# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()

# Positional arguments
ap.add_argument("workout", help="number of workout steps to train",
                type=int)

# Optional arguments
ap.add_argument("-t", "--task", required=False, choices=["reach_target"],
                default="reach_target",
                help="Task to be performed.")
ap.add_argument("-i", "--interface", required=False, choices=["human", "rgb", "headless"],
                default="headless",
                help="Enviroment interface.")
ap.add_argument("-rm", "--rewardmodel", required=False, choices=["mode0", "mode1", "mode2", "mode3"],
                default="mode2",
                help="choose one of the reward modes")
ap.add_argument("-nm", "--networkmodel", required=False, type=int, choices=[0, 1, 2, 3, 4, 5],
                default=0,
                help="Choose one of the network models. Different models for CNN and MLP. 0 is default. Check file for more info.")
ap.add_argument("-ls", "--learningstarts", required=False, type=int,
                default=20000,
                help="How many steps of the model to collect transitions for before learning starts.")
ap.add_argument("-lr", "--learningrate", required=False, type=float,
                default=0.0003,
                help="Learning rate for optimizer, the same learning rate will be used for all networks.")
ap.add_argument("-ga", "--gamma", required=False, type=float,
                default=0.99,
                help="The discount factor")
ap.add_argument("-ta", "--tau", required=False, type=float,
                default=0.005,
                help="The soft update coefficient, between 0 and 1")
ap.add_argument("-s", "--seed", required=False, type=int,
                default=0,
                help="Seed for the pseudo random generators.")
ap.add_argument("-bu", "--buffersize", required=False, type=int,
                default=1000000,
                help="Size of the replay buffer.")
ap.add_argument("-ba", "--batchsize", required=False, type=int,
                default=256,
                help="Minibatch size for each gradient update.")

args = vars(ap.parse_args())

if args.get('task') == 'reach_target':
    env_name = 'reach_target-state-v0'

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

# Network choices based on policy mode are defined here
if policy_mode == 'CnnPolicy':
        policy_kwargs = None
        policy_name = 'Default'
elif policy_mode == 'MlpPolicy':
    if args.get('networkmodel') == 0:
        policy_kwargs = None
        policy_name = 'Default'
    elif args.get('networkmodel') == 1:
        policy_kwargs = dict(net_arch=[128, 128, 128])
        policy_name = 'net=128e3'
    elif args.get('networkmodel') == 2:
        policy_kwargs = dict(net_arch=[64, 64, 64])
        policy_name = 'net=64e3'
    elif args.get('networkmodel') == 3:
        policy_kwargs = dict(net_arch=[256, 256, 256])
        policy_name = 'net=256e3'
    elif args.get('networkmodel') == 4:
        policy_kwargs = dict(net_arch=dict(pi=[64, 64, 64], qf=[256, 256]))
        policy_name = 'pi=64e3-qf=256e2'
    elif args.get('networkmodel') == 5:
        policy_kwargs = dict(net_arch=dict(pi=[64, 64, 64, 64], qf=[128, 128, 128]))
        policy_name = 'pi=64e4-qf=128e3'




reward_model = args.get('rewardmodel')

name_experiment = f"jaco_SAC_{policy_mode}[{policy_name}]_rw={reward_model}_" \
                  f"lr={args.get('learningrate')}_starts={args.get('learningstarts')}_" \
                  f"buffer={args.get('buffersize')}_batch={args.get('batchsize')}_" \
                  f"tau={args.get('tau')}_gamma={args.get('gamma')}_seed={args.get('seed')}"
name_short = f"jaco_SAC_{policy_mode}[{policy_name}]_{reward_model}_seed={args.get('seed')}"
# Summary writer for tensorboard
writer = SummaryWriter()

# Directories
# Create log dir
log_dir = "SAC_eval_logs"
model_dir = "SAC_models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# All ready to create environment
env = gym.make(env_name, render_mode=render_mode, observation_mode=observation_mode, robot='jaco')

# Wrapping custom environment to limit episode steps. Wrapper is in external
# library. Also returns -1 when max limit has been reached.
env = myEnv(env, mode=reward_model)

print(name_short)



# Create Reinforcement Learning model
reset_timesteps = False
try:
    model = SAC.load(f"{model_dir}/{name_experiment}", env)
    print(f"The loaded_model has {model.num_timesteps} transitions in its buffer")
    # Load replay buffer
    try:
        model.load_replay_buffer(f"{model_dir}/{name_experiment}_replay_buffer")
        print(f"Loaded buffer from {model_dir}/{name_experiment}_replay_buffer")
    except:
        print(f"Something went wrong while loading buffer {model_dir}/{name_experiment}_replay_buffer from disk")
    # Load the policy independently from the model
    try:
        if policy_mode == 'CnnPolicy':
            policy = CnnPolicy.load(f"{model_dir}/{name_experiment}_policy")  # ojo. si es CNN es distinto
            print(f"Loaded CNN policy from {model_dir}/{name_experiment}_policy")
        else:
            policy = MlpPolicy.load(f"{model_dir}/{name_experiment}_policy")   # ojo. si es MLP es distinto
            print(f"Loaded MLP policy from {model_dir}/{name_experiment}_policy")
        model.policy = policy
    except:
        print(f"Something went wrong while loading policy {policy_mode} from disk")
except FileNotFoundError:
    print(f"File model not found: {name_experiment} creating new one")
    reset_timesteps = True
    model = SAC(policy_mode, env,
                learning_rate=args.get('learningrate'),
                learning_starts=args.get('learningstarts'),
                buffer_size=args.get('buffersize'),
                batch_size=args.get('batchsize'),
                tau=args.get('tau'),
                gamma=args.get('gamma'),
                tensorboard_log="./tb_log_name/",
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=args.get('seed')
                )  # train_freq = (1, "episode"))

# Estimación de episodios que debería tener
final_steps = int(args.get('workout')) + model.num_timesteps

# Si se interrumpe el entrenamiento por runtimeerror de coppelia, probar si funciona.
while model.num_timesteps < final_steps:
    if reset_timesteps:
        print("No need to reset environment\n")
    else:
        model.env.reset()
    try:
        model.learn(args.get('workout'), eval_freq=10000, n_eval_episodes=5, eval_env = env, eval_log_path=f"./{log_dir}/{name_short}/", tb_log_name=name_short, reset_num_timesteps=reset_timesteps)
    except RuntimeError as e:
        print("\n---------------------------\nRuntime error: {0}\n---------------------------\n".format(e))
        print("Saving model with {0} timesteps\n".format(model.num_timesteps))
        reset_timesteps = False
    # save the model
    model.save(f"{model_dir}/{name_experiment}")


    # now save the replay buffer too
    model.save_replay_buffer(f"{model_dir}/{name_experiment}_replay_buffer")

    # Save the policy independently from the model
    # Note: if you don't save the complete model with `model.save()`
    # you cannot continue training afterward
    policy = model.policy
    policy.save(f"{model_dir}/{name_experiment}_policy")

# close environment
env.close()
print(f"C'est fini. Total timesteps={model.num_timesteps}")
