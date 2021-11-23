import os
import sys
import gym
import torch
import rlbench.gym
from os.path import isfile, join
from pyfiglet import Figlet
from PyInquirer import style_from_dict, Token, prompt
from examples import custom_style_2
from prompt_toolkit.validation import Validator, ValidationError
from myLib.myCosas import myEnv, disabledRobot
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy, CnnPolicy

class NumberValidator(Validator):

    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(message="Please enter a number",
                                  cursor_position=len(document.text))

class GymModelValidator(Validator):
    def validate(self, document):
        pass
mypath = os.path.realpath('.')
mypath = os.path.join(mypath, 'SAC_models')

def return_files():
    return [f for f in os.listdir(mypath) if (not f.endswith("buffer")) and (not f.endswith("policy") and isfile(join(mypath, f)))]



class NumberValidator(Validator):

    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(message="Please enter a number",
                                  cursor_position=len(document.text))

mystyle = style_from_dict({
    Token.Separator: '#cc5454',
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
})

questions = [
    {
        'type': 'list',
        'name': 'model_open',
        'message': 'Choose the model',
        'choices': ["New one"] + return_files()
    },
    {
        'type': 'input',
        'name': 'custom_name',
        'message': 'Enter custom name?',
        'validate': GymModelValidator
    },
    {
        'type': 'input',
        'name': 'model_workout',
        'message': 'How many workout steps?',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    },
]

def main():
    f = Figlet(font='slant')
    print(f.renderText('Disabled Robot'))
    env_name = 'reach_target-state-v0'
    render_mode = None
    observation_mode = 'state'
    policy_mode = 'MlpPolicy'    
    env = gym.make('reach_target-state-v0', render_mode=None, observation_mode='state')
    env = myEnv(env, mode='mode2')
    env = disabledRobot(env, 4)
    answers = prompt(questions, style=mystyle)
    my_name = answers.get("custom_name")
    route_to_save  = os.path.join(mypath, answers.get("custom_name")) 
    if answers.get("model_open") == "New one":
        route_to_model = os.path.join(mypath, answers.get("custom_name"))
        policy_kwargs = dict(net_arch=dict(pi=[64, 64, 64,64], qf=[128, 128, 128]))
        reset_timesteps = True
        model = SAC(policy_mode, env,
                learning_rate=0.0003,
                learning_starts=20000,
                buffer_size=1000000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                tensorboard_log="./tb_log_name/",
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=0
                )  # train_freq = (1, "episode"))
        print(f"Creating fresh model {route_to_model}")
    else:
        route_to_model = os.path.join(mypath, answers.get("model_open"))
        reset_timesteps = False
        try:
            model = SAC.load(route_to_model, env)
            print(f"Opened model in with {model.num_timesteps} transitions in its buffer")
        except:
            print(f"File model not found: {route_to_model}")
    # Estimación de episodios que debería tener
    final_steps = int(answers.get("model_workout")) + model.num_timesteps
    while model.num_timesteps < final_steps:
        if reset_timesteps:
            print("No need to reset environment\n")
        else:
            model.env.reset()
        try:
            model.learn(int(answers.get("model_workout")), eval_freq=10000, n_eval_episodes=5, eval_env = env, eval_log_path=f"./SAC_eval_logs/{my_name}/", tb_log_name=answers.get("custom_name"), reset_num_timesteps=reset_timesteps)
        except RuntimeError as e:
            print("\n---------------------------\nRuntime error: {0}\n---------------------------\n".format(e))
            print("Saving model with {0} timesteps\n".format(model.num_timesteps))
            reset_timesteps = False
        # save the model
        print(f"Saving model in {route_to_save}")
        model.save(route_to_save)
    
    # close environment
    env.close()
    print(f"C'est fini. Total timesteps={model.num_timesteps}")

    
if __name__ == "__main__":
    main()
