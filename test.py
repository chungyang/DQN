import torch
import gym
import argparse
from utils import *
from DQN import DQN
from itertools import count
from train import select_action



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='CartPole-v0')
    parser.add_argument('-policy_net', type=str, default='policy_net.pt')
    parser.add_argument('-target_net', type=str, default='policy_net.pt')


    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(opt.env).unwrapped

    env.reset()

    init_screen = get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    saved_model = torch.load('policy_net.pt')
    policy_net.load_state_dict(saved_model['state_dict'])
    policy_net.eval()

    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(torch.load('target_net.pt'))
    target_net.eval()

    env.reset()
    last_screen = get_screen(env, device)
    current_screen = get_screen(env, device)
    state = current_screen - last_screen
    opt = saved_model['settings']
    opt.steps_done = 0

    for t in count():
        action = select_action(state, opt, device)
        _, _, done, _ = env.step(action.item())

        last_screen = current_screen
        current_screen = get_screen(env, device)

        if not done:
            next_state = current_screen - last_screen
        else:
            break


