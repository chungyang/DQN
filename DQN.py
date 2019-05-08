import math, random
import gym
import torch.optim as optim
import argparse

from torch import nn
from ReplayBuffer import ReplayBuffer
from utils import *


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)
        return action


def train(env, num_frames, model, gamma, replay_buffer_size, batch_size):
    losses = []
    all_rewards = []
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(replay_buffer_size)

    state = env.reset()
    episode_reward = 0

    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            print(episode_reward)
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(model, optimizer, batch_size, gamma, replay_buffer)
            losses.append(loss.item())

        if frame_idx % 200 == 0:
            plot(frame_idx, all_rewards, losses)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_frames',  default=10000)
    parser.add_argument('-replay_buffer_size', default=1000)
    parser.add_argument('-batch_size', default=32)
    parser.add_argument('-gamma', default=0.99)

    opt = parser.parse_args()

    env_id = "CartPole-v0"
    env = gym.make(env_id)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    model = DQN(env.observation_space.shape[0], env.action_space.n)

    if USE_CUDA:
        model = model.cuda()


    num_frames = 10000
    replay_buffer_size = 10000
    batch_size = 32
    gamma = 0.99

    train(env, opt.num_frames, model, opt.gamma, opt.replay_buffer_size, opt.batch_size)