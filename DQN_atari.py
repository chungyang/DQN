import math, random
import torch.nn as nn
import torch.optim as optim
import argparse

from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from utils import *
from ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions )
        return action




def train(env, model, opt):
    losses = []
    all_rewards = []
    episode_reward = 0

    optimizer = optim.Adam(model.parameters(), opt.lr)
    if opt.prioritize:
        replay_buffer = PrioritizedReplayBuffer(opt.replay_buffer_size, alpha=0.6)
    else:
        replay_buffer = ReplayBuffer(opt.replay_buffer_size)

    state = env.reset()
    episode_n = 0
    for frame_idx in range(1, opt.num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            print(episode_n, " ", episode_reward)
            episode_n += 1
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > opt.replay_initial:
            loss = compute_td_loss(model, optimizer, opt.batch_size, opt.gamma, replay_buffer, prioritize=opt.prioritize)
            losses.append(loss.item())

        # if frame_idx % 100000 == 0:
        #     plot(frame_idx, all_rewards)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_frames',  default=1400000)
    parser.add_argument('-replay_buffer_size', default=100000)
    parser.add_argument('-replay_initial', default=10000)
    parser.add_argument('-batch_size', default=32)
    parser.add_argument('-gamma', default=0.99)
    parser.add_argument('-lr', default=0.00001)
    parser.add_argument('-prioritize', default=True)



    opt = parser.parse_args()

    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    model = CnnDQN(env.observation_space.shape, env.action_space.n)

    if USE_CUDA:
        model = model.cuda()

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    train(env,model,opt)

    torch.save({"n_input":env.observation_space.shape,
                "n_action":env.action_space.n,
                "states":model.load_state_dict()},
                "/pretrained/atari_states.pt")

