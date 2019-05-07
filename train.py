import argparse
import gym
import math
import random
import matplotlib
import torch.optim as optim
from DQN import DQN
from torch.functional import F
from ReplayMemory import *
from utils import *
from itertools import count
import matplotlib.pyplot as plt

def select_action(state,opt, device):
    sample = random.random()
    eps_threshold = opt.eps_end + (opt.eps_start - opt.eps_end) * \
        math.exp(-1. * opt.steps_done / opt.eps_decay)
    opt.steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return opt.policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(opt.n_actions)]], device=device, dtype=torch.long)

def optimize_model(opt):
    if len(opt.memory) < opt.batch_size:
        return
    transitions = opt.memory.sample(opt.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = opt.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(opt.batch_size, device=device)
    next_state_values[non_final_mask] = opt.target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * opt.gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    opt.optimizer.zero_grad()
    loss.backward()
    for param in opt.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    opt.optimizer.step()

def plot_durations(episode_durations):
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def train(env, device, opt):

    for i_episode in range(opt.num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env,device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state, opt, device)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env,device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            opt.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(opt)
            if done:
                opt.episode_durations.append(t + 1)
                if i_episode % opt.plot_every == 0:
                    plot_durations(opt.episode_durations)
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % opt.target_update == 0:
            opt.target_net.load_state_dict(opt.policy_net.state_dict())

    torch.save({"settings": opt, "state_dict" : opt.policy_net.state_dict()}, "policy_net.pt")
    torch.save(opt.policy_net.state_dict(), "target_net.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size',type=int, default=128)
    parser.add_argument('-num_episodes', type=int, default=10)
    parser.add_argument('-gamma', type=float, default=0.999)
    parser.add_argument('-eps_start', type=float, default=0.9)
    parser.add_argument('-eps_end', type=float, default=0.05)
    parser.add_argument('-eps_decay', type=int, default=200)
    parser.add_argument('-target_update', type=int, default=10)
    parser.add_argument('-plot_every', type=int, default=5)
    parser.add_argument('-env', type=str, default='CartPole-v0')


    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(opt.env).unwrapped
    env.reset()
    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    init_screen = get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    opt.n_actions = env.action_space.n

    opt.policy_net = DQN(screen_height, screen_width, opt.n_actions).to(device)
    opt.target_net = DQN(screen_height, screen_width, opt.n_actions).to(device)
    opt.target_net.load_state_dict(opt.policy_net.state_dict())
    opt.target_net.eval()

    opt.optimizer = optim.RMSprop(opt.policy_net.parameters())
    opt.memory = ReplayMemory(10000)
    opt.episode_durations = []

    opt.steps_done = 0

    train(env,device, opt)

    print('Complete')



