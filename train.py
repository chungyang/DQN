import torch
import gym
import matplotlib.pyplot as plt
from utils import *


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v0').unwrapped


    env.reset()
    plt.figure()
    plt.imshow(get_screen(env,device).squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


