import torch
import argparse
import gym
import matplotlib.pyplot as plt
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size',type=int, default=128)
    parser.add_argument('-gamma', type=float, default=0.999)
    parser.add_argument('-eps_start', type=float, default=0.9)
    parser.add_argument('-eps_end', type=float, default=0.05)
    parser.add_argument('-eps_decay', type=int, default=200)
    parser.add_argument('-target_update', type=int, default=10)
    parser.add_argument('-env', type=str, default='CartPole-v0')

    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(opt.env).unwrapped


