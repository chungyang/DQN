import torch
import gym
from DQN_cartpole import DQN

if __name__ == "__main__":
    settings = torch.load("./pretrained/cartpole_states.pt")
    model = DQN(settings["n_input"], settings["n_action"])
    model.load_state_dict(settings["states"])
    model.eval()

    env_id = "CartPole-v0"
    env = gym.make(env_id)

    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        action = model.act(state, -1)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    print(total_reward)

