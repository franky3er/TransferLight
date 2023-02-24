import os

import torch
import torch_geometric

from src.params import SCENARIOS_ROOT
from src.rl.environments import TscMarlEnvironment
from src.rl.agents import MaxPressureAgent, LitAgent, HieraGLightAgent


if __name__ == "__main__":
    print(torch_geometric.__version__)
    scenarios_dir = os.path.join(SCENARIOS_ROOT, "grid", "5x5-150m", "train")
    environment = TscMarlEnvironment(scenarios_dir, 180, "HieraGLightTrafficRepresentation",
                                     use_default=False, demo=False)
    #agent = MaxPressureAgent(20)
    #agent = LitAgent(16, 64, 4, 1000, 128, 0.01, 0.99, 0.01, 1.0, 0.1, 10_000)
    agent = HieraGLightAgent(2, 1, 32)
    for i in range(100):
        state = environment.reset()
        while True:
            actions = agent.act(state)
            next_state, rewards, done = environment.step(actions)
            agent.train_step(state, actions, rewards, next_state, done)
            state = next_state
            if done:
                print(f"--- Epoch: {i}   Reward: {torch.mean(rewards)} ---")
                break
    environment.close()
