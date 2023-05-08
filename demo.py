import os

import torch
import torch_geometric

from src.params import SCENARIOS_ROOT
from src.rl.environments import TscMarlEnvironment
from src.rl.agents import MaxPressureAgents, IQLAgents, HieraGLightAgent, IA2CAgents

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    scenarios_dir = os.path.join(SCENARIOS_ROOT, "grid", "1x1-500m", "train")
    #environment = TscMarlEnvironment(scenarios_dir, 180, "MaxPressureTrafficRepresentation", use_default=False, demo=True)

    environment = TscMarlEnvironment(scenarios_dir, 500, "TestTrafficRepresentation", use_default=False, demo=True)

    #agents = MaxPressureAgents(20)
    qnet_params = {"state_size": 48, "hidden_size": 64, "action_size": 4}
    #agents = IQLAgents(qnet_params, 1000, 128, 0.01, 0.9, 0.01, 1.0, 0.1, 10_000, checkpoint_dir=os.path.join("agents", "IQL"), load_checkpoint=True)
    agents = IA2CAgents(52, 64, 4, 0.001, 0.001, 0.9, 0.5, checkpoint_dir=os.path.join("agents", "IA2C"), load_checkpoint=True)

    agents.demo(environment)

