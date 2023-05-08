import os

import torch

from src.params import SCENARIOS_ROOT
from src.rl.environments import TscMarlEnvironment, MultiprocessingTscMarlEnvironment
from src.rl.agents import MaxPressureAgents, IQLAgents, HieraGLightAgent, IA2CAgents, RandomAgents
from src.params import DEVICE as device


if __name__ == "__main__":
    scenarios_dir = os.path.join(SCENARIOS_ROOT, "grid", "1x1-500m", "train")
    traffic_representation = "HieraGLightTrafficRepresentation"
    #environment = TscMarlEnvironment(scenarios_dir, 180, "MaxPressureTrafficRepresentation", use_default=False, demo=False)
    environment = MultiprocessingTscMarlEnvironment(scenarios_dir, 180, traffic_representation, 1)

    model_params = {"state_size": 52, "hidden_size": 64, "action_size": 4}
    train_params = {"buffer_size": 1_000, "batch_size": 128, "learning_rate": 0.01, "discount_factor": 0.9,
                    "tau": 0.01, "eps_greedy_start": 1.0, "eps_greedy_end": 0.1, "eps_greedy_steps": 10_000}

    actor = {
        "class_name": "GraphActorNetwork",
        "init_args": {
            "movement_dim": 6,
            "phase_dim": 1,
            "hidden_dim": 64
        }
    }
    critic = {
        "class_name": "GraphCriticNetwork",
        "init_args": {
            "movement_dim": 6,
            "phase_dim": 1,
            "intersection_dim": 1,
            "hidden_dim": 64
        }
    }

    #agents = RandomAgents()
    #agents = MaxPressureAgents(20)
    #agents = IQLAgents(model_params, 1000, 128, 0.01, 0.9, 0.01, 1.0, 0.1, 10_000, checkpoint_dir=os.path.join("agents", "IQL"), save_checkpoint=True)

    agents = IA2CAgents(actor, critic, 0.001, 0.001, 0.9, 0.01, checkpoint_dir=os.path.join("agents", "IA2C"), save_checkpoint=True)
    #agent = LitAgent(16, 64, 4, 1000, 128, 0.01, 0.99, 0.01, 1.0, 0.1, 10_000)
    #agent = HieraGLightAgent(2, 32, 1000, 128, 0.01, 0.9, 0.01, 1.0, 0.1, 10_000)
    agents.train(environment)
