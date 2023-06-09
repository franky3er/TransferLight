import os

import torch

from src.params import SCENARIOS_ROOT
from src.rl.environments import MarlEnvironment
from src.rl.agents import MaxPressure, A2C, DQN

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    #scenarios_dir = os.path.join(SCENARIOS_ROOT, "grid", "1x1-500m", "train")
    #scenarios_dir = os.path.join(SCENARIOS_ROOT, "grid", "3x3-150m", "train")
    scenarios_dir = os.path.join(SCENARIOS_ROOT, "train", "coordinated")

    #environment = MarlEnvironment(scenarios_dir, 900, "MaxPressureProblemFormulation", use_default=False, demo=True)
    environment = MarlEnvironment(scenarios_dir, 500, "GeneraLightProblemFormulation", use_default=False, demo=True)

    input_dim = 10
    hidden_dim = 64

    network = {
        "class_name": "GeneraLightNetwork",
        "init_args": {
            "output_dim": hidden_dim
        }
    }
    actor_head = {
        "class_name": "ActorHead",
        "init_args": {
            "agent_dim": hidden_dim,
            "action_dim": hidden_dim
        }
    }
    critic_head = {
        "class_name": "CriticHead",
        "init_args": {
            "agent_dim": hidden_dim,
            "action_dim": hidden_dim
        }
    }
    dqn_head = {
        "class_name": "DuelingHead",
        "init_args": {
            "agent_dim": hidden_dim,
            "action_dim": hidden_dim
        }
    }


    #agents = MaxPressure(20)
    agents = A2C(network, actor_head, critic_head, share_network=True, checkpoint_dir=os.path.join("agents", "A2C"), load_checkpoint=True)
    #agents = DQN(network, dqn_head, discount_factor=0.9, batch_size=128, replay_buffer_size=10_000, learning_rate=0.001,
    #             eps_greedy_start=1.0, eps_greedy_end=0.1, eps_greedy_steps=10_000, tau=0.01,
    #             checkpoint_dir=os.path.join("agents", "DQN"), load_checkpoint=True)
    agents.demo_env(environment)

