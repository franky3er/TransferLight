import os

import torch

from src.params import SCENARIOS_ROOT
from src.rl.environments import MarlEnvironment
from src.rl.agents import MaxPressure, A2C, DQN

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    #scenarios_dir = os.path.join(SCENARIOS_ROOT, "grid", "1x1-500m", "train")
    #scenarios_dir = os.path.join(SCENARIOS_ROOT, "grid", "3x3-150m", "train")
    scenario_path = os.path.join(SCENARIOS_ROOT, "demo", "random-network", "1.sumocfg")
    #scenario_path = os.path.join(SCENARIOS_ROOT, "test", "ingolstadt21", "ingolstadt21.sumocfg")

    max_waiting_time = 900
    episodes = 20
    workers = 50
    hidden_dim = 64
    attention_heads = 8

    max_pressure_environment = MarlEnvironment(scenario_path=scenario_path, max_patience=max_waiting_time, problem_formulation="MaxPressureProblemFormulation", use_default=False, demo=True)
    generalight_environment = MarlEnvironment(scenario_path=scenario_path, max_patience=max_waiting_time, problem_formulation="GeneraLightProblemFormulation", use_default=False, demo=True)

    q_network = {
        "class_name": "GeneraLightNetwork",
        "init_args": {
            "network_type": "DuelingDQN",
            "hidden_dim": hidden_dim,
            "n_attention_heads": attention_heads,
            "dropout_prob": 0.1
        }
    }

    actor_critic_network = {
        "class_name": "GeneraLightNetwork",
        "init_args": {
            "network_type": "ActorCritic",
            "hidden_dim": hidden_dim,
            "n_attention_heads": attention_heads,
            "dropout_prob": 0.1
        }
    }


    #MaxPressure(20).demo(max_pressure_environment)
    A2C(actor_critic_network).demo(generalight_environment, checkpoint_path="agents/A2C/actor-critic-checkpoint.pt")
    #DQN(q_network, discount_factor=0.9, batch_size=64, replay_buffer_size=10_000, learning_rate=0.01, epsilon_greedy_prob=0.0, mixing_factor=0.01, update_steps=10).demo(generalight_environment, checkpoint_path="agents/DQN/dqn-checkpoint.pt")
