import os

import torch

from src.params import SCENARIOS_ROOT
from src.rl.environments import MarlEnvironment
from src.rl.agents import MaxPressure, A2C, DQN

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    #scenarios_dir = os.path.join(SCENARIOS_ROOT, "grid", "1x1-500m", "train")
    scenario_path = os.path.join(SCENARIOS_ROOT, "grid", "3x3-150m", "train", "0-scenario", "0.sumocfg")
    #scenario_path = "./scenarios/train/random-network/0851.sumocfg"
    #scenario_path = os.path.join(SCENARIOS_ROOT, "demo", "random-network", "1.sumocfg")
    #scenario_path = os.path.join(SCENARIOS_ROOT, "train", "random-location", "0001.sumocfg")
    #scenario_path = os.path.join(SCENARIOS_ROOT, "train", "random-location", "0001.sumocfg")
    #scenario_path = os.path.join(SCENARIOS_ROOT, "demo", "fixed-all", "1.sumocfg")
    #scenario_path = os.path.join(SCENARIOS_ROOT, "test", "cologne8", "cologne8.sumocfg")
    #scenario_path = os.path.join(SCENARIOS_ROOT, "test", "sumo-rl", "Nguyen", "nguyen.sumocfg")

    max_waiting_time = 900
    episodes = 20
    workers = 50
    hidden_dim = 64
    attention_heads = 8

    max_pressure_environment = MarlEnvironment(scenario_path=scenario_path, max_patience=max_waiting_time, problem_formulation="MaxPressureProblemFormulation", use_default=False, demo=True)
    transferlight_environment = MarlEnvironment(scenario_path=scenario_path, max_patience=max_waiting_time, problem_formulation="TransferLightProblemFormulation", use_default=False, demo=True)
    simple_transferlight_environment = MarlEnvironment(scenario_path=scenario_path, max_patience=max_waiting_time, problem_formulation="SimpleTransferLightProblemFormulation", use_default=False, demo=True)

    q_network = {
        "class_name": "TransferLightNetwork",
        "init_args": {
            "network_type": "QNet",
            "hidden_dim": hidden_dim,
            "n_attention_heads": attention_heads,
            "dropout_prob": 0.1
        }
    }
    q_network = {
        "class_name": "SimpleTransferLightNetwork",
        "init_args": {
            "network_type": "QNet",
            "hidden_dim": hidden_dim,
            "n_attention_heads": attention_heads,
            "dropout_prob": 0.1
        }
    }

    actor_critic_network = {
        "class_name": "TransferLightNetwork",
        "init_args": {
            "network_type": "ActorCriticNet",
            "hidden_dim": hidden_dim,
            "n_attention_heads": attention_heads,
            "dropout_prob": 0.1
        }
    }
    simple_actor_critic_network = {
        "class_name": "SimpleTransferLightNetwork",
        "init_args": {
            "network_type": "ActorCriticNet",
            "hidden_dim": hidden_dim,
            "n_attention_heads": attention_heads,
            "dropout_prob": 0.1
        }
    }


    #MaxPressure(10).demo(max_pressure_environment)
    #A2C(actor_critic_network).demo(transferlight_environment, checkpoint_path="agents/A2C/transferlight-actor-critic-checkpoint.pt")
    A2C(simple_actor_critic_network).demo(simple_transferlight_environment, checkpoint_path="agents/A2C/simple-transferlight-actor-critic-checkpoint.pt")
    #DQN(q_network, discount_factor=0.9, batch_size=64, replay_buffer_size=10_000, learning_rate=0.01, epsilon_greedy_prob=0.0, mixing_factor=0.01, update_steps=10).demo(generalight_environment, checkpoint_path="agents/DQN/dqn-checkpoint.pt")
