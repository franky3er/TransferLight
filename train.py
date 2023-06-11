import os

from src.params import SCENARIOS_ROOT
from src.rl.environments import MultiprocessingMarlEnvironment, MarlEnvironment
from src.rl.agents import MaxPressure, A2C, DQN

if __name__ == "__main__":

    scenarios_dir = os.path.join(SCENARIOS_ROOT, "train", "coordinated")
    traffic_representation = "GeneraLightProblemFormulation"
    max_pressure_environment = MultiprocessingMarlEnvironment(scenarios_dir, 900, "MaxPressureProblemFormulation", 50)
    environment = MultiprocessingMarlEnvironment(scenarios_dir, 900, traffic_representation, 50)
    episodes = 20

    hidden_dim = 64

    model_params = {"state_size": 16, "hidden_size": 64, "action_size": 4}
    train_params = {"buffer_size": 1_000, "batch_size": 128, "learning_rate": 0.01, "discount_factor": 0.9,
                    "tau": 0.01, "eps_greedy_start": 1.0, "eps_greedy_end": 0.1, "eps_greedy_steps": 10_000}

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

    #MaxPressure(20).train_env(max_pressure_environment, episodes=episodes)
    A2C(network, actor_head, critic_head, share_network=True, entropy_loss_weight=0.1).train_env(environment, episodes=episodes, checkpoint_path="agents/A2C/actor-critic-checkpoint.pt")
    #DQN(network, dqn_head, discount_factor=0.9, batch_size=128, replay_buffer_size=10_000, learning_rate=0.01, eps_greedy_start=1.0, eps_greedy_end=0.1, eps_greedy_steps=1_000, tau=0.01).train_env(environment, episodes=episodes, checkpoint_path="agents/DQN/dqn-checkpoint.pt")
