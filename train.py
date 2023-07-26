import os

from src.params import SCENARIOS_ROOT
from src.rl.environments import MultiprocessingMarlEnvironment, MarlEnvironment
from src.rl.agents import MaxPressure, A2C, DQN

if __name__ == "__main__":

    scenarios_dir = os.path.join(SCENARIOS_ROOT, "train", "coordinated")
    max_waiting_time = 300
    steps = 1_000
    workers = 64
    hidden_dim = 64
    attention_heads = 8

    #max_pressure_environment = MultiprocessingMarlEnvironment(scenarios_dir, max_waiting_time,
    #                                                          "MaxPressureProblemFormulation", workers)
    generalight_environment = MultiprocessingMarlEnvironment(scenarios_dir, max_waiting_time,
                                                             "GeneraLightProblemFormulation", workers)

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

    #MaxPressure(20).train_env(max_pressure_environment, episodes=episodes)
    #A2C(actor_critic_network, entropy_loss_weight=0.0).train_env(generalight_environment, steps=steps, checkpoint_path="agents/A2C/actor-critic-checkpoint.pt")
    DQN(q_network, discount_factor=0.9, batch_size=64, replay_buffer_size=10_000, learning_rate=0.01, epsilon_greedy_prob=0.0, mixing_factor=0.01, update_steps=10).fit(generalight_environment, steps=steps, checkpoint_path="agents/DQN/dqn-checkpoint.pt")
