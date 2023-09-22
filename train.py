import os

from src.params import SCENARIOS_ROOT
from src.rl.environments import MultiprocessingMarlEnvironment, MarlEnvironment
from src.rl.agents import MaxPressure, A2C, DQN

if __name__ == "__main__":

    scenarios_dir = os.path.join(SCENARIOS_ROOT, "train", "random-all")
    #scenarios_dir = os.path.join(SCENARIOS_ROOT, "train", "fixed-all")
    max_waiting_time = 300
    steps = 10_000
    workers = 64
    hidden_dim = 64
    attention_heads = 8

    max_pressure_environment = MultiprocessingMarlEnvironment(scenarios_dir=scenarios_dir, max_patience=max_waiting_time,
                                                              problem_formulation="MaxPressureProblemFormulation",
                                                              n_workers=workers)
    #transferlight_environment = MultiprocessingMarlEnvironment(scenarios_dir=scenarios_dir, max_patience=max_waiting_time,
    #                                                           problem_formulation="TransferLightProblemFormulation",
    #                                                           n_workers=workers)
    #simple_transferlight_environment = MultiprocessingMarlEnvironment(scenarios_dir=scenarios_dir,
    #                                                           max_patience=max_waiting_time,
    #                                                           problem_formulation="SimpleTransferLightProblemFormulation",
    #                                                           n_workers=workers)


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

    MaxPressure(10).fit(max_pressure_environment, steps=steps)
    #A2C(actor_critic_network, entropy_loss_weight=0.001).fit(transferlight_environment, steps=steps, checkpoint_path="agents/A2C/transferlight-actor-critic-checkpoint.pt")
    #A2C(simple_actor_critic_network, entropy_loss_weight=0.0).fit(simple_transferlight_environment, steps=steps, checkpoint_path="agents/A2C/simple-transferlight-actor-critic-checkpoint.pt")
    #DQN(q_network, discount_factor=0.9, batch_size=64, replay_buffer_size=10_000, learning_rate=0.01, epsilon_greedy_prob=0.0, mixing_factor=0.01, update_steps=10).fit(generalight_environment, steps=steps, checkpoint_path="agents/DQN/dqn-checkpoint.pt")
